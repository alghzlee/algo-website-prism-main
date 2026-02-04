from scipy.stats import norm
import torch
import numpy as np
import os
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from torch.distributions import Normal
from SAC_deepQnet_bestexp import SACAgent, EnsembleSAC
from BehaviourCloning_model import BehaviorCloning


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_latent(model, state_tensor):
    """
    Encode state using AutoEncoder if available.
    If AE is not present or disabled, return raw state.

    Supports:
    - EnsembleSAC (model.agents)
    - Single SACAgent
    - state shape (B, D) or (B, T, D)
    """

    # ---- 1. Resolve AE safely ----
    ae = None

    if hasattr(model, 'agents') and len(model.agents) > 0:
        ae = getattr(model.agents[0], 'autoencoder', None)
    else:
        ae = getattr(model, 'autoencoder', None)

    # ---- 2. No AE → return raw state ----
    if ae is None:
        return state_tensor

    # ---- 3. AE exists → encode ----
    ae.eval()

    with torch.no_grad():
        # Handle sequence input (B, T, D)
        if state_tensor.ndim == 3:
            state_tensor = state_tensor[:, -1, :]

        return ae.encode(state_tensor)


def get_ensemble_log_prob(agents, latent_state, action_target, device=device):
    """
    Computes the Log Probability of an action under the ENSEMBLE policy 
    using the Numerically Stable Jacobian Correction.
    
    Formula: log(1/N * sum( exp(log_pi_i) ) )
    """
    latent_state = latent_state.to(device)
    action_target = action_target.to(device)
    
    log_probs_list = []

    # 

    for agent in agents:
        actor = agent.actor
        with torch.no_grad():
            # 1. Get Gaussian parameters (mean, std) for the latent 'u'
            mean, std = actor(latent_state) 
            dist = Normal(mean, std)
            
            # 2. Recover 'u' (inverse tanh)
            # We clamp to 1-1e-6 to prevent atanh(1) = inf
            action_clipped = torch.clamp(action_target, -0.999999, 0.999999)
            u = torch.atanh(action_clipped)
            
            # 3. Log Prob in Gaussian Space
            log_prob_u = dist.log_prob(u)
            
            # 4. Numerically Stable Jacobian Correction
            # Instead of log(1 - a^2), we use the derived formula for u:
            # log(1 - tanh(u)^2) = 2 * (log(2) - u - softplus(-2u))
            # This avoids catastrophic cancellation when a ~= 1
            correction = 2 * (np.log(2) - u - F.softplus(-2 * u))
            
            # 5. Final Log Prob for this agent
            # log_pi(a) = log_pi(u) - log(det(da/du))
            log_prob = (log_prob_u - correction).sum(dim=1)
            
            log_probs_list.append(log_prob)

    # 6. Compute LogSumExp for numerical stability (Ensemble Mixture)
    # Formula: log(1/N * sum(probs)) = log(sum(exp(log_probs))) - log(N)
    log_probs_stack = torch.stack(log_probs_list, dim=0) # [Num_Agents, Batch_Size]
    
    mixture_log_prob = torch.logsumexp(log_probs_stack, dim=0) - np.log(len(agents))
    
    return mixture_log_prob

def do_eval(model, batchs, strategy='median'):
    state, next_state, action, next_action, reward, done = batchs
    
    state = state.to(device).float()
    action = action.to(device).float()

    latent_state = get_latent(model, state)

    with torch.no_grad():
        if hasattr(model, 'agents'): # ENSEMBLE CASE
            
            # --- A. Q-Value Physician (Ensemble Average) ---
            q_phys_list = []
            for agent in model.agents:
                q1 = agent.critic_1(latent_state, action)
                q2 = agent.critic_2(latent_state, action)
                q_min = torch.min(q1, q2)
                q_phys_list.append(q_min)
            q_value_phys = torch.mean(torch.stack(q_phys_list), dim=0).squeeze(1)

            # --- B. Get Action ---
            action_pred_np = model.get_action(state.cpu().numpy(), strategy=strategy)
            action_pred = torch.tensor(action_pred_np, dtype=torch.float32).to(device)
            
            # Logging stats (Agent 0 as representative)
            _, log_prob, action_mean, action_std = model.agents[0].actor.sample(latent_state)

            # --- C. Q-Value Agent (Ensemble Average) ---
            q_agent_list = []
            for agent in model.agents:
                q1 = agent.critic_1(latent_state, action_pred)
                q2 = agent.critic_2(latent_state, action_pred)
                q_min = torch.min(q1, q2)
                q_agent_list.append(q_min)
            
            q_mean = torch.mean(torch.stack(q_agent_list, dim=0), dim=0)
            q_agent = q_mean.squeeze(1)

        else: # SINGLE AGENT
            eval_agent = model
            q1 = eval_agent.critic_1(latent_state, action)
            q2 = eval_agent.critic_2(latent_state, action)
            q_value_phys = torch.min(q1, q2).squeeze(1)
            
            # FIXED: Clean single agent actor call
            mean, _ = eval_agent.actor(latent_state)
            action_pred = torch.tanh(mean) * eval_agent.actor.action_scale + eval_agent.actor.action_bias            
            
            log_prob, action_mean, action_std = None, action_pred, None 
            
            q1_a = eval_agent.critic_1(latent_state, action_pred)
            q2_a = eval_agent.critic_2(latent_state, action_pred)
            q_agent = torch.min(q1_a, q2_a).squeeze(1)
            
    return (q_value_phys, q_agent, action_pred, action, 
            log_prob.squeeze(1).cpu().numpy() if log_prob is not None else None,
            action_mean.cpu().numpy() if action_mean is not None else None, 
            action_std.cpu().numpy() if action_std is not None else None)

def compute_dm_return(model, batch, gamma=0.99, device=device):
    state, next_state, _, _, reward, done = batch
    state = state.to(device).float()
    next_state = next_state.to(device).float()
    reward = reward.to(device).float()
    done = done.to(device).float()

    latent_state = get_latent(model, state)
    latent_next_state = get_latent(model, next_state)

    agents = model.agents if hasattr(model, 'agents') else [model]

    with torch.no_grad():
        v_s_list, v_next_list = [], []
        for agent in agents:
            a_s, _, _, _ = agent.actor.sample(latent_state)
            a_next, _, _, _ = agent.actor.sample(latent_next_state)

            q_s = torch.min(
                agent.critic_1(latent_state, a_s),
                agent.critic_2(latent_state, a_s)
            ).squeeze(1)

            q_next = torch.min(
                agent.target_critic_1(latent_next_state, a_next),
                agent.target_critic_2(latent_next_state, a_next)
            ).squeeze(1)

            v_s_list.append(q_s)
            v_next_list.append(q_next)

        v_s = torch.mean(torch.stack(v_s_list), dim=0)
        v_next = torch.mean(torch.stack(v_next_list), dim=0)

        dm_vals = v_s + reward + gamma * (1 - done) * v_next

    return dm_vals.cpu().numpy()

def compute_dr_one_step(model, batch, behavior_model, gamma=0.99, device=device, num_bootstrap=200, alpha=0.05, behavior_stats=None):
    """
    Computes Doubly Robust Estimator with ENSEMBLE CONSISTENCY.
    """
    state, next_state, action_phys, next_action, reward, done = batch
    
    state = state.to(device).float()
    next_state = next_state.to(device).float()
    action_phys = action_phys.to(device).float()
    reward = reward.to(device).float()
    done = done.to(device).float()

    latent_state = get_latent(model, state)
    latent_next_state = get_latent(model, next_state)

    with torch.no_grad():
        # --- 1. Agent Value Estimates (Ensemble Averaging) ---
        # To be mathematically consistent with Ensemble log_pi, we must use Ensemble Q.
        
        # A. Q(s, a_phys) -> "How good was the doctor's action according to the group?"
        q_phys_list = []
        v_next_list = []
        v_s_list = []

        agents = model.agents if hasattr(model, 'agents') else [model]

        for agent in agents:
            # 1. Q(s, a_phys)
            q1_phys = agent.critic_1(latent_state, action_phys).squeeze(1)
            q2_phys = agent.critic_2(latent_state, action_phys).squeeze(1)
            q_phys_list.append(torch.min(q1_phys, q2_phys))

            # 2. V(s') = E[Q(s', a')] -> Use Target Network
            next_act_sample, _, _, _ = agent.actor.sample(latent_next_state)
            q1_next = agent.target_critic_1(latent_next_state, next_act_sample).squeeze(1)
            q2_next = agent.target_critic_2(latent_next_state, next_act_sample).squeeze(1)
            v_next_list.append(torch.min(q1_next, q2_next))

            # 3. V(s) = E[Q(s, a)] -> Use Current Network
            curr_act_sample, _, _, _ = agent.actor.sample(latent_state)
            q1_curr = agent.critic_1(latent_state, curr_act_sample).squeeze(1)
            q2_curr = agent.critic_2(latent_state, curr_act_sample).squeeze(1)
            v_s_list.append(torch.min(q1_curr, q2_curr))

        # Average across ensemble (Option B - rigorous consistency)
        q_hat_s_a_phys = torch.mean(torch.stack(q_phys_list), dim=0).view(-1)
        v_hat_next     = torch.mean(torch.stack(v_next_list), dim=0).view(-1)
        v_hat_s        = torch.mean(torch.stack(v_s_list), dim=0).view(-1)

        td_target = reward + gamma * (1.0 - done) * v_hat_next

        # --- 2. Importance Weights (Log Space) ---
        # A. Numerator: log pi(a_phys | s)
        if hasattr(model, 'agents'):
            log_pi_tensor = get_ensemble_log_prob(model.agents, latent_state, action_phys, device)
        else:
            # Fallback for single agent
            log_pi_tensor = get_ensemble_log_prob([model], latent_state, action_phys, device)
            
        log_pi = log_pi_tensor.cpu().numpy()

        # B. Denominator: log mu(a_phys | s)
        if behavior_model is not None:
            behavior_model.eval()
            beh_mu, beh_std = behavior_model(latent_state) 
            beh_dist = Normal(beh_mu, beh_std)
            # WARNING: Assuming BC output matches the action space of action_phys
            log_mu_tensor = behavior_model.get_log_prob(latent_state, action_phys).squeeze(1)
            log_mu = log_mu_tensor.cpu().numpy()
        else:
            # Fallback (Global Gaussian)
            phys_np = action_phys.cpu().numpy()
            if behavior_stats is not None:
                mu_mean = behavior_stats['mean']
                mu_std = behavior_stats['std']
            else:
                mu_mean = np.mean(phys_np, axis=0) 
                mu_std = np.std(phys_np, axis=0) + 1e-3
            
            mu_mean_batch = np.tile(mu_mean, (phys_np.shape[0], 1))
            mu_std_batch = np.tile(mu_std, (phys_np.shape[0], 1))
            log_mu = norm.logpdf(phys_np, loc=mu_mean_batch, scale=mu_std_batch).sum(axis=1)

        # C. Rho (Robust Log-Space Clipping)
        log_rho = log_pi - log_mu
        
        # FIX: Clip LOG-nya, bukan hasil exp-nya.
        # log(20) ~= 3.0. Kita clip di range [-10, 4] agar max weight sekitar 54.6
        log_rho = np.clip(log_rho, -10.0, 4.0) 
        
        rho = np.exp(log_rho)
        # (Tidak perlu np.clip lagi di sini karena log-nya sudah dijaga)

        # --- 3. DR Calculation ---
        v_hat_s_np = v_hat_s.cpu().numpy()
        q_hat_s_a_phys_np = q_hat_s_a_phys.cpu().numpy()
        td_target_np = td_target.cpu().numpy()

        dr_vals = v_hat_s_np + rho * (td_target_np - q_hat_s_a_phys_np)
        dr_mean = float(np.mean(dr_vals))

        # Bootstrap CI
        rng = np.random.RandomState(0)
        boot_means = []
        n = len(dr_vals)
        for _ in range(num_bootstrap):
            idx = rng.choice(n, n, replace=True)
            boot_means.append(np.mean(dr_vals[idx]))
        boot_means = np.array(boot_means)
        lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
        upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2.0)))
        
        # --- ESS & Stats Logging ---
        ess = np.sum(rho)**2 / (np.sum(rho**2) + 1e-12)

    return {
        'dr_mean': dr_mean,
        'dr_ci_batch': (lower, upper),
        'dr_vals': dr_vals,
        'wis_weights': rho,
        'ess': ess, 
        'rho_stats': (np.min(rho), np.max(rho), np.mean(rho)), 
        'log_pi': log_pi,
        'log_mu': log_mu
    }

def compute_wis_with_behavior(phys_actions, agent_log_probs, rewards, behavior_log_probs=None):
    rewards = np.asarray(rewards).squeeze()
    log_pi = np.asarray(agent_log_probs)

    # Behavior Estimate (Mu)
    if behavior_log_probs is not None and not np.any(np.equal(behavior_log_probs, None)):
        log_mu = np.asarray(behavior_log_probs, dtype=np.float64)
        if log_mu.ndim > 1: log_mu = log_mu.squeeze()
    else:
        # Fallback
        print("Warning: Using Global Gaussian stats for WIS")
        mu_mean = np.mean(phys_actions, axis=0, keepdims=True)
        mu_std  = np.std(phys_actions, axis=0, keepdims=True) + 1e-8
        phys_actions_np = np.asarray(phys_actions)
        mu_mean_rep = np.repeat(mu_mean, phys_actions_np.shape[0], axis=0)
        mu_std_rep  = np.repeat(mu_std, phys_actions_np.shape[0], axis=0)
        log_mu = norm.logpdf(phys_actions_np, loc=mu_mean_rep, scale=mu_std_rep).sum(axis=1)

    log_ratio = log_pi - log_mu
    
    # FIX: Aggressive Clipping untuk WIS juga
    # Mencegah satu sampel mendominasi 100% bobot
    log_ratio = np.clip(log_ratio, -10.0, 4.0) 
    
    # Stabilize
    log_ratio_max = np.max(log_ratio)
    w_unn = np.exp(log_ratio - log_ratio_max) 
    weights = w_unn / (np.sum(w_unn) + 1e-12) # Normalized

    v_wis = np.sum(weights * rewards)
    
    # Calculate Effective Sample Size (Normalized)
    ess = 1.0 / np.sum(weights ** 2)
    
    return v_wis, weights, ess

# ---- Bootstrap for WIS using log-probs (returns mean, CI, bootstrap samples) ----
def bootstrap_wis_from_logprobs(agent_log_probs, behavior_log_probs, rewards, num_bootstrap=1000, alpha=0.05, rng_seed=0):
    agent_log_probs = np.asarray(agent_log_probs)
    behavior_log_probs = np.asarray(behavior_log_probs)
    rewards = np.asarray(rewards).squeeze()
    n = len(rewards)
    rng = np.random.RandomState(rng_seed)

    log_ratio = agent_log_probs - behavior_log_probs
    wis_estimates = []
    for _ in range(num_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        sampled_lr = log_ratio[idx]
        sampled_rewards = rewards[idx]
        lr_stab = sampled_lr - np.max(sampled_lr)
        w_unn = np.exp(lr_stab)
        w = w_unn / (np.sum(w_unn) + 1e-12)
        wis = np.sum(w * sampled_rewards)
        wis_estimates.append(wis)

    wis_estimates = np.array(wis_estimates)
    mean = float(wis_estimates.mean())
    lower = float(np.percentile(wis_estimates, 100 * (alpha/2.0)))
    upper = float(np.percentile(wis_estimates, 100 * (1 - alpha/2.0)))
    return mean, (lower, upper), wis_estimates


def compute_dr_dataset_stats(dr_vals_all, wis_weights_all, num_bootstrap=2000, alpha=0.05, rng_seed=1):
    """
    Compute dataset-level DR mean, bootstrap CI, std, and ESS.
    (This function is VALID. Keep it.)
    """
    dr_vals_all = np.array(dr_vals_all)
    wis_weights_all = np.array(wis_weights_all)
    
    mean_dr = float(np.mean(dr_vals_all))

    # bootstrap CI
    rng = np.random.RandomState(rng_seed)
    n = len(dr_vals_all)
    boot_means = []
    for _ in range(num_bootstrap):
        idx = rng.choice(n, n, replace=True)
        boot_means.append(np.mean(dr_vals_all[idx]))
    boot_means = np.array(boot_means)
    lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2.0)))

    std_dr = float(np.std(dr_vals_all, ddof=1))
    
    # Effective sample size from normalized weights
    weights = wis_weights_all + 1e-12
    weights = weights / np.sum(weights)
    ess = float(1.0 / np.sum(weights ** 2))

    return {
        'dr_mean': mean_dr,
        'dr_ci': (lower, upper),
        'std_dr': std_dr,
        'ess': ess,
        'n': n
    }

def compute_dr_clinician(model, batch, gamma=0.99, device=device):
    """
    Menghitung DR Estimator khusus untuk Policy Clinician (Physician).
    (This function is VALID. Keep it.)
    """
    state, next_state, action_phys, next_action_phys, reward, done = batch
    
    state = state.to(device).float()
    next_state = next_state.to(device).float()
    action_phys = action_phys.to(device).float()
    next_action_phys = next_action_phys.to(device).float()
    reward = reward.to(device).float()
    done = done.to(device).float()

    if hasattr(model, 'agents'):
        eval_agent = model.agents[0]
    else:
        eval_agent = model

    latent_next_state = get_latent(model, next_state)

    # Use Target Critic for stability
    with torch.no_grad():
        q1_next = eval_agent.target_critic_1(latent_next_state, next_action_phys).squeeze(1)
        q2_next = eval_agent.target_critic_2(latent_next_state, next_action_phys).squeeze(1)
        q_hat_next_phys = torch.min(q1_next, q2_next).view(-1)

        dr_vals_clinician = reward + gamma * (1.0 - done) * q_hat_next_phys

    return dr_vals_clinician.cpu().numpy()


def compute_episode_dr_return(dr_vals, bloc_num, gamma=0.99):
    """
    Compute episode-level DR return.
    dr_vals  : per-timestep DR values (1D array)
    bloc_num : episode id per timestep (1D array)
    """
    episode_returns = {}

    for dr, eid in zip(dr_vals, bloc_num):
        if eid not in episode_returns:
            episode_returns[eid] = []
        episode_returns[eid].append(dr)

    G_eps = []
    for eid, rewards in episode_returns.items():
        G = 0.0
        for t, r in enumerate(rewards):
            G += (gamma ** t) * r
        G_eps.append(G)

    return np.array(G_eps)

def estimate_survival_rate_episode_level(
    bloc_num_array,
    log_pi_array,
    log_mu_array,
    y90_array,
    num_bootstrap=1000,
    alpha=0.05,
    rng_seed=42
):
    """
    Menghitung estimasi survival rate pada level EPISODE (Pasien) untuk mengurangi varian.
    Menggunakan Geometric Mean dari Importance Weights per episode untuk stabilitas.
    """
    print(f"   -> Calculating Episode-Level Survival OPE (N={len(np.unique(bloc_num_array))})...")
    
    unique_ids = np.unique(bloc_num_array)
    episode_weights = []
    episode_outcomes = [] # 1 = Survive, 0 = Dead

    # 1. AGREGASI PER PASIEN
    for uid in unique_ids:
        # Ambil indeks untuk pasien ini
        idx = np.where(bloc_num_array == uid)[0]
        
        # Ambil Log Prob
        l_pi = log_pi_array[idx]
        l_mu = log_mu_array[idx]
        
        # Hitung Log Ratio: log(pi/mu)
        log_rho = l_pi - l_mu
        
        # CLIPPING (Sangat penting untuk mengurangi varian ekstrim)
        # Clip di log space: -10 s.d 4 (artinya rho max ~54.6)
        log_rho = np.clip(log_rho, -10.0, 4.0)
        
        # --- STRATEGI PEMBOBOTAN EPISODE ---
        # Opsi A (Product): exp(sum(log_rho)) -> Secara teoritis benar, tapi varian meledak.
        # Opsi B (Mean): exp(mean(log_rho)) -> "Average probability ratio", bias tapi stabil (Disarankan).
        
        # Kita gunakan Opsi B (Geometric Mean per Step) untuk kestabilan numerik
        w_episode = np.exp(np.mean(log_rho))
        
        episode_weights.append(w_episode)
        
        # Outcome (Y90 konstan per episode, ambil yang pertama)
        is_dead = y90_array[idx][0]
        episode_outcomes.append(1 - is_dead) # 1 = Survive

    episode_weights = np.array(episode_weights)
    episode_outcomes = np.array(episode_outcomes)

    # 2. NORMALISASI BOBOT GLOBAL
    # W_norm = W_i / Sum(W)
    w_norm = episode_weights / (np.sum(episode_weights) + 1e-12)

    # 3. POINT ESTIMATE
    surv_est = np.sum(w_norm * episode_outcomes)

    # 4. BOOTSTRAP PADA LEVEL PASIEN (Correct Way)
    rng = np.random.RandomState(rng_seed)
    n_episodes = len(episode_weights)
    boot_vals = []

    for _ in range(num_bootstrap):
        # Sample indeks pasien dengan pengembalian
        boot_idx = rng.choice(n_episodes, n_episodes, replace=True)
        
        w_sample = episode_weights[boot_idx]
        o_sample = episode_outcomes[boot_idx]
        
        # Normalisasi ulang di dalam bootstrap
        w_sample_norm = w_sample / (np.sum(w_sample) + 1e-12)
        
        boot_est = np.sum(w_sample_norm * o_sample)
        boot_vals.append(boot_est)

    lower = float(np.percentile(boot_vals, 100 * alpha / 2))
    upper = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))

    return surv_est, (lower, upper), len(unique_ids)


def compute_dr_survival_calibrated(states, actions_phys, actions_agent, y90s, log_pi_agent, log_mu_beh):
    """
    Menghitung Survival Rate menggunakan Standard Doubly Robust (DR) Estimator.
    
    Formula: V_DR = Q_model(s, a_agent) + rho * (Y_actual - Q_model(s, a_phys))
    
    1. Q_model(s, a_agent): Prediksi survival jika Agen yang bertindak (Direct Method).
    2. Residual Correction: Menggunakan IS (rho) untuk mengoreksi bias model pada data dokter.
    """
    print("\n[DR-Survival] Training Calibrated Survival Model...")
    
    # --- 1. PREPARE DATA ---
    # Input: State + Action Physician
    # Target: Survival (1 - Y90)
    X_obs = np.hstack([states, actions_phys])
    y_obs = (1 - y90s).astype(int)
    
    # --- 2. TRAIN PROBABILISTIC MODEL (Q-Function for Survival) ---
    # Logistic Regression + Sigmoid Calibration
    base_clf = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    surv_model = CalibratedClassifierCV(base_clf, method='sigmoid', cv=5)
    surv_model.fit(X_obs, y_obs)
    
    # --- 3. PREDICTIONS ---
    # A. Q_hat(s, a_phys): Prediksi survival pada data observasi
    q_hat_phys = surv_model.predict_proba(X_obs)[:, 1]
    
    # B. Q_hat(s, a_agent): Prediksi survival hipotesis (Counterfactual)
    # Kita gabungkan State asli dengan Action dari Agen
    X_agent = np.hstack([states, actions_agent])
    q_hat_agent = surv_model.predict_proba(X_agent)[:, 1]
    
    # --- 4. IMPORTANCE WEIGHTS (Rho) ---
    log_rho = log_pi_agent - log_mu_beh
    log_rho = np.clip(log_rho, -10.0, 4.0) # Clipping untuk stabilitas
    rho = np.exp(log_rho)
    
    # --- 5. DR CALCULATION ---
    # DR = Model_Prediction_Agent + Rho * (Real_Outcome - Model_Prediction_Phys)
    dr_values = q_hat_agent + rho * (y_obs - q_hat_phys)
    
    # Clipping hasil DR ke probability range [0, 1] agar logis secara medis
    dr_values = np.clip(dr_values, 0.0, 1.0)
    
    mean_dr = np.mean(dr_values)
    
    # --- 6. BOOTSTRAP CI ---
    rng = np.random.RandomState(42)
    boot_means = []
    n_samples = len(dr_values)
    for _ in range(1000):
        idx = rng.choice(n_samples, n_samples, replace=True)
        boot_means.append(np.mean(dr_values[idx]))
        
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    
    return mean_dr, (ci_lower, ci_upper)

def print_final_comparison(phys_surv, ep_est, ep_ci, dr_est, dr_ci):
    """Mencetak tabel perbandingan final untuk paper/laporan."""
    data = {
        "Metric Approach": ["Physician (Observed)", "Episode IS (Sensitivity)", "DR-Calibrated (Primary)"],
        "Survival Rate": [f"{phys_surv:.2%}", f"{ep_est:.2%}", f"{dr_est:.2%}"],
        "95% CI": ["N/A", f"[{ep_ci[0]:.2%}, {ep_ci[1]:.2%}]", f"[{dr_ci[0]:.2%}, {dr_ci[1]:.2%}]"],
        "Confidence Level": ["High (Ground Truth)", "Low (Bias & High Variance)", "High (Robust)"],
        "Role": ["Baseline", "Appendix / Sensitivity", "Main Result"]
    }
    
    df = pd.DataFrame(data)
    print("\n" + "="*85)
    print(" FINAL SURVIVAL ESTIMATION COMPARISON ")
    print("="*85)
    print(df.to_string(index=False))
    print("="*85 + "\n")


def do_test(model, Xtest, actionbloctest, bloctest, Y90, SOFA, reward_value, beat, bc_model=None):
    # [Trajectory Generation Code is correct - Keep it]
    bloc_max = max(bloctest)
    r = np.array([reward_value, -reward_value]).reshape(1, -1)
    r2 = r * (2 * (1 - Y90.reshape(-1, 1)) - 1)
    R3 = r2[:, 0]

    RNNstate = Xtest
    statesize = RNNstate.shape[1]
    num_samples = RNNstate.shape[0]

    states = np.zeros((num_samples, statesize))
    actions = np.zeros((num_samples, 2), dtype=np.float32)
    next_actions = np.zeros((num_samples, 2), dtype=np.float32)
    rewards = np.zeros((num_samples, 1))
    next_states = np.zeros((num_samples, statesize))
    done_flags = np.zeros((num_samples, 1))
    bloc_num = np.zeros((num_samples, 1))

    c = 0
    blocnum1 = 1

    for i in range(num_samples - 1):
        states[c] = RNNstate[i, :]
        actions[c] = actionbloctest[i]
        bloc_num[c] = blocnum1

        if bloctest[i + 1] == 1:
            next_states1 = np.zeros(statesize)
            next_actions1 = -1
            done_flags1 = 1
            blocnum1 += 1
            reward1 = (-beat[0] * (SOFA[i]) + R3[i])
        else:
            next_states1 = RNNstate[i + 1, :]
            next_actions1 = actionbloctest[i + 1]
            done_flags1 = 0
            reward1 = (-beat[1] * (SOFA[i + 1] - SOFA[i]))

        next_states[c] = next_states1
        next_actions[c] = next_actions1
        rewards[c] = reward1
        done_flags[c] = done_flags1
        c += 1

    states[c] = RNNstate[c, :]
    actions[c] = actionbloctest[c]
    bloc_num[c] = blocnum1
    next_states1 = np.zeros(statesize)
    next_actions1 = -1
    done_flags1 = 1
    reward1 = -beat[0] * (SOFA[c]) + R3[c]
    next_states[c] = next_states1
    next_actions[c] = next_actions1
    rewards[c] = reward1
    done_flags[c] = done_flags1
    c += 1

    bloc_num = np.squeeze(bloc_num[:c, :])
    states = states[: c, :]
    next_states = next_states[: c, :]
    actions = np.squeeze(actions[: c, :])
    next_actions = np.squeeze(next_actions[: c, :])
    rewards = np.squeeze(rewards[: c, :])
    done_flags = np.squeeze(done_flags[: c, :])

    state = torch.FloatTensor(states).to(device)
    next_state = torch.FloatTensor(next_states).to(device)
    action = torch.FloatTensor(actions).to(device)
    next_action = torch.FloatTensor(next_actions).to(device)
    reward = torch.FloatTensor(rewards).to(device)
    done = torch.FloatTensor(done_flags).to(device)

    # --- EVALUATION LOOP ---
    rec_phys_q, rec_agent_q = [], []
    rec_agent_a, rec_phys_a, rec_sur, rec_reward_user = [], [], [], []
    rec_agent_q_pro = []
    rec_action_mean = []
    rec_action_std = []
    rec_agent_log_pi = [] 
    rec_behavior_log_mu = []
    rho_stats_list = []
    
    ess_list = [] 
    rec_dm_vals = []
    rec_dr_vals = []         
    rec_dr_clinician = []    
    rec_wis_weights = []
    rec_dr_batch_means = []
    rec_dr_batch_cis = []

    batch_size = 256
    uids = np.unique(bloc_num)
    num_batch = uids.shape[0] // batch_size

    all_actions = actions 
    glob_beh_mean = np.mean(all_actions, axis=0)
    glob_beh_std = np.std(all_actions, axis=0) + 1e-3
    behavior_stats = {'mean': glob_beh_mean, 'std': glob_beh_std}

    save_dir = 'SACEnsembleexp-algorithm/'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting Evaluation on {num_batch} batches...")

    for batch_idx in range(num_batch + 1):
        batch_uids = uids[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_mask = np.isin(bloc_num, batch_uids)
        if batch_mask.sum() == 0:
            continue
        
        batch = (state[batch_mask], next_state[batch_mask], action[batch_mask],
                 next_action[batch_mask], reward[batch_mask], done[batch_mask])

        q_phys, q_agent_a, agent_actions, phys_actions, agent_log_prob, action_mean, action_std = do_eval(model, batch, strategy='expertise')

        rec_phys_q.extend(q_phys.cpu().numpy())
        rec_agent_q.extend(q_agent_a.cpu().numpy())
        rec_agent_a.extend(agent_actions.cpu().numpy())
        rec_phys_a.extend(phys_actions.cpu().numpy())
        rec_sur.extend(Y90[batch_mask])
        rec_reward_user.extend(reward[batch_mask].cpu().numpy().squeeze().tolist())
        
        if action_mean is not None:
            rec_action_mean.extend(action_mean)
            rec_action_std.extend(action_std)
        else:
            rec_action_mean.extend([np.nan] * int(batch_mask.sum()))
            rec_action_std.extend([np.nan] * int(batch_mask.sum()))

        # B. Compute DR AGENT (Ensemble Consistent)
        dr_res = compute_dr_one_step(model, batch, behavior_model=bc_model, 
                                     gamma=0.99, device=device, 
                                     num_bootstrap=200, behavior_stats=behavior_stats)
        
        rec_dr_vals.extend(dr_res['dr_vals'])
        rec_wis_weights.extend(dr_res['wis_weights'])
        
        # FIXED: Append ESS to list
        ess_list.append(dr_res['ess'])
        rho_stats_list.append(dr_res['rho_stats'])

        rec_agent_log_pi.extend(dr_res['log_pi'].tolist())
        rec_behavior_log_mu.extend(dr_res['log_mu'].tolist())
        rec_agent_q_pro.extend(dr_res['log_pi'].tolist())

        # D. Compute DR CLINICIAN
        dr_vals_phys = compute_dr_clinician(model, batch, gamma=0.99, device=device)
        rec_dr_clinician.extend(dr_vals_phys)

        dm_vals = compute_dm_return(
        model,
        batch,
        gamma=0.99,
        device=device
        )
        rec_dm_vals.extend(dm_vals)

    # --- SAVING & STATS ---
    # [Saving code remains identical to yours...]
    np.save(os.path.join(save_dir, 'shencunlv_testexp2.npy'), rec_sur)
    np.save(os.path.join(save_dir, 'phys_bQ_testexp2.npy'), rec_phys_q)
    np.save(os.path.join(save_dir, 'agent_bQ_testexp2.npy'), rec_agent_q)
    np.save(os.path.join(save_dir, 'reward_testexp2.npy'), rec_reward_user)
    np.save(os.path.join(save_dir, 'agent_actionsb_testexp2.npy'), rec_agent_a)
    np.save(os.path.join(save_dir, 'phys_actionsb_testexp2.npy'), rec_phys_a)
    np.save(os.path.join(save_dir, 'rec_agent_q_pro_testexp2.npy'), rec_agent_q_pro)
    np.save(os.path.join(save_dir, 'rec_agent_action_mean_testexp2.npy'), rec_action_mean)
    np.save(os.path.join(save_dir, 'rec_agent_action_std_testexp2.npy'), rec_action_std)

    print(f"Q_phys avg: {np.mean(rec_phys_q):.4f}")
    print(f"Q_agent avg: {np.mean(rec_agent_q):.4f}")

    # --- DR STATS ---
    rec_dr_vals = np.array(rec_dr_vals)
    rec_wis_weights = np.array(rec_wis_weights)
    
    np.save(os.path.join(save_dir, 'dr_vals_testexp2.npy'), rec_dr_vals)
    np.save(os.path.join(save_dir, 'dr_wis_weights_testexp2.npy'), rec_wis_weights)
    
    dr_stats = compute_dr_dataset_stats(rec_dr_vals, rec_wis_weights, num_bootstrap=2000)
    print("\n--- AGENT PERFORMANCE ---")
    print(f"DR Estimated Return (mean): {dr_stats['dr_mean']:.6f}")
    print(f"DR 95% CI: ({dr_stats['dr_ci'][0]:.6f}, {dr_stats['dr_ci'][1]:.6f})")
    
    # FIXED: Report Average Batch ESS instead of Sum
    avg_batch_ess = np.mean(ess_list)
    print(f"Avg Batch ESS: {avg_batch_ess:.2f}, N: {dr_stats['n']}")
    
    np.save(os.path.join(save_dir, 'dr_stats_testexp2.npy'), dr_stats)

    # --- CLINICIAN ---
    rec_dr_clinician = np.array(rec_dr_clinician)
    dummy_weights_phys = np.ones_like(rec_dr_clinician)
    dr_stats_phys = compute_dr_dataset_stats(rec_dr_clinician, dummy_weights_phys, num_bootstrap=2000)
    
    print("\n--- CLINICIAN PERFORMANCE (Baseline) ---")
    print(f"DR Estimated Return (mean): {dr_stats_phys['dr_mean']:.6f}")
    print(f"DR 95% CI: ({dr_stats_phys['dr_ci'][0]:.6f}, {dr_stats_phys['dr_ci'][1]:.6f})")
    
    np.save(os.path.join(save_dir, 'dr_vals_clinician_testexp2.npy'), rec_dr_clinician)
    np.save(os.path.join(save_dir, 'dr_stats_clinician_testexp2.npy'), dr_stats_phys)

    # --- WIS ---
    rec_phys_a = np.array(rec_phys_a)
    rec_reward_user = np.array(rec_reward_user).squeeze()
    log_pi_arr = np.array(rec_agent_log_pi)
    log_mu_arr = np.array(rec_behavior_log_mu)

    v_wis, wis_weights, global_ess = compute_wis_with_behavior(
        rec_phys_a, log_pi_arr, rec_reward_user,
        behavior_log_probs=log_mu_arr 
    )

    print(f"\nWIS Estimated Return: {v_wis:.4f}")
    print(f"Global WIS ESS: {global_ess:.2f}")
    
    np.save(os.path.join(save_dir, 'wis_weights_testexp2.npy'), wis_weights)
    np.save(os.path.join(save_dir, 'wis_value_testexp2.npy'), np.array([v_wis]))

    v_behavior = behavior_policy_estimator(rec_reward_user)
    print(f"Behavior Policy Estimated Return: {v_behavior:.4f}")
    np.save(os.path.join(save_dir, 'behavior_policy_value_testexp2.npy'), np.array([v_behavior]))

    wis_mean, wis_ci, wis_boot_samples = bootstrap_wis_from_logprobs(log_pi_arr, log_mu_arr, rec_reward_user, num_bootstrap=1000)
    print(f"WIS Mean (bootstrap): {wis_mean:.4f}")
    print(f"WIS 95% Confidence Interval: {wis_ci}")

    np.save(os.path.join(save_dir, 'wis_log_pi_testexp2.npy'), log_pi_arr)
    np.save(os.path.join(save_dir, 'wis_log_mu_testexp2.npy'), log_mu_arr)
    np.save(os.path.join(save_dir, 'wis_boot_samples_testexp2.npy'), wis_boot_samples)

    # =========================================================================
    # 5. EPISODE-LEVEL EXPECTED RETURN & SURVIVAL
    # =========================================================================
    print("\n" + "="*60)
    print(" STEP 5: REWARD VALIDATION (EPISODE RETURN vs SURVIVAL) ")
    print("="*60)

    bloc_num_arr = np.array(bloc_num)                    # episode ID per timestep
    rec_sur_arr = np.array(rec_sur).astype(int)          # label Y90 per timestep (0/1)
    rec_reward_arr = np.array(rec_reward_user).astype(float)  # reward per timestep

    episode_ids = np.unique(bloc_num_arr)

    episode_returns = []
    episode_survival = []

    gamma = 0.99 

    for eid in episode_ids:
        idx = np.where(bloc_num_arr == eid)[0]
        # Expected Return per Episode: G = sum_t gamma^t * r_t
        rewards_ep = rec_reward_arr[idx]
        G = 0.0
        for t, r_t in enumerate(rewards_ep):
            G += (gamma ** t) * r_t
        episode_returns.append(G)

        # Survival per Episode
        y90_ep = rec_sur_arr[idx][0]        # 1 value per patient
        surv_flag = 1 - y90_ep              # survive = 1, death = 0
        episode_survival.append(surv_flag)

    episode_returns = np.array(episode_returns)
    episode_survival = np.array(episode_survival)

    np.save(os.path.join(save_dir, 'episode_returns_testexp2.npy'), episode_returns)
    np.save(os.path.join(save_dir, 'episode_survival_testexp2.npy'), episode_survival)

    print(f"Total Episodes           : {len(episode_returns)}")
    print(f"Mean Expected Return     : {episode_returns.mean():.4f}")
    print(f"Observed Survival Rate   : {episode_survival.mean():.4f}")

    # Correlation
    if len(episode_returns) > 1 and np.std(episode_returns) > 0 and np.std(episode_survival) > 0:
        corr = np.corrcoef(episode_returns, episode_survival)[0, 1]
    else:
        corr = np.nan

    print("\n[RETURN–SURVIVAL RELATIONSHIP (EPISODE-LEVEL)]")
    print(f"Correlation(G_episode, survival): {corr:.4f}")
    if not np.isnan(corr) and corr > 0:
        print("✅ POSITIF: Expected return per episode selaras dengan survival.")
    else:
        print("⚠️ PERINGATAN: Korelasi tidak positif / tidak terdefinisi.")

    if np.any(episode_survival == 1) and np.any(episode_survival == 0):
        mean_ret_survive = episode_returns[episode_survival == 1].mean()
        mean_ret_death   = episode_returns[episode_survival == 0].mean()
        print("\nMean Expected Return by Outcome:")
        print(f"  Survivors (1): {mean_ret_survive:.4f}")
        print(f"  Deaths    (0): {mean_ret_death:.4f}")

    # =========================================================================
    # 6. POLICY-LEVEL COMPARISON & FINAL REPORT
    # =========================================================================
    print("\n" + "="*60)
    print(" STEP 6: SURVIVAL RATE ESTIMATION (IS vs DR) ")
    print("="*60)

    # A. PREPARE DATA ARRAYS
    # Kita butuh data mentah dari seluruh trajectory (bukan batch)
    bloc_num_arr = np.array(bloc_num).flatten()
    log_pi_arr   = np.array(rec_agent_log_pi)
    log_mu_arr   = np.array(rec_behavior_log_mu)
    rec_sur_arr  = np.array(rec_sur).flatten() # Y90 (0/1)
    
    # Untuk DR, kita butuh state dan action lengkap
    # rec_agent_a dan rec_phys_a sudah list, ubah ke numpy
    actions_agent_all = np.array(rec_agent_a)
    actions_phys_all  = np.array(rec_phys_a)
    
    action_dist = np.linalg.norm(actions_agent_all - actions_phys_all, axis=1)

    sofa_all = SOFA[:len(action_dist)]

    bins = [0, 5, 10, 15, np.inf]
    labels = ["Mild", "Moderate", "Severe", "Critical"]

    sofa_group = pd.cut(sofa_all, bins=bins, labels=labels)

    df_align = pd.DataFrame({
        "action_dist": action_dist,
        "sofa_group": sofa_group
    })

    align_stats = df_align.groupby("sofa_group")["action_dist"].agg(["mean", "median"])
    print(align_stats)

    time_index = np.arange(len(action_dist))
    early = time_index < np.median(time_index)

    print(np.mean(action_dist[early]), np.mean(action_dist[~early]))

    survival_flag = 1 - np.array(rec_sur).astype(int)

    mean_dist_survive = action_dist[survival_flag == 1].mean()
    mean_dist_death   = action_dist[survival_flag == 0].mean()


    low_q  = np.percentile(actions_phys_all, 5, axis=0)
    high_q = np.percentile(actions_phys_all, 95, axis=0)

    out_of_range = (
    (actions_agent_all < low_q) |
    (actions_agent_all > high_q)
    )

    out_rate_per_dim = out_of_range.mean(axis=0)
    out_rate_any = out_of_range.any(axis=1).mean()

    print("Out-of-range rate per action dim:", out_rate_per_dim)
    print("Any-dimension violation rate:", out_rate_any)

    # State asli agak tricky karena 'state' adalah tensor di loop
    # Kita ambil dari Xtest (RNNstate) yang sudah ada di awal fungsi do_test
    # Pastikan dipotong sesuai jumlah sampel yang valid (variabel 'c' di loop trajectory)
    # Variable 'states' di awal fungsi do_test menyimpan state numpy
    states_all = states # Ini sudah numpy array (num_samples, 24)

    # 1. GROUND TRUTH (PHYSICIAN)
    phys_survival_rate_true = (1 - rec_sur_arr).mean()

    # 2. SENSITIVITY ANALYSIS: EPISODE-LEVEL IS
    # (Metode yang Anda buat sebelumnya - bagus untuk cek kestabilan, tapi bias)
    ep_surv_est, ep_surv_ci, n_patients = estimate_survival_rate_episode_level(
        bloc_num_array=bloc_num_arr,
        log_pi_array=log_pi_arr,
        log_mu_array=log_mu_arr,
        y90_array=rec_sur_arr,
        num_bootstrap=2000
    )

    # 3. PRIMARY METRIC: DOUBLY ROBUST CALIBRATED
    # (Metode Gold Standard)
    dr_surv_est, dr_surv_ci = compute_dr_survival_calibrated(
        states=states_all,
        actions_phys=actions_phys_all,
        actions_agent=actions_agent_all,
        y90s=rec_sur_arr,
        log_pi_agent=log_pi_arr,
        log_mu_beh=log_mu_arr
    )

    # 4. PRINT COMPARISON TABLE
    print_final_comparison(
        phys_survival_rate_true, 
        ep_surv_est, ep_surv_ci, 
        dr_surv_est, dr_surv_ci
    )

    # 5. SAVE ALL RESULTS
    save_path_dr = os.path.join(save_dir, 'survival_comparison_results_testexp2.npy')
    results_dict = {
        'phys_true': phys_survival_rate_true,
        'episode_is_mean': ep_surv_est,
        'episode_is_ci': ep_surv_ci,
        'dr_mean': dr_surv_est,
        'dr_ci': dr_surv_ci,
        'n_patients': n_patients
    }
    np.save(save_path_dr, results_dict)
    print(f"Results saved to: {save_path_dr}")

    
    rec_dm_vals = np.array(rec_dm_vals)
              
    np.save(
        os.path.join(save_dir, "dm_vals_testexp2.npy"),
        rec_dm_vals
    )

    dm_stats = {
    "mean": float(np.mean(rec_dm_vals)),
    "std":  float(np.std(rec_dm_vals, ddof=1)),
    "n":    int(len(rec_dm_vals))
    }

    np.save(
        os.path.join(save_dir, "dm_stats_testexp2.npy"),
        dm_stats
    )

    print("\n--- DIRECT METHOD (DM) ---")
    print(f"DM Mean Return : {dm_stats['mean']:.6f}")
    print(f"DM Std        : {dm_stats['std']:.6f}")
    print(f"N Samples     : {dm_stats['n']}")

    corr_dm_dr = np.corrcoef(rec_dm_vals, rec_dr_vals)[0, 1]
    print(f"Correlation(DM, DR): {corr_dm_dr:.4f}")



def behavior_policy_estimator(rewards):
    """
    Estimasi nilai dari policy behavior berdasarkan data (mean reward).

    Output:
    - Nilai ekspektasi reward (float)
    """
    return np.mean(rewards)