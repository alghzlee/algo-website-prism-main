from scipy.stats import norm
import torch
import numpy as np
import os
import torch.nn.functional as F
from SAC_deepQnet_bestexp import SACAgent, EnsembleSAC



# # device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- HELPER FUNCTIONS (Wajib ada di atas do_test) ---

def get_latent(model, state_tensor):
    """Helper untuk encode state secara otomatis jika AE tersedia."""
    # Cek apakah model adalah Ensemble atau Single Agent
    if hasattr(model, 'agents'):
        # Ambil AE dari agent pertama (karena shared)
        ae = model.agents[0].autoencoder
    else:
        ae = model.autoencoder
        
    if ae is not None:
        with torch.no_grad():
            return ae.encode(state_tensor)
    return state_tensor


def do_eval(model, batchs, strategy='vote'):
    state, next_state, action, next_action, reward, done = batchs
    
    # Pastikan data di device yang benar
    state = state.to(device).float()
    action = action.to(device).float()

    # --- 1. ENCODE STATE (Raw 37 -> Latent 24) ---
    latent_state = get_latent(model, state)

    with torch.no_grad():
        # --- A. Q-Value untuk Action Physician (Baseline) ---
        if hasattr(model, 'agents'): # Ensemble
            eval_agent = model.agents[0] 
        else:
            eval_agent = model

        # Critic menilai action dokter berdasarkan latent state
        q_value_phys1 = eval_agent.critic_1(latent_state, action)
        q_value_phys2 = eval_agent.critic_2(latent_state, action)
        q_value_phys = torch.min(q_value_phys1, q_value_phys2).squeeze(1)

        # --- B. Get Action dari Agent (Prediction) ---
        if hasattr(model, 'agents'):
            # Ensemble: get_action menghandle encoding internal jika di-set, 
            # tapi kita sudah punya latent, jadi kita manual saja biar cepat/konsisten
            # Note: Ensemble get_action return numpy, kita perlu tensor
            
            # Kita pakai sample dari salah satu agent untuk statistik (mean/std/log_prob)
            # Ini aproksimasi untuk WIS
            action_sampled, log_prob, action_mean, action_std = eval_agent.actor.sample(latent_state)
            
            # Untuk Q-value Agent, kita gunakan strategi ensemble (vote/mean)
            # Kita gunakan mean action dari ensemble untuk performa terbaik
            action_pred_np = model.get_action(state.cpu().numpy(), strategy=strategy)
            action_pred = torch.tensor(action_pred_np, dtype=torch.float32).to(device)
            
        else:
            # Single Agent
            action_pred, log_prob, action_mean, action_std = model.actor.sample(latent_state)
            action_sampled = action_pred

        # --- C. Q-Value untuk Action Agent ---
        q_agent_1 = eval_agent.critic_1(latent_state, action_pred)
        q_agent_2 = eval_agent.critic_2(latent_state, action_pred)
        q_agent = torch.min(q_agent_1, q_agent_2).squeeze(1)

    return (q_value_phys, q_agent, action_pred, action, 
            log_prob.squeeze(1).cpu().numpy() if log_prob is not None else None,
            action_mean.cpu().numpy() if action_mean is not None else None, 
            action_std.cpu().numpy() if action_std is not None else None)

def compute_dr_one_step(model, batch, gamma=0.99, device=device, num_bootstrap=200, alpha=0.05):
    """
    Menghitung Doubly Robust (DR) Estimator dengan dukungan AutoEncoder.
    """
    state, next_state, action_phys, next_action, reward, done = batch
    
    # Pindah ke GPU
    state = state.to(device).float()
    next_state = next_state.to(device).float()
    action_phys = action_phys.to(device).float()
    reward = reward.to(device).float()
    done = done.to(device).float()

    # Pilih agent evaluasi
    if hasattr(model, 'agents'):
        eval_agent = model.agents[0]
    else:
        eval_agent = model

    # --- ENCODE STATE & NEXT STATE ---
    latent_state = get_latent(model, state)
    latent_next_state = get_latent(model, next_state)

    eval_agent.critic_1.eval(); eval_agent.critic_2.eval(); eval_agent.actor.eval()

    with torch.no_grad():
        # 1. Dapatkan statistik policy agent pada state saat ini
        mean_tensor, std_tensor = eval_agent.actor(latent_state)
        std_tensor = std_tensor.clamp(min=1e-6)

        # 2. Sample action agent untuk menghitung V_hat (Expected Value)
        #    Kita pakai sample untuk menghitung expectation Q
        action_agent_sample, _, _, _ = eval_agent.actor.sample(latent_state)

        # 3. Hitung Q_hat(s, a_agent) -> Estimasi nilai policy agent
        q1_agent = eval_agent.critic_1(latent_state, action_agent_sample).squeeze(1)
        q2_agent = eval_agent.critic_2(latent_state, action_agent_sample).squeeze(1)
        q_hat_s_a_agent = torch.min(q1_agent, q2_agent).view(-1) 

        # 4. Hitung V_hat(s') untuk TD Target (DR correction term)
        next_action_sample, _, _, _ = eval_agent.actor.sample(latent_next_state)
        q1_next = eval_agent.target_critic_1(latent_next_state, next_action_sample).squeeze(1)
        q2_next = eval_agent.target_critic_2(latent_next_state, next_action_sample).squeeze(1)
        q_hat_next = torch.min(q1_next, q2_next).view(-1)

        # One-step TD target: r + gamma * V(s')
        td_target = reward + gamma * (1.0 - done) * q_hat_next

        # 5. Hitung Importance Weights (rho)
        #    rho = pi(a_phys | s) / mu(a_phys | s)
        #    Disini kita gunakan self-normalized importance weights berdasarkan pi saja 
        #    (Asumsi behavior uniform/unknown, atau fallback strategy)
        phys_np = action_phys.cpu().numpy()
        agent_mean_np = mean_tensor.cpu().numpy()
        agent_std_np = std_tensor.cpu().numpy()
        
        # Probabilitas physician action menurut policy agent
        pi_probs = _agent_pdf_multivariate(phys_np, agent_mean_np, agent_std_np)
        
        # Self-normalized weights
        weights = pi_probs + 1e-12
        wis_weights = weights / np.sum(weights)

        # 6. DR Calculation per sample
        #    DR = V_hat(s) + rho * (r + gamma*V_hat(s') - Q_hat(s, a_phys))
        #    Note: Implementasi Anda sebelumnya menggunakan q_hat_s_a_agent sebagai base estimation
        #    Formula standard DR: V_DR = V_model + rho * (R_obs - Q_model)
        #    Kita ikuti logika Anda: dr_vals = q_hat_agent + w * (target - q_hat_agent)
        
        q_hat_s_a_agent_np = q_hat_s_a_agent.cpu().numpy()
        td_target_np = td_target.cpu().numpy()
        
        dr_vals = q_hat_s_a_agent_np + wis_weights * (td_target_np - q_hat_s_a_agent_np)

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

    return {
        'dr_mean': dr_mean,
        'dr_ci_batch': (lower, upper),
        'dr_vals': dr_vals,
        'wis_weights': wis_weights,
        'pi_probs': pi_probs,
        'q_hat_s_a_agent': q_hat_s_a_agent_np,
        'td_target': td_target_np
    }

def _logpdf_multivariate(x_np, mean_np, std_np):
    std_np = np.asarray(std_np) + 1e-8
    logpdf = norm.logpdf(x_np, loc=mean_np, scale=std_np)
    return np.sum(logpdf, axis=1)  # [N]

# ---- Stable WIS using behavior estimate (global Gaussian fallback) ----
def compute_wis_with_behavior(phys_actions, agent_means, agent_stds, rewards, behavior_mean=None, behavior_std=None):
    phys_actions = np.asarray(phys_actions)
    agent_means = np.asarray(agent_means)
    agent_stds = np.asarray(agent_stds)
    rewards = np.asarray(rewards).squeeze()
    N = phys_actions.shape[0]

    # compute log pi(a|s)
    log_pi = _logpdf_multivariate(phys_actions, agent_means, agent_stds)

    # behavior estimate: global gaussian per-dim if not provided
    if behavior_mean is None or behavior_std is None:
        behavior_mean = np.mean(phys_actions, axis=0, keepdims=True)  # [1,adim]
        behavior_std  = np.std(phys_actions, axis=0, keepdims=True) + 1e-8
        behavior_mean = np.repeat(behavior_mean, N, axis=0)
        behavior_std  = np.repeat(behavior_std, N, axis=0)
    log_mu = _logpdf_multivariate(phys_actions, behavior_mean, behavior_std)

    log_ratio = log_pi - log_mu
    # stabilize and normalize
    log_ratio_stab = log_ratio - np.max(log_ratio)
    w_unn = np.exp(log_ratio_stab)
    weights = w_unn / (np.sum(w_unn) + 1e-12)

    v_wis = np.sum(weights * rewards)
    return v_wis, weights, log_pi, log_mu

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
    
def _agent_pdf_multivariate(phys_actions_np, agent_means_np, agent_stds_np):
    """
    Compute product of univariate normal pdf across action dims -> [N]
    """
    probs = norm.pdf(phys_actions_np, loc=agent_means_np, scale=agent_stds_np + 1e-8)
    probs = np.prod(probs, axis=1)
    return probs

def compute_dr_dataset_stats(dr_vals_all, wis_weights_all, num_bootstrap=2000, alpha=0.05, rng_seed=1):
    """
    Compute dataset-level DR mean, bootstrap CI, std, and ESS (effective sample size).
    dr_vals_all: concatenated dr_vals across batches (1D np.array)
    wis_weights_all: concatenated weights (1D np.array) aligned with dr_vals_all
    """
    dr_vals_all = np.array(dr_vals_all)
    wis_weights_all = np.array(wis_weights_all)
    # dataset mean
    mean_dr = float(np.mean(dr_vals_all))

    # bootstrap CI (resampling indices)
    rng = np.random.RandomState(rng_seed)
    n = len(dr_vals_all)
    boot_means = []
    for _ in range(num_bootstrap):
        idx = rng.choice(n, n, replace=True)
        boot_means.append(np.mean(dr_vals_all[idx]))
    boot_means = np.array(boot_means)
    lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2.0)))

    # statistics
    std_dr = float(np.std(dr_vals_all, ddof=1))
    # Effective sample size from normalized weights (ESS = 1 / sum(w^2)), but ensure normalized to sum=1
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
    Karena Target = Behavior, maka Importance Weight (rho) = 1.
    Formula simplifikasi: DR_clinician = r + gamma * Q_target(s', a'_phys)
    """
    state, next_state, action_phys, next_action_phys, reward, done = batch
    
    # Pindah ke GPU
    state = state.to(device).float()
    next_state = next_state.to(device).float()
    action_phys = action_phys.to(device).float()
    next_action_phys = next_action_phys.to(device).float() # Gunakan next action asli dokter
    reward = reward.to(device).float()
    done = done.to(device).float()

    # Pilih agent evaluasi
    if hasattr(model, 'agents'):
        eval_agent = model.agents[0]
    else:
        eval_agent = model

    # --- ENCODE STATE ---
    latent_state = get_latent(model, state)
    latent_next_state = get_latent(model, next_state)

    eval_agent.critic_1.eval(); eval_agent.critic_2.eval()

    with torch.no_grad():
        # 1. Hitung Q-value untuk next state & next action ASLI (Physician)
        #    Kita gunakan Target Critic untuk stabilitas
        q1_next = eval_agent.target_critic_1(latent_next_state, next_action_phys).squeeze(1)
        q2_next = eval_agent.target_critic_2(latent_next_state, next_action_phys).squeeze(1)
        q_hat_next_phys = torch.min(q1_next, q2_next).view(-1)

        # 2. Hitung DR Clinician
        #    Formula: Q(s,a) + 1 * (r + gamma*Q(s',a') - Q(s,a)) 
        #    Ini menyederhanakan diri menjadi TD Target standard:
        dr_vals_clinician = reward + gamma * (1.0 - done) * q_hat_next_phys

    return dr_vals_clinician.cpu().numpy()


def compute_dr_clinician(model, batch, gamma=0.99, device=device):
    """
    Menghitung Doubly Robust (DR) Estimator untuk Clinician.
    Formula: r + gamma * Q_target(s', a'_phys)
    """
    state, next_state, action_phys, next_action_phys, reward, done = batch
    
    state = state.to(device).float()
    next_state = next_state.to(device).float()
    next_action_phys = next_action_phys.to(device).float()
    reward = reward.to(device).float()
    done = done.to(device).float()

    if hasattr(model, 'agents'):
        eval_agent = model.agents[0]
    else:
        eval_agent = model

    # Encode ke Latent Space
    latent_next_state = get_latent(model, next_state)

    eval_agent.target_critic_1.eval()
    eval_agent.target_critic_2.eval()

    with torch.no_grad():
        # Hitung Q-value untuk next state & next action ASLI (Physician)
        q1_next = eval_agent.target_critic_1(latent_next_state, next_action_phys).squeeze(1)
        q2_next = eval_agent.target_critic_2(latent_next_state, next_action_phys).squeeze(1)
        q_hat_next_phys = torch.min(q1_next, q2_next).view(-1)

        # DR Clinician (Rho = 1)
        dr_vals_clinician = reward + gamma * (1.0 - done) * q_hat_next_phys

    return dr_vals_clinician.cpu().numpy()

# --- UPDATED DO_TEST FUNCTION ---

def do_test(model, Xtest, actionbloctest, bloctest, Y90, SOFA, reward_value, beat):
    # =========================================================================
    # 1. TRAJECTORY GENERATION (TIDAK DIUBAH - Sama persis dengan kode Anda)
    # =========================================================================
    bloc_max = max(bloctest)
    r = np.array([reward_value, -reward_value]).reshape(1, -1)
    r2 = r * (2 * (1 - Y90.reshape(-1, 1)) - 1)
    R3 = r2[:, 0]

    RNNstate = Xtest
    print('####  Generating test set traces  ####')
    statesize = RNNstate.shape[1]
    num_samples = RNNstate.shape[0]

    # preallocate
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

    # last step
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

    # trim arrays
    bloc_num = np.squeeze(bloc_num[:c, :])
    states = states[: c, :]
    next_states = next_states[: c, :]
    actions = np.squeeze(actions[: c, :])
    next_actions = np.squeeze(next_actions[: c, :])
    rewards = np.squeeze(rewards[: c, :])
    done_flags = np.squeeze(done_flags[: c, :])

    # tensors
    state = torch.FloatTensor(states).to(device)
    next_state = torch.FloatTensor(next_states).to(device)
    action = torch.FloatTensor(actions).to(device)
    next_action = torch.FloatTensor(next_actions).to(device)
    reward = torch.FloatTensor(rewards).to(device)
    done = torch.FloatTensor(done_flags).to(device)

    # =========================================================================
    # 2. EVALUATION LOOP
    # =========================================================================
    
    # records
    rec_phys_q, rec_agent_q = [], []
    rec_agent_a, rec_phys_a, rec_sur, rec_reward_user = [], [], [], []
    rec_agent_q_pro = []  # log pi (agent)
    rec_action_mean = []
    rec_action_std = []

    # DR accumulators
    rec_dr_vals = []         # DR Agent
    rec_dr_clinician = []    # DR Clinician (New)
    rec_wis_weights = []
    rec_dr_batch_means = []
    rec_dr_batch_cis = []

    batch_size = 256
    uids = np.unique(bloc_num)
    num_batch = uids.shape[0] // batch_size
    
    save_dir = 'SACEnsemble-algorithm/'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting Evaluation on {num_batch} batches...")

    for batch_idx in range(num_batch + 1):
        batch_uids = uids[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_mask = np.isin(bloc_num, batch_uids)
        if batch_mask.sum() == 0:
            continue
        
        # Batch Tuple (Raw States)
        batch = (state[batch_mask], next_state[batch_mask], action[batch_mask],
                 next_action[batch_mask], reward[batch_mask], done[batch_mask])

        # A. FQE eval (Menggunakan do_eval yang sudah support AutoEncoder)
        q_phys, q_agent_a, agent_actions, phys_actions, agent_log_prob, action_mean, action_std = do_eval(model, batch, strategy='vote')

        # append records
        rec_phys_q.extend(q_phys.cpu().numpy())
        rec_agent_q.extend(q_agent_a.cpu().numpy())
        rec_agent_a.extend(agent_actions.cpu().numpy())
        rec_phys_a.extend(phys_actions.cpu().numpy())
        rec_sur.extend(Y90[batch_mask])
        rec_reward_user.extend(reward[batch_mask].cpu().numpy().squeeze().tolist())
        
        # agent_log_prob logic
        if agent_log_prob is not None:
            if isinstance(agent_log_prob, torch.Tensor):
                agent_log_prob = agent_log_prob.detach().cpu().numpy()
            if agent_log_prob.ndim > 1 and agent_log_prob.shape[1] == 1:
                agent_log_prob = agent_log_prob.squeeze(1)
            rec_agent_q_pro.extend(agent_log_prob.tolist())
        else:
            rec_agent_q_pro.extend([np.nan] * int(batch_mask.sum()))

        # action mean/std logic
        if action_mean is not None:
            rec_action_mean.extend(action_mean)
            rec_action_std.extend(action_std)
        else:
            rec_action_mean.extend([np.nan] * int(batch_mask.sum()))
            rec_action_std.extend([np.nan] * int(batch_mask.sum()))

        # B. Compute DR AGENT (one-step)
        dr_res = compute_dr_one_step(model, batch, gamma=0.99, device=device, num_bootstrap=200)
        rec_dr_batch_means.append(dr_res['dr_mean'])
        rec_dr_batch_cis.append(dr_res['dr_ci_batch'])
        rec_dr_vals.extend(dr_res['dr_vals'])
        rec_wis_weights.extend(dr_res['wis_weights'])

        # C. Compute DR CLINICIAN (NEW)
        dr_vals_phys = compute_dr_clinician(model, batch, gamma=0.99, device=device)
        rec_dr_clinician.extend(dr_vals_phys)
        

    # =========================================================================
    # 3. SAVING & STATS CALCULATION
    # =========================================================================
    
    # Save basic outputs
    np.save(os.path.join(save_dir, 'shencunlv.npy'), rec_sur)
    np.save(os.path.join(save_dir, 'phys_bQ.npy'), rec_phys_q)
    np.save(os.path.join(save_dir, 'agent_bQ.npy'), rec_agent_q)
    np.save(os.path.join(save_dir, 'reward.npy'), rec_reward_user)
    np.save(os.path.join(save_dir, 'agent_actionsb.npy'), rec_agent_a)
    np.save(os.path.join(save_dir, 'phys_actionsb.npy'), rec_phys_a)
    np.save(os.path.join(save_dir, 'rec_agent_q_pro.npy'), rec_agent_q_pro)
    np.save(os.path.join(save_dir, 'rec_agent_action_mean.npy'), rec_action_mean)
    np.save(os.path.join(save_dir, 'rec_agent_action_std.npy'), rec_action_std)

    print(f"Q_phys avg: {np.mean(rec_phys_q):.4f}")
    print(f"Q_agent avg: {np.mean(rec_agent_q):.4f}")

    # --- DR STATS AGENT ---
    rec_dr_vals = np.array(rec_dr_vals)
    rec_wis_weights = np.array(rec_wis_weights)
    
    np.save(os.path.join(save_dir, 'dr_vals.npy'), rec_dr_vals)
    np.save(os.path.join(save_dir, 'dr_wis_weights.npy'), rec_wis_weights)
    np.save(os.path.join(save_dir, 'dr_batch_means.npy'), np.array(rec_dr_batch_means))
    np.save(os.path.join(save_dir, 'dr_batch_cis.npy'), np.array(rec_dr_batch_cis))
    
    dr_stats = compute_dr_dataset_stats(rec_dr_vals, rec_wis_weights, num_bootstrap=2000)
    print("\n--- AGENT PERFORMANCE ---")
    print(f"DR Estimated Return (mean): {dr_stats['dr_mean']:.6f}")
    print(f"DR 95% CI: ({dr_stats['dr_ci'][0]:.6f}, {dr_stats['dr_ci'][1]:.6f})")
    print(f"DR std: {dr_stats['std_dr']:.6f}, ESS: {dr_stats['ess']:.2f}, N: {dr_stats['n']}")
    np.save(os.path.join(save_dir, 'dr_stats.npy'), dr_stats)

    # --- DR STATS CLINICIAN (NEW) ---
    rec_dr_clinician = np.array(rec_dr_clinician)
    
    # Weights untuk clinician adalah uniform (1.0) karena rho=1
    dummy_weights_phys = np.ones_like(rec_dr_clinician)
    dr_stats_phys = compute_dr_dataset_stats(rec_dr_clinician, dummy_weights_phys, num_bootstrap=2000)
    
    print("\n--- CLINICIAN PERFORMANCE (Baseline) ---")
    print(f"DR Estimated Return (mean): {dr_stats_phys['dr_mean']:.6f}")
    print(f"DR 95% CI: ({dr_stats_phys['dr_ci'][0]:.6f}, {dr_stats_phys['dr_ci'][1]:.6f})")
    
    np.save(os.path.join(save_dir, 'dr_vals_clinician.npy'), rec_dr_clinician)
    np.save(os.path.join(save_dir, 'dr_stats_clinician.npy'), dr_stats_phys)

    # =========================================================================
    # 4. WIS CALCULATION (Existing)
    # =========================================================================
    rec_phys_a = np.array(rec_phys_a)                    # [N,adim]
    rec_action_mean = np.array(rec_action_mean)          # [N,adim]
    rec_action_std = np.array(rec_action_std)            # [N,adim]
    rec_reward_user = np.array(rec_reward_user).squeeze()# [N]

    # fallback: if mean/std has NaNs
    nan_mask = np.any(np.isnan(rec_action_mean), axis=1) if rec_action_mean.size>0 else np.ones(rec_phys_a.shape[0], dtype=bool)
    if np.any(nan_mask):
        rec_agent_a = np.array(rec_agent_a)
        if rec_agent_a.shape == rec_phys_a.shape:
            rec_action_mean[nan_mask] = rec_agent_a[nan_mask]
            rec_action_std[nan_mask] = 0.1
        else:
            glob_mean = np.mean(rec_phys_a, axis=0)
            rec_action_mean = np.tile(glob_mean, (rec_phys_a.shape[0], 1))
            rec_action_std = np.tile(np.std(rec_phys_a, axis=0) + 1e-3, (rec_phys_a.shape[0], 1))

    # compute WIS
    v_wis, wis_weights, log_pi, log_mu = compute_wis_with_behavior(
        rec_phys_a, rec_action_mean, rec_action_std, rec_reward_user)

    print(f"\nWIS Estimated Return: {v_wis:.4f}")
    np.save(os.path.join(save_dir, 'wis_weights.npy'), wis_weights)
    np.save(os.path.join(save_dir, 'wis_value.npy'), np.array([v_wis]))

    v_behavior = behavior_policy_estimator(rec_reward_user)
    print(f"Behavior Policy Estimated Return: {v_behavior:.4f}")
    np.save(os.path.join(save_dir, 'behavior_policy_value.npy'), np.array([v_behavior]))

    # bootstrap CI
    wis_mean, wis_ci, wis_boot_samples = bootstrap_wis_from_logprobs(log_pi, log_mu, rec_reward_user, num_bootstrap=1000)
    print(f"WIS Mean (bootstrap): {wis_mean:.4f}")
    print(f"WIS 95% Confidence Interval: {wis_ci}")

    np.save(os.path.join(save_dir, 'wis_log_pi.npy'), log_pi)
    np.save(os.path.join(save_dir, 'wis_log_mu.npy'), log_mu)
    np.save(os.path.join(save_dir, 'wis_boot_samples.npy'), wis_boot_samples)
    # =========================================================================
    # 5. EPISODE-LEVEL EXPECTED RETURN & SURVIVAL (VALIDASI REWARD)
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

    gamma = 0.99  # discount factor RL

    for eid in episode_ids:
        idx = np.where(bloc_num_arr == eid)[0]

        # ---- Expected Return per Episode: G = sum_t gamma^t * r_t ----
        rewards_ep = rec_reward_arr[idx]
        G = 0.0
        for t, r_t in enumerate(rewards_ep):
            G += (gamma ** t) * r_t
        episode_returns.append(G)

        # ---- Survival per Episode ----
        # Asumsi dari kode awal: Y90 = 1 => death, Y90 = 0 => survive
        # Jadi kita balik: survival_flag = 1 - Y90
        y90_ep = rec_sur_arr[idx][0]        # 1 nilai untuk 1 pasien
        surv_flag = 1 - y90_ep              # survive = 1, death = 0
        episode_survival.append(surv_flag)

    episode_returns = np.array(episode_returns)
    episode_survival = np.array(episode_survival)

    np.save(os.path.join(save_dir, 'episode_returns.npy'), episode_returns)
    np.save(os.path.join(save_dir, 'episode_survival.npy'), episode_survival)

    print(f"Total Episodes           : {len(episode_returns)}")
    print(f"Mean Expected Return     : {episode_returns.mean():.4f}")
    print(f"Observed Survival Rate   : {episode_survival.mean():.4f}")

    # Korelasi langsung antara return episode dan survival episode
    if len(episode_returns) > 1 and np.std(episode_returns) > 0 and np.std(episode_survival) > 0:
        corr = np.corrcoef(episode_returns, episode_survival)[0, 1]
    else:
        corr = np.nan

    print("\n[RETURN–SURVIVAL RELATIONSHIP (EPISODE-LEVEL)]")
    print(f"Correlation(G_episode, survival): {corr:.4f}")
    if not np.isnan(corr) and corr > 0:
        print("✅ POSITIF: Expected return per episode selaras dengan survival.")
        print("   (Episode dengan return lebih tinggi cenderung lebih sering survive.)")
    else:
        print("⚠️ PERINGATAN: Korelasi tidak positif / tidak terdefinisi.")
        print("   Cek kembali reward shaping (SOFA term, death penalty, dll).")

    # Bandingkan mean return survive vs death
    if np.any(episode_survival == 1) and np.any(episode_survival == 0):
        mean_ret_survive = episode_returns[episode_survival == 1].mean()
        mean_ret_death   = episode_returns[episode_survival == 0].mean()
        print("\nMean Expected Return by Outcome:")
        print(f"  Survivors (1): {mean_ret_survive:.4f}")
        print(f"  Deaths    (0): {mean_ret_death:.4f}")
    else:
        print("\n[INFO] Outcome hanya satu kelas (semua survive atau semua death).")

    # =========================================================================
    # 6. POLICY-LEVEL COMPARISON: PHYSICIAN vs AGENT (RETURN & SURVIVAL via WIS)
    # =========================================================================
    print("\n" + "="*60)
    print(" STEP 6: POLICY-LEVEL COMPARISON (PHYSICIAN vs AGENT) ")
    print("="*60)

    # rec_sur_arr dan rec_reward_arr sudah terdefinisi di atas
    # gunakankan wis_weights dari compute_wis_with_behavior untuk agent
    wis_weights = np.asarray(wis_weights)
    wis_weights = wis_weights / (np.sum(wis_weights) + 1e-12)

    # 1) Physician (Behavior) - berdasarkan raw data
    phys_survival_rate = rec_sur_arr.mean()      # ini masih dalam definisi Y90 asli
    phys_return_avg    = rec_reward_arr.mean()   # mean reward per timestep (behavior)

    # Jika Y90 = 1 = death, konversi ke survival untuk interpretasi yang konsisten
    phys_survival_rate_true = (1 - rec_sur_arr).mean()  # survive=1, death=0

    # 2) Agent (Estimated) - Weighted oleh WIS
    #    Survival: Weighted average 1 - Y90
    agent_survival_rate_est = np.sum(wis_weights * (1 - rec_sur_arr))
    agent_return_est        = np.sum(wis_weights * rec_reward_arr)

    print(f"{'METRIC':<30} | {'PHYSICIAN (Raw)':<20} | {'AGENT (Estimated)':<20}")
    print("-" * 75)
    print(f"{'Expected Return (reward)':<30} | {phys_return_avg:<20.4f} | {agent_return_est:<20.4f}")
    print(f"{'Survival Rate (true)':<30}     | {phys_survival_rate_true:<20.4%} | {agent_survival_rate_est:<20.4%}")
    print("-" * 75)

    # Analisis arah perubahan (konsistensi dengan reward)
    return_diff   = agent_return_est - phys_return_avg
    survival_diff = agent_survival_rate_est - phys_survival_rate_true

    print("\n[ANALISIS KORELASI KUALITATIF (POLICY-LEVEL)]")
    if (return_diff > 0 and survival_diff > 0) or (return_diff < 0 and survival_diff < 0):
        print("✅ Konsisten: Perubahan Expected Return searah dengan Survival Rate.")
        print("   (Misal: Agent punya return lebih tinggi DAN survival estimasi lebih tinggi.)")
    else:
        print("⚠️ Tidak searah:")
        print("   Agent mungkin punya return lebih tinggi tapi survival estimasi tidak naik,")
        print("   atau sebaliknya. Ini indikasi reward belum sepenuhnya mencerminkan survival.")
    
    # Simpan survival estimasi agent untuk analisis lanjutan
    np.save(os.path.join(save_dir, 'est_agent_survival.npy'),
            np.array([agent_survival_rate_est]))
    np.save(os.path.join(save_dir, 'phys_survival_true.npy'),
            np.array([phys_survival_rate_true]))

def behavior_policy_estimator(rewards):
    """
    Estimasi nilai dari policy behavior berdasarkan data (mean reward).

    Output:
    - Nilai ekspektasi reward (float)
    """
    return np.mean(rewards)

# import numpy as np
# import torch
# import torch.nn.functional as F
# from scipy.stats import norm
# import os

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # ==============================================================================
# # 1. HELPER FUNCTIONS (STACKED AE COMPATIBLE)
# # ==============================================================================

# def get_padded_sequence(data, current_idx, current_bloc_num, seq_len):
#     """
#     Helper untuk membuat sequence window (5 jam terakhir) dari raw data.
#     Input: Data 2D (N, 37)
#     Output: Sequence 3D (1, 5, 37) -> Nanti di stack jadi (Batch, 5, 37)
#     """
#     feature_dim = data.shape[1]
#     available_steps = int(current_bloc_num)
    
#     if available_steps >= seq_len:
#         start_idx = current_idx - seq_len + 1
#         sequence = data[start_idx : current_idx + 1, :]
#     else:
#         valid_start = current_idx - available_steps + 1
#         valid_data = data[valid_start : current_idx + 1, :]
#         padding_len = seq_len - available_steps
#         padding = np.zeros((padding_len, feature_dim))
#         sequence = np.vstack([padding, valid_data])
#     return sequence

# def get_latent(model, state_tensor):
#     """
#     Encode state (3D Sequence) ke Latent Space (2D).
#     Stacked AutoEncoder .encode() menerima (Batch, 5, 37) dan mereturn (Batch, 64).
#     """
#     if hasattr(model, 'agents'):
#         ae = model.agents[0].autoencoder
#     else:
#         ae = model.autoencoder
        
#     if ae is not None:
#         with torch.no_grad():
#             # state_tensor: (Batch, Seq, Feat) -> encode -> (Batch, Latent)
#             return ae.encode(state_tensor)
#     return state_tensor

# def get_ensemble_q(model, latent_state, action):
#     """Rata-rata Q-value dari seluruh ensemble."""
#     if hasattr(model, 'agents'):
#         q_list = []
#         for agent in model.agents:
#             q1 = agent.critic_1(latent_state, action)
#             q2 = agent.critic_2(latent_state, action)
#             q_min = torch.min(q1, q2)
#             q_list.append(q_min)
#         return torch.stack(q_list).mean(dim=0).squeeze(1)
#     else:
#         q1 = model.critic_1(latent_state, action)
#         q2 = model.critic_2(latent_state, action)
#         return torch.min(q1, q2).squeeze(1)

# def calculate_importance_weights(phys_actions_np, agent_mean_np, agent_std_np):
#     """Hitung rho = pi / mu (dengan mu estimasi global)."""
#     # pi(a|s)
#     pi_probs = norm.pdf(phys_actions_np, loc=agent_mean_np, scale=agent_std_np + 1e-8)
#     pi_probs = np.prod(pi_probs, axis=1)
    
#     # mu(a|s) - Global Gaussian Approximation
#     mu_mean = np.mean(phys_actions_np, axis=0)
#     mu_std = np.std(phys_actions_np, axis=0) + 1e-6
#     mu_probs = norm.pdf(phys_actions_np, loc=mu_mean, scale=mu_std)
#     mu_probs = np.prod(mu_probs, axis=1)

#     rho = pi_probs / (mu_probs + 1e-12)
#     rho = np.clip(rho, 0, 100.0) # Clip agar tidak meledak
#     return rho, pi_probs, mu_probs

# def compute_dataset_stats(vals, num_bootstrap=2000, alpha=0.05):
#     vals = np.array(vals)
#     mean_val = float(np.mean(vals))
#     std_val = float(np.std(vals)) # Tambahkan Std
#     rng = np.random.RandomState(42)
#     n = len(vals)
#     boot_means = []
#     for _ in range(num_bootstrap):
#         idx = rng.choice(n, n, replace=True)
#         boot_means.append(np.mean(vals[idx]))
#     lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
#     upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2.0)))
#     return mean_val, (lower, upper), std_val # Return tuple 3 elemen

# # ==============================================================================
# # 2. EVALUATION LOGIC
# # ==============================================================================

# def do_eval(model, batchs, strategy='vote'):
#     state, next_state, action, next_action, reward, done = batchs
    
#     state = state.to(device).float()   # (B, 5, 37)
#     action = action.to(device).float() # (B, 2)

#     # Encode ke Latent (Stacked AE akan flatten di dalam)
#     latent_state = get_latent(model, state) # (B, 64)

#     with torch.no_grad():
#         # A. Q-Value Physician
#         q_value_phys = get_ensemble_q(model, latent_state, action)

#         # B. Agent Action & Stats
#         if hasattr(model, 'agents'):
#             eval_agent = model.agents[0]
#             _, log_prob, action_mean, action_std = eval_agent.actor.sample(latent_state)
            
#             # PENTING: Pass 3D state ke get_action
#             # state.cpu().numpy() sudah (B, 5, 37)
#             action_pred_np = model.get_action(state.cpu().numpy(), strategy=strategy)
#             action_pred = torch.tensor(action_pred_np, dtype=torch.float32).to(device)
#         else:
#             action_pred, log_prob, action_mean, action_std = model.actor.sample(latent_state)

#         # C. Q-Value Agent
#         q_agent = get_ensemble_q(model, latent_state, action_pred)

#     return (q_value_phys, q_agent, action_pred, action, 
#             log_prob.squeeze(1).cpu().numpy() if log_prob is not None else None,
#             action_mean.cpu().numpy() if action_mean is not None else None, 
#             action_std.cpu().numpy() if action_std is not None else None)

# def compute_dr_one_step(model, batch, gamma=0.99, device=device):
#     """Doubly Robust Estimator."""
#     state, next_state, action_phys, next_action, reward, done = batch
    
#     state = state.to(device).float()
#     next_state = next_state.to(device).float()
#     action_phys = action_phys.to(device).float()
#     reward = reward.to(device).float()
#     done = done.to(device).float()

#     latent_state = get_latent(model, state)
#     latent_next_state = get_latent(model, next_state)

#     if hasattr(model, 'agents'): policy_agent = model.agents[0]
#     else: policy_agent = model

#     with torch.no_grad():
#         # V(s)
#         action_agent, _, mean_tensor, std_tensor = policy_agent.actor.sample(latent_state)
#         v_hat_s = get_ensemble_q(model, latent_state, action_agent)

#         # V(s')
#         next_action_agent, _, _, _ = policy_agent.actor.sample(latent_next_state)
#         v_hat_next = get_ensemble_q(model, latent_next_state, next_action_agent)
        
#         # Q(s, a_phys)
#         q_hat_s_a_phys = get_ensemble_q(model, latent_state, action_phys)

#         # Weights
#         phys_np = action_phys.cpu().numpy()
#         agent_mean_np = mean_tensor.cpu().numpy()
#         agent_std_np = std_tensor.clamp(min=1e-6).cpu().numpy()
#         rho, pi_probs, mu_probs = calculate_importance_weights(phys_np, agent_mean_np, agent_std_np)
#         rho_tensor = torch.tensor(rho, device=device, dtype=torch.float32)

#         # DR Formula
#         target = reward + gamma * (1.0 - done) * v_hat_next
#         dr_vals = v_hat_s + rho_tensor * (target - q_hat_s_a_phys)
        
#     return dr_vals.cpu().numpy(), rho, pi_probs, mu_probs

# def compute_dr_clinician(model, batch, gamma=0.99, device=device):
#     state, next_state, action_phys, next_action_phys, reward, done = batch
#     next_state = next_state.to(device).float()
#     next_action_phys = next_action_phys.to(device).float()
#     reward = reward.to(device).float()
#     done = done.to(device).float()

#     latent_next_state = get_latent(model, next_state)

#     with torch.no_grad():
#         q_hat_next_phys = get_ensemble_q(model, latent_next_state, next_action_phys)
#         dr_vals = reward + gamma * (1.0 - done) * q_hat_next_phys

#     return dr_vals.cpu().numpy()

# # ==============================================================================
# # 3. MAIN TEST FUNCTION (STACKED AE VERSION)
# # ==============================================================================

# def do_test(model, Xtest, actionbloctest, bloctest, Y90, SOFA, reward_value, beat):
    
#     # --- 1. TRAJECTORY GENERATION (GENERATE 3D SEQUENCES) ---
#     print('####  Generating test set traces (Stacked Sequence)  ####')
#     SEQ_LEN = 5 # HARUS SAMA DENGAN TRAINING STACKED AE
#     statesize = Xtest.shape[1]
#     num_samples = Xtest.shape[0]
    
#     # ALOKASI 3D PENTING: (N, 5, 37)
#     states = np.zeros((num_samples, SEQ_LEN, statesize))      
#     next_states = np.zeros((num_samples, SEQ_LEN, statesize)) 
    
#     actions = np.zeros((num_samples, 2), dtype=np.float32)
#     next_actions = np.zeros((num_samples, 2), dtype=np.float32)
#     rewards = np.zeros((num_samples, 1))
#     done_flags = np.zeros((num_samples, 1))
#     bloc_num = np.zeros((num_samples, 1))

#     # Pre-calculate rewards
#     r = np.array([reward_value, -reward_value]).reshape(1, -1)
#     r2 = r * (2 * (1 - Y90.reshape(-1, 1)) - 1)
#     R3 = r2[:, 0]

#     c = 0
#     blocnum1 = 1

#     for i in range(num_samples - 1):
#         # BUAT SEQUENCE (t-4 s.d. t)
#         states[c] = get_padded_sequence(Xtest, i, bloctest[i], SEQ_LEN)
#         actions[c] = actionbloctest[i]
#         bloc_num[c] = blocnum1

#         if bloctest[i + 1] == 1: # Terminal
#             next_states[c] = np.zeros((SEQ_LEN, statesize))
#             next_actions1 = -1
#             done_flags1 = 1
#             reward1 = (-beat[0] * (SOFA[i]) + R3[i])
#             blocnum1 += 1
#         else: # Transition
#             next_states[c] = get_padded_sequence(Xtest, i + 1, bloctest[i] + 1, SEQ_LEN)
#             next_actions1 = actionbloctest[i + 1]
#             done_flags1 = 0
#             reward1 = (-beat[1] * (SOFA[i + 1] - SOFA[i]))

#         next_actions[c] = next_actions1
#         rewards[c] = reward1
#         done_flags[c] = done_flags1
#         c += 1

#     # Last row
#     states[c] = get_padded_sequence(Xtest, c, bloctest[c], SEQ_LEN)
#     actions[c] = actionbloctest[c]
#     bloc_num[c] = blocnum1
#     next_states[c] = np.zeros((SEQ_LEN, statesize))
#     next_actions[c] = -1
#     done_flags[c] = 1
#     rewards[c] = -beat[0] * (SOFA[c]) + R3[c]
#     c += 1

#     # Trim
#     bloc_num = np.squeeze(bloc_num[:c, :])
#     states = states[: c, :, :]
#     next_states = next_states[: c, :, :]
#     actions = np.squeeze(actions[: c, :])
#     next_actions = np.squeeze(next_actions[: c, :])
#     rewards = np.squeeze(rewards[: c, :])
#     done_flags = np.squeeze(done_flags[: c, :])

#     # Tensors
#     state = torch.FloatTensor(states).to(device)
#     next_state = torch.FloatTensor(next_states).to(device)
#     action = torch.FloatTensor(actions).to(device)
#     next_action = torch.FloatTensor(next_actions).to(device)
#     reward = torch.FloatTensor(rewards).to(device)
#     done = torch.FloatTensor(done_flags).to(device)

#     # --- 2. EVALUATION LOOP ---
#     rec_phys_q, rec_agent_q = [], []
#     rec_agent_a, rec_phys_a = [], []
#     rec_sur, rec_reward_user = [], []
    
#     rec_agent_q_pro = [] 
#     rec_action_mean = []
#     rec_action_std = []
#     rec_dr_agent = []
#     rec_dr_phys = []
#     rec_rho_weights = []
#     rec_log_pi = []
#     rec_log_mu = []

#     batch_size = 256
#     uids = np.unique(bloc_num)
#     num_batch = uids.shape[0] // batch_size
    
#     save_dir = 'SACEnsemble-algorithm/'
#     os.makedirs(save_dir, exist_ok=True)
    
#     print(f"Starting Evaluation on {num_batch} batches...")

#     for batch_idx in range(num_batch + 1):
#         batch_uids = uids[batch_idx * batch_size: (batch_idx + 1) * batch_size]
#         batch_mask = np.isin(bloc_num, batch_uids).flatten()
        
#         if batch_mask.sum() == 0: continue
        
#         batch = (state[batch_mask], next_state[batch_mask], action[batch_mask],
#                  next_action[batch_mask], reward[batch_mask], done[batch_mask])

#         # A. FQE Eval
#         q_phys, q_agent_a, agent_actions, phys_actions, agent_log_prob, action_mean, action_std = do_eval(model, batch, strategy='vote')

#         rec_phys_q.extend(q_phys.cpu().numpy())
#         rec_agent_q.extend(q_agent_a.cpu().numpy())
#         rec_agent_a.extend(agent_actions.cpu().numpy())
#         rec_phys_a.extend(phys_actions.cpu().numpy())
#         rec_sur.extend(Y90[batch_mask])
#         rec_reward_user.extend(reward[batch_mask].cpu().numpy().squeeze().tolist())
        
#         if agent_log_prob is not None: rec_agent_q_pro.extend(agent_log_prob.tolist())
#         else: rec_agent_q_pro.extend([np.nan] * int(batch_mask.sum()))

#         if action_mean is not None:
#             rec_action_mean.extend(action_mean)
#             rec_action_std.extend(action_std)
#         else:
#             rec_action_mean.extend([np.nan] * int(batch_mask.sum()))
#             rec_action_std.extend([np.nan] * int(batch_mask.sum()))

#         # B. DR AGENT
#         dr_vals_a, rho, pi_p, mu_p = compute_dr_one_step(model, batch)
#         rec_dr_agent.extend(dr_vals_a)
#         rec_rho_weights.extend(rho)
#         rec_log_pi.extend(np.log(pi_p + 1e-12))
#         rec_log_mu.extend(np.log(mu_p + 1e-12))

#         # C. DR CLINICIAN
#         dr_phys_vals = compute_dr_clinician(model, batch)
#         rec_dr_phys.extend(dr_phys_vals)

#     # =========================================================================
#     # 3. SAVING & STATS
#     # =========================================================================
#     rec_sur = np.array(rec_sur)
#     rec_reward_user = np.array(rec_reward_user)
#     rec_rho_weights = np.array(rec_rho_weights)
#     rec_dr_agent = np.array(rec_dr_agent)
#     rec_dr_phys = np.array(rec_dr_phys)
#     rec_agent_a = np.array(rec_agent_a)
#     rec_phys_a = np.array(rec_phys_a)
#     rec_action_mean = np.array(rec_action_mean)
#     rec_action_std = np.array(rec_action_std)

#     np.save(os.path.join(save_dir, 'shencunlv.npy'), rec_sur)
#     np.save(os.path.join(save_dir, 'phys_bQ.npy'), rec_phys_q)
#     np.save(os.path.join(save_dir, 'agent_bQ.npy'), rec_agent_q)
#     np.save(os.path.join(save_dir, 'reward.npy'), rec_reward_user)
#     np.save(os.path.join(save_dir, 'agent_actionsb.npy'), rec_agent_a)
#     np.save(os.path.join(save_dir, 'phys_actionsb.npy'), rec_phys_a)
#     np.save(os.path.join(save_dir, 'rec_agent_q_pro.npy'), rec_agent_q_pro)
#     np.save(os.path.join(save_dir, 'rec_agent_action_mean.npy'), rec_action_mean)
#     np.save(os.path.join(save_dir, 'rec_agent_action_std.npy'), rec_action_std)
#     np.save(os.path.join(save_dir, 'wis_log_pi.npy'), np.array(rec_log_pi))
#     np.save(os.path.join(save_dir, 'wis_log_mu.npy'), np.array(rec_log_mu))

#     print(f"Q_phys avg: {np.mean(rec_phys_q):.4f}")
#     print(f"Q_agent avg: {np.mean(rec_agent_q):.4f}")

#     dr_agent_mean, dr_agent_ci, _ = compute_dataset_stats(rec_dr_agent)
#     dr_phys_mean, dr_phys_ci, _   = compute_dataset_stats(rec_dr_phys)
    
#     print("\n" + "="*40)
#     print(" DOUBLY ROBUST (DR) ESTIMATION ")
#     print("="*40)
#     print(f"Agent Estimated Value : {dr_agent_mean:.4f} (95% CI: {dr_agent_ci[0]:.4f}, {dr_agent_ci[1]:.4f})")
#     print(f"Phys Estimated Value  : {dr_phys_mean:.4f} (95% CI: {dr_phys_ci[0]:.4f}, {dr_phys_ci[1]:.4f})")
    
#     np.save(os.path.join(save_dir, 'dr_vals_agent.npy'), rec_dr_agent)
#     np.save(os.path.join(save_dir, 'dr_vals_phys.npy'), rec_dr_phys)
#     np.save(os.path.join(save_dir, 'wis_weights.npy'), rec_rho_weights)

#     wis_weights_norm = rec_rho_weights / (np.sum(rec_rho_weights) + 1e-12)
#     ess = 1.0 / np.sum(wis_weights_norm ** 2)
#     v_wis = np.sum(wis_weights_norm * rec_reward_user)
    
#     print(f"\n[WIS] Estimated Return: {v_wis:.4f}")
#     print(f"[WIS] Effective Sample Size (ESS): {ess:.2f} / {len(rec_rho_weights)}")
#     np.save(os.path.join(save_dir, 'wis_value.npy'), np.array([v_wis]))

#     # --- STEP 5 & 6 (Old Code Logic) ---
#     print("\n" + "="*60)
#     print(" STEP 5: REWARD VALIDATION (EPISODE RETURN vs SURVIVAL) ")
#     print("="*60)

#     bloc_num_arr = np.array(bloc_num).flatten()
#     rec_sur_arr = np.array(rec_sur).astype(int)
#     rec_reward_arr = np.array(rec_reward_user).astype(float)

#     episode_ids = np.unique(bloc_num_arr)
#     episode_returns = []
#     episode_survival = []
#     gamma_eval = 0.99 

#     for eid in episode_ids:
#         idx = np.where(bloc_num_arr == eid)[0]
#         rewards_ep = rec_reward_arr[idx]
#         G = 0.0
#         for t, r_t in enumerate(rewards_ep):
#             G += (gamma_eval ** t) * r_t
#         episode_returns.append(G)
#         y90_ep = rec_sur_arr[idx][0]
#         episode_survival.append(1 - y90_ep)

#     episode_returns = np.array(episode_returns)
#     episode_survival = np.array(episode_survival)

#     np.save(os.path.join(save_dir, 'episode_returns.npy'), episode_returns)
#     np.save(os.path.join(save_dir, 'episode_survival.npy'), episode_survival)

#     if len(episode_returns) > 1 and np.std(episode_returns) > 0:
#         corr = np.corrcoef(episode_returns, episode_survival)[0, 1]
#     else:
#         corr = np.nan

#     print(f"Mean Expected Return     : {episode_returns.mean():.4f}")
#     print(f"Observed Survival Rate   : {episode_survival.mean():.4f}")
#     print(f"Correlation(G_ep, surv)  : {corr:.4f}")

#     print("\n" + "="*60)
#     print(" STEP 6: POLICY-LEVEL COMPARISON (PHYSICIAN vs AGENT) ")
#     print("="*60)

#     phys_return_avg = rec_reward_arr.mean()
#     phys_survival_rate_true = (1 - rec_sur_arr).mean()
#     agent_survival_rate_est = np.sum(wis_weights_norm * (1 - rec_sur_arr))
#     agent_return_est        = np.sum(wis_weights_norm * rec_reward_arr)

#     print(f"{'METRIC':<30} | {'PHYSICIAN (Raw)':<20} | {'AGENT (Estimated)':<20}")
#     print("-" * 75)
#     print(f"{'Expected Return (reward)':<30} | {phys_return_avg:<20.4f} | {agent_return_est:<20.4f}")
#     print(f"{'Survival Rate (true)':<30}     | {phys_survival_rate_true:<20.4%} | {agent_survival_rate_est:<20.4%}")
#     print("-" * 75)

#     np.save(os.path.join(save_dir, 'est_agent_survival.npy'), np.array([agent_survival_rate_est]))
#     np.save(os.path.join(save_dir, 'phys_survival_true.npy'), np.array([phys_survival_rate_true]))
    
#     print(f"\nEvaluation Complete. All files saved to {save_dir}")