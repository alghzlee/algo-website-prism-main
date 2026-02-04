import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
import pickle
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. ARSITEKTUR AUTOENCODER (Improved for MIMIC) ---
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=37, hidden_dim=128, latent_dim=24, dropout_rate=0.1):
        super(AutoEncoder, self).__init__()
        # Encoder: 37 -> 64 -> 24 (Compression)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # Stabilisasi
            nn.Mish(),                # Aktivasi Modern
            nn.Dropout(dropout_rate), # Mencegah Overfitting
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Mish(),
            
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.LayerNorm(latent_dim)  # Output Latent Bersih & Ter-normalisasi
        )
        # Decoder: 24 -> 64 -> 37 (Reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Mish(),
            
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            
            nn.Linear(hidden_dim, input_dim) # Output Reconstruction (Tanpa Aktivasi)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

# --- 2. FUNGSI TRAINING KHUSUS (Menerima Xtrain & Xvalidat Numpy) ---
def train_autoencoder_mimic(X_train_np, X_val_np, input_dim=37, epochs=50, batch_size=256, lr=1e-3, save_path='best_autoencoder_mimic.pth'):
    
    # Konversi Numpy ke Tensor
    train_tensor = torch.FloatTensor(X_train_np)
    val_tensor = torch.FloatTensor(X_val_np)
    
    # Buat DataLoader
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 10
    history = {'train_loss': [], 'val_loss': []}
    noise_factor = 0.1
    print(f"\n[AutoEncoder] Start Training on {len(X_train_np)} samples...")

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch[0].to(device)
            noisy_inputs = inputs + noise_factor * torch.randn_like(inputs)
            optimizer.zero_grad()
            outputs = model(noisy_inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Scheduler & Logging
        scheduler.step(avg_val_loss)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        # --- EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"[AutoEncoder] Early stopping at epoch {epoch+1}")
                break
    
    print(f"[AutoEncoder] Training Finished. Best Model Saved to {save_path}")
    return model, history

def plot_reconstruction(model, X_data, index=0):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        original = torch.FloatTensor(X_data[index]).to(device).unsqueeze(0)
        reconstructed = model(original)
    
    orig = original.cpu().numpy().flatten()
    recon = reconstructed.cpu().numpy().flatten()
    
    plt.figure(figsize=(10, 4))
    plt.plot(orig, label='Original (Normalized)', marker='o', alpha=0.7)
    plt.plot(recon, label='Reconstructed', linestyle='--', linewidth=2)
    plt.title(f"Reconstruction Check (Sample {index})")
    plt.legend()
    plt.show()



# --- 2. BEHAVIOR CLONING (FIXED: STABILITY & TANH COMPATIBILITY) ---
class BehaviorCloning(nn.Module):
    def __init__(self, state_dim=24, action_dim=2, hidden_dim=256):
        super(BehaviorCloning, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        # CRITICAL FIX 1: Clamp min to -4 (not -10) to prevent variance collapse
        log_std = self.log_std_layer(x).clamp(-4, 2) 
        std = log_std.exp()
        return mean, std

    def get_log_prob(self, state, action):
        """
        CRITICAL FIX 2: Calculate Tanh-Corrected Log Prob.
        This matches the Agent's probability space for valid IS weights.
        """
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)

        # Inverse Tanh (Atanh) to get latent u
        action_clipped = torch.clamp(action, -0.999999, 0.999999)
        u = torch.atanh(action_clipped)

        # Gaussian Log Prob + Jacobian Correction
        log_prob_u = dist.log_prob(u)
        correction = torch.log(1 - action_clipped.pow(2) + 1e-6)
        log_prob = (log_prob_u - correction).sum(dim=-1, keepdim=True)
        
        return log_prob

def train_behavior_cloning(X_train, y_train, X_val, y_val, state_dim=24, action_dim=2, epochs=50, batch_size=256, lr=1e-3, save_path='best_bc_mimic.pth', device='cuda'):
    print(f"\n[BehaviorCloning] Start Training on {len(X_train)} samples...")
    
    # Ensure data is tensors
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    model = BehaviorCloning(state_dim, action_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device)
            optimizer.zero_grad()
            
            # Use the new Tanh-aware log_prob
            log_prob = model.get_log_prob(states, actions)
            loss = -log_prob.mean() # NLL
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for states, actions in val_loader:
                states, actions = states.to(device), actions.to(device)
                log_prob = model.get_log_prob(states, actions)
                val_loss += -log_prob.mean().item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if (epoch+1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{epochs} | Train NLL: {avg_train_loss:.4f} | Val NLL: {avg_val_loss:.4f}")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            
    return model

class Actor(nn.Module):
    # Tambahkan action_min dan action_max di init
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_min=None, action_max=None):
        super(Actor, self).__init__()
        
        # Golden Tuning: Wide Network + GELU
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        # --- LOGIKA ACTION RESCALING ---
        # if action_min is not None and action_max is not None:
            # Pastikan format tensor float32 dan masuk ke buffer model
            # Rumus: Scale = (Max - Min) / 2
            # Rumus: Bias  = (Max + Min) / 2
        self.register_buffer('action_scale', torch.tensor(1.0))
        self.register_buffer('action_bias', torch.tensor(0.0))
        print(f"[Actor] Rescaling enabled: Scale={self.action_scale}, Bias={self.action_bias}")


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(-20, 2) 
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() 
        
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob -= 2 * (np.log(2) - x_t - F.softplus(-2 * x_t))
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean, std

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # 1. ARSITEKTUR: Ganti ReLU dengan Mish
        # Q-Function surface itu sangat rumit. Mish membantu memperhalus landscape loss
        # sehingga gradien mengalir lebih baik ke layer awal.
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.Mish(), 
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 2. INISIALISASI: Orthogonal Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Gain 1.0 biasanya cukup untuk Critic, atau sqrt(2) jika ingin agresif
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        # Pastikan input dim sesuai sebelum concat
        x = torch.cat([state,  action], dim=1)
        return self.net(x)

# =============================================================================
# 5. SAC AGENT WITH LAGRANGIAN SAFETY
# =============================================================================
class SACAgent:
    def __init__(self, state_dim=24, action_dim=2, gamma=0.99, tau=0.005,
                 alpha=0.2, action_space_min=None, action_space_max=None, hidden_dim=1024):

        self.device = device
        self.actor = Actor(state_dim, action_dim, action_min=action_space_min,
                           action_max=action_space_max, hidden_dim=hidden_dim).to(device)

        self.critic_1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=3e-4)
        self.critic_1_optimizer = optim.AdamW(self.critic_1.parameters(), lr=3e-4)
        self.critic_2_optimizer = optim.AdamW(self.critic_2.parameters(), lr=3e-4)

        self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.actor_optimizer, mode='min', factor=0.5, patience=5)
        self.critic_1_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.critic_1_optimizer, mode='min', factor=0.5, patience=5)
        self.critic_2_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.critic_2_optimizer, mode='min', factor=0.5, patience=5)
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.alpha_optimizer, mode='min', factor=0.5, patience=5)

        # ===== LAGRANGIAN =====
        self.log_lambda = torch.tensor(np.log(1.0), requires_grad=True, device=device)
        self.lambda_opt = optim.Adam([self.log_lambda], lr=1e-3)
        self.target_safety = 0.15

        self.autoencoder = None
        self.norm_stats = self._load_stats()

    def compute_uncertainty_weight(self, state, action):
        with torch.no_grad():
            q1 = self.target_critic_1(state, action)
            q2 = self.target_critic_2(state, action)

            uncertainty = torch.abs(q1 - q2)
            scale = torch.abs(q1) + torch.abs(q2) + 1e-6

            u = uncertainty / scale
            return torch.clamp(u, 0.0, 1.0)

    def compute_clinician_q_target(
        self,
        next_state,
        action_clin_next,
        reward,
        done
    ):
        """
        Wu et al. Eq.(6):
        Q_clin(s,a) = r + gamma * Q_clin(s', a_clin')
        """
        with torch.no_grad():
            q1 = self.target_critic_1(next_state, action_clin_next)
            q2 = self.target_critic_2(next_state, action_clin_next)
            q_next = torch.min(q1, q2)
            q_clin = reward + (1 - done) * self.gamma * q_next
        return q_clin

    def _load_stats(self):
        """Helper internal untuk memuat statistik normalisasi"""
        try:
            with open('app/data/state_norm_stats.pkl', 'rb') as f:
                stats = pickle.load(f)
            print("✓ [SACAgent] State normalization stats loaded.")
            return stats
        except Exception as e:
            print(f"! [SACAgent] WARNING: Gagal memuat 'state_norm_stats.pkl'. Menggunakan default.")
            # Default fallback agar tidak crash (gunakan nilai kira-kira)
            return {
                'MEAN_MAP': 78.5, 'STD_MAP': 16.0, 'IDX_MAP': 6,
                'MEAN_BAL': 1500.0, 'STD_BAL': 3500.0, 'IDX_BAL': 30
            }

    def set_autoencoder(self, ae_model):
        self.autoencoder = ae_model.to(self.device)
        self.autoencoder.eval()

    def calculate_clinical_penalty(self, raw_state, action):
        # 1. Pastikan input adalah Tensor dan float
        if not isinstance(raw_state, torch.Tensor):
            raw_state = torch.tensor(raw_state, dtype=torch.float32, device=self.device)
        
        # 2. Ambil step terakhir jika sequence
        if raw_state.ndim == 3:
            current_state = raw_state[:, -1, :]
        else:
            current_state = raw_state

        # 3. Denormalisasi State (Pastikan unit fisik benar)
        # Gunakan self.norm_stats['IDX_...'] yang sudah integer
        idx_map = int(self.norm_stats["IDX_MAP"])
        idx_bal = int(self.norm_stats["IDX_BAL"])
        
        # Ekstrak & Denormalisasi
        # MAP dalam mmHg
        norm_map = current_state[:, idx_map]
        MAP = norm_map * self.norm_stats["STD_MAP"] + self.norm_stats["MEAN_MAP"]
        
        # Balance dalam mL
        norm_bal = current_state[:, idx_bal]
        BAL = norm_bal * self.norm_stats["STD_BAL"] + self.norm_stats["MEAN_BAL"]

        # 4. Hitung Pelanggaran Batas (Constraint Violation)
        # Threshold: MAP > 78, Balance > 5000
        # Relu memastikan kita hanya menghukum jika MELEBIHI batas
        violation_map = torch.relu(MAP - 78.0) / 10.0  # Scaling factor agar gradien tidak meledak
        violation_bal = torch.relu(BAL - 5000.0) / 1000.0 

        # 5. Masking dengan Action
        # Hanya menghukum jika agen MEMBERIKAN obat saat kondisi melanggar
        # Menggunakan Relu pada action untuk fokus pada pemberian dosis positif (di atas mean/bias)
        vaso_usage = torch.relu(action[:, 1])
        fluid_usage = torch.relu(action[:, 0])

        p_vaso = violation_map * vaso_usage
        p_fluid = violation_bal * fluid_usage

        # Return rata-rata batch (scalar)
        return (p_vaso + p_fluid).mean()
        
    def calculate_clinical_penalty(self, raw_state, action):
        """
        raw_state: State Fisik (sudah didenormalisasi di luar atau di dalam fungsi)
        action: Output Actor [-1, 1]
        """
        # 1. Pastikan input adalah Tensor dan float
        if not isinstance(raw_state, torch.Tensor):
            raw_state = torch.tensor(raw_state, dtype=torch.float32, device=self.device)
        
        # 2. Ambil step terakhir jika sequence
        if raw_state.ndim == 3:
            current_state = raw_state[:, -1, :]
        else:
            current_state = raw_state
        
        # 1. Load Batas Log (Dari preprocessing tadi)
        # Sebaiknya load dari self.norm_stats_action, tapi hardcode untuk kejelasan logika:
        IV_LOG_MAX = 8.294   # log1p(4000)
        VASO_LOG_MAX = 1.099 # log1p(2.0)

        # 2. INVERSE TRANSFORM (Action -> Log -> Raw)
        
        # A. Inverse MinMax: [-1, 1] -> [0, LogMax]
        # Rumus: log_val = (action + 1) / 2 * LOG_MAX
        
        # IV Fluid (Index 0)
        act_iv = torch.clamp(action[:, 0], -1.0, 1.0) # Safety clamp
        iv_log = (act_iv + 1) / 2 * IV_LOG_MAX
        
        # Vasopressor (Index 1)
        act_vaso = torch.clamp(action[:, 1], -1.0, 1.0)
        vaso_log = (act_vaso + 1) / 2 * VASO_LOG_MAX

        # B. Inverse Log1p: [0, LogMax] -> [0, RealMax]
        # Rumus: raw = exp(log_val) - 1
        iv_raw_ml = torch.expm1(iv_log)        # Range: 0 - 4000 mL
        vaso_raw_mcg = torch.expm1(vaso_log)   # Range: 0 - 2.0 mcg
        # 3. Denormalisasi State (Pastikan unit fisik benar)
        # Gunakan self.norm_stats['IDX_...'] yang sudah integer
        idx_map = int(self.norm_stats["IDX_MAP"])
        idx_bal = int(self.norm_stats["IDX_BAL"])
        
        # 3. HITUNG VIOLATION (State Constraints)
        # Ambil MAP dan Balance dari raw_state (pastikan raw_state sudah denorm)
        # Ekstrak & Denormalisasi
        # MAP dalam mmHg
        norm_map = current_state[:, idx_map]
        MAP = norm_map * self.norm_stats["STD_MAP"] + self.norm_stats["MEAN_MAP"]
        
        # Balance dalam mL
        norm_bal = current_state[:, idx_bal]
        BAL = norm_bal * self.norm_stats["STD_BAL"] + self.norm_stats["MEAN_BAL"]
        # threshold violation map > 78, bal > 5000
        
        violation_map = torch.relu(MAP - 78.0) / 10.0
        violation_bal = torch.relu(BAL - 5000.0) / 1000.0

        # 4. HITUNG PINALTI (Menggunakan NILAI FISIK)
        # Disini letak kebenaran klinisnya:
        # Jika iv_raw_ml = 0 (karena action = -1), maka pinalti = 0.
        # Jika iv_raw_ml = 2000, pinalti besar.
        
        p_fluid = violation_bal * (iv_raw_ml / 100.0) # Scaling factor 100 biar loss stabil
        p_vaso = violation_map * (vaso_raw_mcg * 10.0) 

        return (p_vaso + p_fluid).mean()

    def train(self, batches, epoch):

        # 1. Unpack Data
        (state, next_state, action, next_action,
         reward, done, bloc_num, SOFAS) = batches

        # 2. Pindahkan semua ke GPU
        state = state.clone().detach().float().to(self.device)
        next_state = next_state.clone().detach().float().to(self.device)
        action = action.clone().detach().float().to(self.device)
        reward = reward.clone().detach().float().to(self.device)
        done = done.clone().detach().float().to(self.device)
        bloc_num = torch.tensor(bloc_num).long().to(self.device)
        sofa = torch.tensor(SOFAS).float().to(self.device)

        batch_size = 128
        uids = torch.unique(bloc_num)
        shuffled_indices = torch.randperm(len(uids))
        uids = uids[shuffled_indices]
        num_batches = len(uids) // batch_size
        
        record_loss = []
        rec_rewards = []
        
        for batch_idx in range(num_batches + 1):
            batch_uids = uids[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_mask = torch.isin(bloc_num, batch_uids)

            # A. AMBIL RAW DATA (Wajib 37 dimensi)
            batch_state_raw = state[batch_mask]       
            batch_next_state_raw = next_state[batch_mask]
            
            batch_action = action[batch_mask] 
            batch_reward = reward[batch_mask].unsqueeze(1)
            batch_done = done[batch_mask].unsqueeze(1)
            batch_sofas = sofa[batch_mask].unsqueeze(1)
            b_action_clin = batch_action.to(self.device)    

            # B. ENCODING KE LATENT (Untuk Masuk Neural Network)
            if self.autoencoder:
                with torch.no_grad():
                    # Encode Raw -> Latent (24 dim)
                    batch_state_latent = self.autoencoder.encode(batch_state_raw)
                    batch_next_state_latent = self.autoencoder.encode(batch_next_state_raw)
            else:
                # Fallback untuk Standard AE tanpa encoder class
                batch_state_latent = batch_state_raw[:, -1, :] if batch_state_raw.ndim == 3 else batch_state_raw
                batch_next_state_latent = batch_next_state_raw[:, -1, :] if batch_next_state_raw.ndim == 3 else batch_next_state_raw

            # === SAC TARGET (STANDARD) ===
            with torch.no_grad():
                a_next, logp_next, _, _ = self.actor.sample(batch_next_state_latent)
                q_next = torch.min(
                    self.target_critic_1(batch_next_state_latent, a_next),
                    self.target_critic_2(batch_next_state_latent, a_next)
                ) - self.alpha * logp_next

                q_sac_target = batch_reward + (1 - batch_done) * self.gamma * q_next

            # === CLINICIAN Q TARGET ===
            with torch.no_grad():
                q1_clin = self.target_critic_1(batch_next_state_latent, b_action_clin)
                q2_clin = self.target_critic_2(batch_next_state_latent, b_action_clin)
                q_clin_target = batch_reward + (1 - batch_done) * self.gamma * torch.min(q1_clin, q2_clin)

            # === CURRENT Q ===
            q1 = self.critic_1(batch_state_latent, b_action_clin)
            q2 = self.critic_2(batch_state_latent, b_action_clin)

            loss_sac = F.mse_loss(q1, q_sac_target) + F.mse_loss(q2, q_sac_target)

            # === UNCERTAINTY-AWARE CLINICIAN GUIDANCE ===
            uncertainty = self.compute_uncertainty_weight(batch_next_state_latent, b_action_clin)
            is_mild = (batch_sofas < 5.0).float().unsqueeze(1)

            w_clin = is_mild * (1.0 - uncertainty)

            lambda_clin = 0.3  # hyperparameter

            loss_clin = (
                w_clin *
                (F.mse_loss(q1, q_clin_target, reduction='none') +
                F.mse_loss(q2, q_clin_target, reduction='none'))
            ).mean()

            critic_loss = loss_sac + lambda_clin * loss_clin

            q1 = self.critic_1(batch_state_latent, batch_action)
            q2 = self.critic_2(batch_state_latent, batch_action)

            # === OPTIMIZE CRITIC ===
            self.critic_1_optimizer.zero_grad(set_to_none=True)
            self.critic_2_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_1_optimizer.step()
            self.critic_2_optimizer.step()
         

            # ================= ACTOR UPDATE =================
            # Gunakan LATENT untuk sample action
            new_action, log_prob, _, _ = self.actor.sample(batch_state_latent)
            
            q1_new = self.critic_1(batch_state_latent, new_action)
            q2_new = self.critic_2(batch_state_latent, new_action)
            
            sac_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()
        
            # --- Gunakan RAW STATE untuk hitung pinalti ---
            # batch_state_raw ukurannya (B, 37) atau (B, 5, 37), aman untuk index 27
            safety_penalty = self.calculate_clinical_penalty(batch_state_raw, new_action)
            
            lambda_val = torch.exp(self.log_lambda)
            actor_loss = sac_loss + (lambda_val.detach() * safety_penalty)
         
            rec_rewards.append(batch_reward.detach().cpu().numpy())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # ================= ALPHA UPDATE =================
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

            # ----- Lambda -----
            delta = safety_penalty.detach() - self.target_safety
            delta = torch.clamp(delta, min=-0.02)  # deadzone toleransi
            lambda_loss = -lambda_val * delta
            # lambda_loss = -lambda_val * (safety_penalty.detach() - self.target_safety)
            self.lambda_opt.zero_grad()
            lambda_loss.backward()
            self.lambda_opt.step()
            # self.log_lambda.data.clamp_(np.log(1e-6), np.log(100))
            self.log_lambda.data.clamp_(np.log(0.1), np.log(10.0))
            self.soft_update()

            # avg_loss = (critic_1_loss + critic_2_loss + actor_loss).item() / 3
            avg_loss = (critic_loss + actor_loss).item() / 3
            record_loss.append(avg_loss)

            if batch_idx % 25 == 0:
                if len(record_loss) > 0:
                    curr_loss = np.mean(record_loss)
                    self.actor_scheduler.step(curr_loss)
                    self.critic_1_scheduler.step(curr_loss)
                    self.critic_2_scheduler.step(curr_loss)
                    self.alpha_scheduler.step(curr_loss)
                
                temp_rewards = np.concatenate(rec_rewards).squeeze()
                est_reward = np.mean(temp_rewards)
                # Print Status 
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {avg_loss:.4f}, Safety: {safety_penalty.item():.4f}, Rew: {est_reward:.2f}, Lambda: {lambda_val.item():.4f}")

        return record_loss

    def soft_update(self):
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_action(self, state, deterministic=True):
        state = torch.tensor(state).float().to(self.device)
        if state.ndim == 1: state = state.unsqueeze(0)
        
        # Encode jika ada AE
        if self.autoencoder:
            with torch.no_grad():
                state = self.autoencoder.encode(state)

        with torch.no_grad():
            mean, std = self.actor(state)
            if deterministic:
                action = torch.tanh(mean)
            else:
                dist = torch.distributions.Normal(mean, std)
                action = torch.tanh(dist.sample())
        return action.cpu().numpy()
# --- 4. FIXED ENSEMBLE SAC ---
class EnsembleSAC:
    # Update init arguments
    def __init__(self, num_agents=5, state_dim=24, action_dim=2, 
                 action_space_min=None, action_space_max=None,hidden_dim=256, bc_weight_init=1.0, bc_decay_tau=10.0, bc_stop_epoch=15): # <--- PARAMETER BARU
        
        self.num_agents = num_agents
        # Teruskan ke SACAgent
        self.agents = [
            SACAgent(state_dim, action_dim,
                     action_space_min=action_space_min, action_space_max=action_space_max, hidden_dim=hidden_dim) 
            for _ in range(num_agents)
        ]


    def set_autoencoder(self, autoencoder):
        for agent in self.agents:
            agent.set_autoencoder(autoencoder)

    def train(self, batches, epoch):
        losses = []
        for agent in self.agents:
            l = agent.train(batches, epoch)
            losses.append(l)
        return losses

    def get_action(self, state, strategy='median'):
        if isinstance(state, list):
            state = np.array(state)

        all_actions = []
        for agent in self.agents:
            action = agent.get_action(state, deterministic=True)
            all_actions.append(action)

        all_actions = np.array(all_actions)

        if strategy == 'median':
            return np.median(all_actions, axis=0)

        elif strategy == 'mean':
            return np.mean(all_actions, axis=0)

        elif strategy == 'lcb':
            mean_act = np.mean(all_actions, axis=0)
            std_act = np.std(all_actions, axis=0)
            return mean_act - 0.5 * std_act

        else:
            # default = safest
            return np.median(all_actions, axis=0)