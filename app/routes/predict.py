from flask import jsonify, request, Blueprint
import numpy as np
import pickle
import pandas as pd
import os
import torch
from huggingface_hub import hf_hub_download
from app.data.SAC_deepQnet import EnsembleSAC, AutoEncoder
from app.services.treatment_recommendation_service import physicianAction, aiRecommendation
from app.middleware.authenticate import token_required

predict_ = Blueprint('predict', __name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hugging Face Configuration
HF_REPO_ID = os.getenv('HF_REPO_ID', 'alghzlee/sepsis-treatment-model')
HF_TOKEN = os.getenv('HF_TOKEN', None)  # Optional for private repos

def download_model_from_hf(filename, local_dir='app/data'):
    """
    Download model dari Hugging Face jika belum ada di local.
    
    Args:
        filename: Nama file di HF repo (e.g., 'best_agent_ensemble.pt')
        local_dir: Directory local untuk save file
    
    Returns:
        local_path: Path ke file yang sudah di-download
    """
    import time
    local_path = os.path.join(local_dir, filename)
    
    # Check if file already exists and is valid (not LFS pointer)
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        
        # LFS pointer files are typically < 200 bytes and start with "version https://git-lfs"
        if file_size < 1000:
            with open(local_path, 'rb') as f:
                header = f.read(50)
                if b'version https://git-lfs' in header:
                    print(f"[HF] Detected LFS pointer, removing and downloading from HF...")
                    os.remove(local_path)
                else:
                    # Small but valid file (like best_ae_mimic.pth ~128KB)
                    print(f"[HF] Model exists: {local_path} ({file_size/1024:.1f} KB)")
                    return local_path
        else:
            # Large file, likely valid
            print(f"[HF] Model exists: {local_path} ({file_size/1024/1024:.1f} MB)")
            return local_path
    
    # Download from Hugging Face with retry logic
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[HF] Downloading {filename} from {HF_REPO_ID} (attempt {attempt}/{max_retries})...")
            
            # Try download without force first (use cache if available)
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                token=HF_TOKEN,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            
            # Verify download success
            if os.path.exists(downloaded_path):
                file_size = os.path.getsize(downloaded_path)
                if file_size > 1000:  # Valid file
                    print(f"[HF] ✓ Downloaded: {filename} ({file_size/1024/1024:.1f} MB)")
                    return downloaded_path
                else:
                    raise Exception(f"Downloaded file too small ({file_size} bytes), likely failed")
            
        except Exception as e:
            error_msg = str(e)
            print(f"[HF] ✗ Attempt {attempt} failed: {error_msg}")
            
            if attempt < max_retries:
                print(f"[HF] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                # Final attempt failed
                print(f"[HF] ❌ All {max_retries} attempts failed")
                print(f"[HF] Repository: https://huggingface.co/{HF_REPO_ID}")
                print(f"[HF] File: {filename}")
                print(f"[HF] Please verify:")
                print(f"   1. Repository exists and is public")
                print(f"   2. File '{filename}' is uploaded")
                print(f"   3. Network connectivity to huggingface.co")
                
                # Last resort: check if file somehow exists after failed download
                if os.path.exists(local_path):
                    file_size = os.path.getsize(local_path)
                    if file_size > 1000:
                        print(f"[HF] ⚠️  Using existing file despite error: {local_path}")
                        return local_path
                
                raise FileNotFoundError(f"Failed to download {filename} after {max_retries} attempts: {error_msg}")

def inverse_transform_action(action_norm, action_stats):
    """
    Mengubah output agent (MinMax [-1, 1]) menjadi Dosis Asli (mL/mcg).

    Args:
        action_norm: Normalized action from model (numpy array or torch tensor)
        action_stats: Dictionary containing 'iv_log_max' and 'vaso_log_max'

    Logika Inverse:
    1. Clamp Action ke [-1, 1] (Safety)
    2. Inverse MinMax: [-1, 1] -> [0, LogMax]
    3. Inverse Log1p:  [0, LogMax] -> [0, RealMax]
    """
    # 1. Pastikan input adalah numpy array
    if isinstance(action_norm, torch.Tensor):
        action_norm = action_norm.detach().cpu().numpy()

    iv_log_max = action_stats['iv_log_max']
    vaso_log_max = action_stats['vaso_log_max']

    iv_act = np.clip(action_norm[:, 0], -1.0, 1.0)
    vaso_act = np.clip(action_norm[:, 1], -1.0, 1.0)

    # Rumus: log = (action + 1) / 2 * max_log
    iv_log_val = (iv_act + 1.0) / 2.0 * iv_log_max
    vaso_log_val = (vaso_act + 1.0) / 2.0 * vaso_log_max

    # Rumus: real = exp(log) - 1
    iv_ml = np.expm1(iv_log_val)
    vaso_mcg = np.expm1(vaso_log_val)

    # Kadang float error bikin hasil -0.000001, kita nol-kan
    iv_ml = np.maximum(iv_ml, 0.0)
    vaso_mcg = np.maximum(vaso_mcg, 0.0)

    return np.stack([iv_ml, vaso_mcg], axis=1)

def load_model(model_path, device):
    """
        Memuat model Ensemble SAC + Temporal VAE yang sudah disimpan untuk deployment.
        
        Args:
            model_path (str): Path ke file .pt (misal: 'SACEnsemble-algorithm/best_agent_ensemble.pt')
            device (torch.device): CPU atau CUDA
            
        Returns:
            ensemble (EnsembleSAC): Model siap pakai (sudah berisi VAE & weight terlatih)
    """

    print(f"\n[Loader] Loading model from: {model_path}")
    
    # Download from Hugging Face if needed
    model_path = download_model_from_hf(os.path.basename(model_path))
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan di: {model_path}")

    # ==============================================================================
    # 1. SETUP & LOAD AUTOENCODER
    # ==============================================================================
    LATENT_DIM = 24  # Harus sama dengan output encoder yg dilatih sebelumnya
    INPUT_DIM = 37   # Raw feature dimension
    NUM_AGENTS = 5
    ACTION_DIM = 2
    # BC weights tidak digunakan saat inference, tapi class butuh untuk init

    # Inisialisasi arsitektur AE (pastikan class AutoEncoder sudah didefinisikan di atas)
    ae_model = AutoEncoder(input_dim=INPUT_DIM, hidden_dim=128, latent_dim=LATENT_DIM).to(device)

    # Load bobot yang sudah dilatih (dari tahap sebelumnya)
    ae_path = download_model_from_hf('best_ae_mimic.pth')
    try:
        ae_model.load_state_dict(
            torch.load(ae_path, map_location=device, weights_only=False)
        )
        print("✓ Pre-trained AutoEncoder loaded successfully.")
    except FileNotFoundError:
        print("! WARNING: 'best_ae_mimic.pth' not found. Please train AutoEncoder first!")

    # Bekukan AutoEncoder (Freeze) agar tidak berubah saat training RL
    for param in ae_model.parameters():
        param.requires_grad = False
    ae_model.eval()

    
    # B. Inisialisasi Ensemble SAC
    # PENTING: state_dim harus VAE_LATENT_DIM (24), bukan raw 37
    # CRITICAL: hidden_dim HARUS 1024 (sesuai dengan checkpoint training)
    ensemble = EnsembleSAC(
        num_agents=NUM_AGENTS, 
        state_dim=LATENT_DIM, 
        action_dim=ACTION_DIM,
        hidden_dim=1024  # Must match checkpoint architecture
    )

    # --- 3. LOAD WEIGHTS ---
    try:
        # Load checkpoint ke device yang benar
        checkpoint = torch.load(model_path, map_location=device)
        
        # A. Load VAE Weights (if available in checkpoint)
        if 'autoencoder_state_dict' in checkpoint:
            ae_model.load_state_dict(checkpoint['autoencoder_state_dict'])
            print("✓ Temporal AE weights loaded from checkpoint.")
        else:
            print("⚠ Warning: 'autoencoder_state_dict' not in checkpoint, using pre-loaded AE weights")

        # B. Load Ensemble Weights (Actor & Critic)
        # Checkpoint menyimpan list of state_dicts
        actor_dicts = checkpoint['actor_state_dicts']
        critic1_dicts = checkpoint['critic1_state_dicts']
        critic2_dicts = checkpoint['critic2_state_dicts']

        for i, agent in enumerate(ensemble.agents):
            agent.actor.load_state_dict(actor_dicts[i])
            agent.critic_1.load_state_dict(critic1_dicts[i])
            agent.critic_2.load_state_dict(critic2_dicts[i])
            
            # Pindahkan agent ke device
            agent.actor.to(device)
            agent.critic_1.to(device)
            agent.critic_2.to(device)
            
        print(f"✓ Ensemble weights loaded for {len(ensemble.agents)} agents.")
        
        # Metadata check (optional)
        if 'best_mean_agent_q' in checkpoint:
            print(f"  > Best Validation Q-Value recorded: {checkpoint['best_mean_agent_q']:.4f}")

    except KeyError as e:
        import traceback
        print(f"! ERROR: Struktur file model tidak cocok. Key hilang: {e}")
        print(f"! Checkpoint keys yang ada: {list(checkpoint.keys()) if 'checkpoint' in locals() else 'N/A'}")
        print(f"! Traceback:\n{traceback.format_exc()}")
        return None
    except Exception as e:
        import traceback
        print(f"! ERROR Loading Model: {e}")
        print(f"! Traceback:\n{traceback.format_exc()}")
        return None

    # --- 4. INTEGRASI & FREEZE ---
    # Masukkan VAE ke dalam Ensemble
    ensemble.set_autoencoder(ae_model)
    
    # Set mode evaluasi untuk agents
    for agent in ensemble.agents:
        agent.actor.eval()
        agent.critic_1.eval()
        agent.critic_2.eval()

    print("[Loader] Model ready for deployment.")
    return ensemble

# ============================================================================
# MODEL CACHING - Lazy Loading dengan Singleton Pattern
# Model di-cache di memory setelah first request untuk realtime inference
# ============================================================================
_cached_model = None
_cached_physpol = None
_cached_action_stats = None

def get_model():
    global _cached_model
    if _cached_model is None:
        model_path = 'app/data/best_agent_ensemble.pt'
        print(f"[Model Cache] First request - loading model from {model_path}")
        _cached_model = load_model(model_path, device)
        
        if _cached_model is None:
            raise RuntimeError(
                "Model loading failed! Check terminal for detailed error messages. "
                "Common issues: 1) Checkpoint file corrupted, 2) Wrong keys in checkpoint, "
                "3) Model architecture mismatch"
            )
        
        print("[Model Cache] Model cached in memory for realtime inference")
    return _cached_model

def get_physpol():
    global _cached_physpol
    if _cached_physpol is None:
        print("[Model Cache] Loading physician policy data...")
        _cached_physpol = np.load('app/data/phys_actionsb.npy', allow_pickle=True)
        print(f"[Model Cache] Physician policy loaded: {_cached_physpol.shape}")
    return _cached_physpol

def get_action_stats():
    """Load action normalization statistics from pickle file.
    CRITICAL FIX: Properly loads the pickle file as dictionary.
    """
    global _cached_action_stats
    if _cached_action_stats is None:
        stats_path = 'app/data/action_norm_stats.pkl'
        print(f"[Action Stats] Loading from {stats_path}")
        with open(stats_path, 'rb') as f:
            _cached_action_stats = pickle.load(f)
        print(f"[Action Stats] Loaded: {list(_cached_action_stats.keys())}")
    return _cached_action_stats

# PREDICTION RESULT CACHING - Avoid redundant inference
import hashlib
import time as time_module

_prediction_cache = {}
_cache_ttl = 15 

def _get_input_hash(input_data):
    sorted_items = sorted(input_data.items())
    input_str = str(sorted_items)
    return hashlib.md5(input_str.encode()).hexdigest()

def _get_cached_prediction(cache_key, cache_type):
    global _prediction_cache
    cache_entry = _prediction_cache.get(f"{cache_type}_{cache_key}")
    if cache_entry:
        if time_module.time() - cache_entry['timestamp'] < _cache_ttl:
            return cache_entry['result']
        else:
            # Expired, remove from cache
            del _prediction_cache[f"{cache_type}_{cache_key}"]
    return None

def _set_cached_prediction(cache_key, cache_type, result):
    global _prediction_cache
    if len(_prediction_cache) > 100:
        oldest_keys = sorted(_prediction_cache.keys(), 
                            key=lambda k: _prediction_cache[k]['timestamp'])[:50]
        for k in oldest_keys:
            del _prediction_cache[k]
    
    _prediction_cache[f"{cache_type}_{cache_key}"] = {
        'result': result,
        'timestamp': time_module.time()
    }

# PREPROCESSING - Reusable preprocessing function
def preprocess_patient_data(input_data):
    """
    Preprocess patient data untuk inference.
    
    Args:
        input_data: Dictionary dengan patient features
        
    Returns:
        processed_state: Numpy array yang sudah di-normalize
    """
    user_input = pd.DataFrame([input_data])
    user_input = user_input.apply(pd.to_numeric, errors='coerce')
    
    reformat = user_input.values.copy()

    # Daftar kolom
    colnorm = ['SOFA', 'age', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'Temp_C',
               'Sodium', 'Chloride', 'Glucose', 'Calcium', 'Hb', 'WBC_count', 'Platelets_count',
               'PTT', 'PT', 'Arterial_pH', 'paO2', 'paCO2', 'HCO3', 'Arterial_lactate', 'Shock_Index',
               'PaO2_FiO2', 'cumulated_balance', 'CO2_mEqL', 'Ionised_Ca']
    collog = ['SpO2', 'BUN', 'Creatinine', 'SGOT',
              'Total_bili', 'INR', 'input_total', 'output_total']

    # Indeks
    colnorm_indices = np.where(np.isin(user_input.columns, colnorm))[0]
    collog_indices = np.where(np.isin(user_input.columns, collog))[0]

    if colnorm_indices.size > 0:
        reformat_colnorm = reformat[:, colnorm_indices]
        mean = np.mean(reformat_colnorm)
        std_dev = np.std(reformat_colnorm)
        reformat_colnorm = (reformat_colnorm - mean) / std_dev
    else:
        reformat_colnorm = np.zeros((reformat.shape[0], len(colnorm)))

    if collog_indices.size > 0:
        reformat_collog = reformat[:, collog_indices]
        reformat_collog = np.log(0.1 + reformat_collog)
    else:
        reformat_collog = np.zeros((reformat.shape[0], len(collog)))

    processed_state = np.hstack((reformat_colnorm, reformat_collog))

    if np.isnan(processed_state).any():
        raise ValueError("Preprocessed input contains NaNs. Check your data.")
    
    return processed_state


@predict_.route("/predict", methods=["POST"])
@token_required
def predict():
    try:
        input_data = request.json
        
        cache_key = _get_input_hash(input_data)
        cached_result = _get_cached_prediction(cache_key, 'clinician')
        if cached_result:
            return jsonify(cached_result)
        
        processed_state = preprocess_patient_data(input_data)

        # Get action stats
        action_stats = get_action_stats()

        # Get random physician action from policy (lazy loaded & cached)
        physpol = get_physpol()
        idx = np.random.randint(len(physpol))
        physician_action = physpol[idx]  # normalized action

        # Inverse transform from normalized to raw dosage
        physician_action = inverse_transform_action(physician_action.reshape(1, -1), action_stats)[0]
        
        result = {"recommended_action_clinician": physicianAction(physician_action)}
        _set_cached_prediction(cache_key, 'clinician', result)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@predict_.route("/predict-sac", methods=["POST"])
def predict_personalize():
    try:
        input_data = request.json
        
        # Check cache first
        cache_key = _get_input_hash(input_data)
        cached_result = _get_cached_prediction(cache_key, 'sac')
        if cached_result:
            return jsonify(cached_result)
        
        processed_state = preprocess_patient_data(input_data)

        # Convert to tensor dan pindahkan ke device yang benar
        state_tensor = torch.tensor(processed_state, dtype=torch.float32).unsqueeze(0).to(device)

        # Get action from model (tanh output in [-1, 1])
        model = get_model()
        with torch.no_grad():
            norm_action = model.get_action(state_tensor, strategy='mean')
            norm_action = norm_action.reshape(1, -1)

        action_stats = get_action_stats()

        # Inverse transform from normalized to raw dosage
        raw_action = inverse_transform_action(norm_action, action_stats)

        result = {"recommended_action_model_personalize": aiRecommendation(raw_action[0])}
        _set_cached_prediction(cache_key, 'sac', result)
        
        return jsonify(result)

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[ERROR] /predict-sac failed: {str(e)}")
        print(f"[ERROR] Traceback:\n{error_detail}")
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 400