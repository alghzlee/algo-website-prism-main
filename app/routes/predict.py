from flask import jsonify, request, Blueprint
import numpy as np
import pickle
import pandas as pd
import os
import torch
from app.data.SAC_deepQnet import EnsembleSAC, AutoEncoder
from app.services.treatment_recommendation_service import physicianAction, aiRecommendation
from app.middleware.authenticate import token_required

predict_ = Blueprint('predict', __name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# physpol is now lazy-loaded via get_physpol() function


def inverse_transform_action(norm_action, stats_path):
    """
    Transform normalized action back to original dosage values.
    
    Args:
        norm_action: Normalized action array from model
        stats_path: Path to action_norm_stats.pkl file
    
    Returns:
        Array with raw IV and vasopressor dosages
    """
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    mean_log_iv = stats['mean_log_iv']
    std_log_iv = stats['std_log_iv']
    mean_log_vaso = stats['mean_log_vaso']
    std_log_vaso = stats['std_log_vaso']

    # Transform back from z-score to log1p
    iv_log = norm_action[:, 0] * std_log_iv + mean_log_iv
    vaso_log = norm_action[:, 1] * std_log_vaso + mean_log_vaso
    iv_log = np.abs(iv_log)
    vaso_log = np.abs(vaso_log)
    
    # Transform back to original domain
    iv_raw = np.expm1(iv_log)
    vaso_raw = np.expm1(vaso_log)
    
    return np.stack([iv_raw, vaso_raw], axis=1)

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
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan di: {model_path}")

    # ==============================================================================
    # 1. SETUP & LOAD AUTOENCODER
    # ==============================================================================
    LATENT_DIM = 24  # Harus sama dengan output encoder yg dilatih sebelumnya
    INPUT_DIM = 37   # Raw feature dimension
    NUM_AGENTS = 5
    ACTION_DIM = 2
    BC_WEIGHT = 0.25   # Tidak berpengaruh saat inferensi, tapi butuh untuk init

    # Inisialisasi arsitektur AE (pastikan class AutoEncoder sudah didefinisikan di atas)
    ae_model = AutoEncoder(input_dim=INPUT_DIM, hidden_dim=128, latent_dim=LATENT_DIM).to(device)

    # Load bobot yang sudah dilatih (dari tahap sebelumnya)
    try:
        ae_model.load_state_dict(
            torch.load('app/data/best_ae_mimic.pth', map_location=device)
        )
        print("✓ Pre-trained AutoEncoder loaded successfully.")
    except FileNotFoundError:
        print("! WARNING: 'best_ae_mimic.pth' not found. Please train AutoEncoder first!")

    # Bekukan AutoEncoder (Freeze) agar tidak berubah saat training RL
    for param in ae_model.parameters():
        param.requires_grad = False
    ae_model.eval()

    
    # B. Inisialisasi Ensemble SAC
    # PENTING: state_dim harus VAE_LATENT_DIM (64), bukan raw 37
    ensemble = EnsembleSAC(
        num_agents=NUM_AGENTS, 
        state_dim=LATENT_DIM, 
        action_dim=ACTION_DIM, 
        bc_weight=BC_WEIGHT
    )
    print(ensemble)

    # --- 3. LOAD WEIGHTS ---
    try:
        # Load checkpoint ke device yang benar
        checkpoint = torch.load(model_path, map_location=device)
        
        # A. Load VAE Weights
        ae_model.load_state_dict(checkpoint['autoencoder_state_dict'])
        print("✓ Temporal AE weights loaded.")

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
        print(f"! ERROR: Struktur file model tidak cocok. Key hilang: {e}")
        return None
    except Exception as e:
        print(f"! ERROR Loading Model: {e}")
        return None

    # --- 4. INTEGRASI & FREEZE ---
    # Masukkan VAE ke dalam Ensemble
    ensemble.set_autoencoder(ae_model)
    
    # Set mode evaluasi (Matikan Dropout, Batchnorm statistik beku)
    ae_model.eval()
    for agent in ensemble.agents:
        agent.actor.eval()
        agent.critic_1.eval()
        agent.critic_2.eval()
        
    # Freeze Gradients (Hemat memori saat inferensi)
    for param in ae_model.parameters(): param.requires_grad = False
    for agent in ensemble.agents:
        for param in agent.actor.parameters(): param.requires_grad = False
        for param in agent.critic_1.parameters(): param.requires_grad = False
        for param in agent.critic_2.parameters(): param.requires_grad = False

    print("[Loader] Model ready for deployment.")
    return ensemble

# ============================================================================
# MODEL CACHING - Lazy Loading dengan Singleton Pattern
# Model di-cache di memory setelah first request untuk realtime inference
# ============================================================================
_cached_model = None
_cached_physpol = None

def get_model():
    """
    Get cached SAC Ensemble model. Lazy loads on first call.
    Subsequent calls return cached model instantly for realtime inference.
    """
    global _cached_model
    if _cached_model is None:
        model_path = 'app/data/best_agent_ensemble.pt'
        print(f"[Model Cache] First request - loading model from {model_path}")
        _cached_model = load_model(model_path, device)
        print("[Model Cache] Model cached in memory for realtime inference")
    return _cached_model

def get_physpol():
    """
    Get cached physician policy data. Lazy loads on first call.
    """
    global _cached_physpol
    if _cached_physpol is None:
        print("[Model Cache] Loading physician policy data...")
        _cached_physpol = np.load('app/data/phys_actionsb.npy')
        print(f"[Model Cache] Physician policy loaded: {_cached_physpol.shape}")
    return _cached_physpol

# ============================================================================
# PREDICTION RESULT CACHING - Avoid redundant inference
# ============================================================================
import hashlib
import time as time_module

_prediction_cache = {}
_cache_ttl = 15  # Cache results for 15 seconds (balance between performance and responsiveness)

def _get_input_hash(input_data):
    """Generate hash from input data for caching."""
    # Create a consistent string representation
    sorted_items = sorted(input_data.items())
    input_str = str(sorted_items)
    return hashlib.md5(input_str.encode()).hexdigest()

def _get_cached_prediction(cache_key, cache_type):
    """Get cached prediction if exists and not expired."""
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
    """Cache prediction result."""
    global _prediction_cache
    # Limit cache size to prevent memory issues
    if len(_prediction_cache) > 100:
        # Remove oldest entries
        oldest_keys = sorted(_prediction_cache.keys(), 
                            key=lambda k: _prediction_cache[k]['timestamp'])[:50]
        for k in oldest_keys:
            del _prediction_cache[k]
    
    _prediction_cache[f"{cache_type}_{cache_key}"] = {
        'result': result,
        'timestamp': time_module.time()
    }


@predict_.route("/predict", methods=["POST"])
@token_required
def predict():
    try:
        input_data = request.json
        
        # Check cache first
        cache_key = _get_input_hash(input_data)
        cached_result = _get_cached_prediction(cache_key, 'clinician')
        if cached_result:
            print("[Cache] Returning cached clinician prediction")
            return jsonify(cached_result)
        
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

            # Hardcoded mean dan std
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

        single_state = torch.tensor(processed_state, dtype=torch.float32).unsqueeze(0)
        
        stats_path = 'app/./data/action_norm_stats.pkl'

        # Get random physician action from policy (lazy loaded & cached)
        physpol = get_physpol()
        idx = np.random.randint(len(physpol))
        physician_action = physpol[idx]  # normalized action

        # Inverse transform from normalized to raw dosage
        physician_action = inverse_transform_action(physician_action.reshape(1, -1), stats_path)[0]

        print("physpol shape:", np.array(physpol).shape)
        print("sample physpol[idx]:", physician_action)
        
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
            print("[Cache] Returning cached SAC prediction")
            return jsonify(cached_result)
        
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

            # Hardcoded mean dan std
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

        single_state = torch.tensor(processed_state, dtype=torch.float32).unsqueeze(0)

        # Get action from model (tanh output in [-1, 1]) - lazy loaded & cached
        model = get_model()
        with torch.no_grad():
            norm_action = model.get_action(single_state, strategy='mean')
            norm_action = norm_action.reshape(1, -1)

        stats_path = 'app/./data/action_norm_stats.pkl'

        # Inverse transform from normalized to raw dosage
        raw_action = inverse_transform_action(norm_action, stats_path)

        print(f"Predicted raw action: IV={raw_action[0,0]:.3f} ml, Vaso={raw_action[0,1]:.3f} ug/kg/min")

        result = {"recommended_action_model_personalize": aiRecommendation(raw_action[0])}
        _set_cached_prediction(cache_key, 'sac', result)
        
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 400