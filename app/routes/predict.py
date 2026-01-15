"""
Predict Routes - Treatment recommendation using SAC Ensemble model.
Models and data are loaded from MongoDB GridFS.
"""

from flask import Flask, jsonify, request, Blueprint
from scipy.stats import zscore
import numpy as np
import pickle
import pandas as pd
from pydantic import BaseModel, Field
import copy
import os
import random
import io
import torch
from flask_cors import CORS
from app.data.SAC_deepQnet import EnsembleSAC, AutoEncoder
from app.services.treatment_recommendation_service import physicianAction, aiRecommendation
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from app.middleware.authenticate import token_required
from app.services.gridfs_service import download_file, download_to_tempfile, file_exists

predict_ = Blueprint('predict', __name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables - lazy loaded on first request
_physpol = None
_loaded_model = None
_model_loaded = False


def load_physpol():
    """Load physpol from GridFS (lazy loading)."""
    global _physpol
    if _physpol is None:
        try:
            print("[Loader] Loading physpol from GridFS...")
            physpol_bytes = download_file('data/phys_actionsb.npy')
            _physpol = np.load(io.BytesIO(physpol_bytes))
            print(f"✓ physpol loaded, shape: {_physpol.shape}")
        except FileNotFoundError:
            print("! WARNING: phys_actionsb.npy not found in GridFS")
            _physpol = np.array([])
    return _physpol


def load_stats():
    """Load action normalization stats from GridFS."""
    try:
        stats_bytes = download_file('data/action_norm_stats.pkl')
        return pickle.loads(stats_bytes)
    except FileNotFoundError:
        raise FileNotFoundError("action_norm_stats.pkl not found in GridFS")


def load_model_from_gridfs(device):
    """
    Memuat model Ensemble SAC + Temporal VAE dari MongoDB GridFS.
    
    Args:
        device (torch.device): CPU atau CUDA
        
    Returns:
        ensemble (EnsembleSAC): Model siap pakai (sudah berisi VAE & weight terlatih)
    """
    global _loaded_model, _model_loaded
    
    if _model_loaded:
        return _loaded_model

    print("\n[Loader] Loading models from GridFS...")
    
    # Check if files exist in GridFS
    ae_exists = file_exists('data/best_ae_mimic.pth')
    model_exists = file_exists('data/best_agent_ensemble.pt')
    
    if not ae_exists:
        print("! WARNING: best_ae_mimic.pth not found in GridFS")
    if not model_exists:
        print("! WARNING: best_agent_ensemble.pt not found in GridFS")
        return None

    # ==============================================================================
    # 1. SETUP & LOAD AUTOENCODER
    # ==============================================================================
    LATENT_DIM = 24
    INPUT_DIM = 37
    NUM_AGENTS = 5
    ACTION_DIM = 2
    BC_WEIGHT = 0.25

    ae_model = AutoEncoder(input_dim=INPUT_DIM, hidden_dim=128, latent_dim=LATENT_DIM).to(device)

    # Download and load AE weights from GridFS
    try:
        temp_ae_path = download_to_tempfile('data/best_ae_mimic.pth')
        ae_model.load_state_dict(torch.load(temp_ae_path, map_location=device))
        print("✓ Pre-trained AutoEncoder loaded from GridFS.")
        # Clean up temp file
        os.unlink(temp_ae_path)
    except Exception as e:
        print(f"! WARNING: Could not load AutoEncoder: {e}")

    for param in ae_model.parameters():
        param.requires_grad = False
    ae_model.eval()

    # B. Inisialisasi Ensemble SAC
    ensemble = EnsembleSAC(
        num_agents=NUM_AGENTS, 
        state_dim=LATENT_DIM, 
        action_dim=ACTION_DIM, 
        bc_weight=BC_WEIGHT
    )
    print(ensemble)

    # --- 3. LOAD WEIGHTS from GridFS ---
    try:
        temp_model_path = download_to_tempfile('data/best_agent_ensemble.pt')
        checkpoint = torch.load(temp_model_path, map_location=device)
        # Clean up temp file
        os.unlink(temp_model_path)
        
        # A. Load VAE Weights
        ae_model.load_state_dict(checkpoint['autoencoder_state_dict'])
        print("✓ Temporal AE weights loaded from GridFS.")

        # B. Load Ensemble Weights (Actor & Critic)
        actor_dicts = checkpoint['actor_state_dicts']
        critic1_dicts = checkpoint['critic1_state_dicts']
        critic2_dicts = checkpoint['critic2_state_dicts']

        for i, agent in enumerate(ensemble.agents):
            agent.actor.load_state_dict(actor_dicts[i])
            agent.critic_1.load_state_dict(critic1_dicts[i])
            agent.critic_2.load_state_dict(critic2_dicts[i])
            
            agent.actor.to(device)
            agent.critic_1.to(device)
            agent.critic_2.to(device)
            
        print(f"✓ Ensemble weights loaded for {len(ensemble.agents)} agents.")
        
        if 'best_mean_agent_q' in checkpoint:
            print(f"  > Best Validation Q-Value recorded: {checkpoint['best_mean_agent_q']:.4f}")

    except KeyError as e:
        print(f"! ERROR: Struktur file model tidak cocok. Key hilang: {e}")
        return None
    except Exception as e:
        print(f"! ERROR Loading Model from GridFS: {e}")
        return None

    # --- 4. INTEGRASI & FREEZE ---
    ensemble.set_autoencoder(ae_model)
    
    ae_model.eval()
    for agent in ensemble.agents:
        agent.actor.eval()
        agent.critic_1.eval()
        agent.critic_2.eval()
        
    for param in ae_model.parameters(): param.requires_grad = False
    for agent in ensemble.agents:
        for param in agent.actor.parameters(): param.requires_grad = False
        for param in agent.critic_1.parameters(): param.requires_grad = False
        for param in agent.critic_2.parameters(): param.requires_grad = False

    print("[Loader] Model ready for deployment (loaded from GridFS).")
    
    _loaded_model = ensemble
    _model_loaded = True
    
    return ensemble


def get_model():
    """Get the loaded model, loading from GridFS if needed."""
    global _loaded_model, _model_loaded
    if not _model_loaded:
        _loaded_model = load_model_from_gridfs(device)
    return _loaded_model


def inverse_transform_action(norm_action):
    """Inverse transform action from normalized to raw dosage."""
    stats = load_stats()

    mean_log_iv = stats['mean_log_iv']
    std_log_iv = stats['std_log_iv']
    mean_log_vaso = stats['mean_log_vaso']
    std_log_vaso = stats['std_log_vaso']

    # Transformasi balik dari z-score ke log1p
    iv_log = norm_action[:, 0] * std_log_iv + mean_log_iv
    vaso_log = norm_action[:, 1] * std_log_vaso + mean_log_vaso
    iv_log = np.abs(iv_log)
    vaso_log = np.abs(vaso_log)
    # Transformasi balik ke domain asli
    iv_raw = np.expm1(iv_log)
    vaso_raw = np.expm1(vaso_log)
    return np.stack([iv_raw, vaso_raw], axis=1)


@predict_.route("/predict", methods=["POST"])
@token_required
def predict():
    try:
        input_data = request.json
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

        single_state = torch.tensor(processed_state, dtype=torch.float32).unsqueeze(0)
        
        # Load physpol from GridFS (lazy)
        physpol = load_physpol()
        
        if len(physpol) == 0:
            return jsonify({"error": "Model data not found in GridFS"}), 500

        # Ambil aksi acak dari physpol
        idx = np.random.randint(len(physpol))
        physician_action = physpol[idx]  # normalized action

        # Inverse transform dari normalized → raw
        physician_action = inverse_transform_action(physician_action.reshape(1, -1))[0]

        print("physpol shape:", np.array(physpol).shape)
        print("sample physpol[idx]:", physician_action)
        print("physician_action shape:", physician_action.shape)
        
        return jsonify({
            "recommended_action_clinician": physicianAction(physician_action)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@predict_.route("/predict-sac", methods=["POST"])
def predict_personalize():
    try:
        input_data = request.json
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

        single_state = torch.tensor(processed_state, dtype=torch.float32).unsqueeze(0)

        # Get model (lazy load from GridFS)
        loaded_model = get_model()
        
        if loaded_model is None:
            return jsonify({"error": "Model not loaded. Please upload models to GridFS."}), 500

        # Get action dari model
        with torch.no_grad():
            norm_action = loaded_model.get_action(single_state, strategy='mean')
            norm_action = norm_action.reshape(1, -1)

        # Load stats and inverse transform
        stats = load_stats()
        print("Stats Loaded:", stats)

        raw_action = inverse_transform_action(norm_action)

        print(f"Predicted raw action: IV={raw_action[0,0]:.3f} ml, Vaso={raw_action[0,1]:.3f} ug/kg/min")

        return jsonify({
            "recommended_action_model_personalize": aiRecommendation(raw_action[0])
        })

    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 400