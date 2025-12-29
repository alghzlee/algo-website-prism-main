from flask import Flask, jsonify, request, Blueprint
from scipy.stats import zscore
import numpy as np
import pickle
import pandas as pd
from pydantic import BaseModel, Field
import numpy as np
import copy
import os
import random
import pandas as pd
import torch
from flask_cors import CORS
from app.data.SAC_deepQnet import EnsembleSAC, AutoEncoder
from app.services.treatment_recommendation_service import physicianAction, aiRecommendation
import numpy as np
from joblib import load
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from app.middleware.authenticate import token_required

predict_ = Blueprint('predict', __name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

physpol = np.load('app/./data/phys_actionsb.npy')

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

# Muat model SAC
model_path = 'app/data/best_agent_ensemble.pt'
loaded_model = load_model(model_path, device)

@predict_.route("/predict", methods=["POST"])
@token_required
def predict():
    try:
        input_data = request.json
        user_input = pd.DataFrame([input_data])
        user_input = user_input.apply(pd.to_numeric, errors='coerce')
        
        # loaded_model.actor.eval()
        
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
        
        # 1. LOAD STATS DARI FILE DULU
        stats_path = 'app/./data/action_norm_stats.pkl'

        # Inference action dari model
        # with torch.no_grad():
        #     norm_action, _, _, _ = loaded_model.sample(single_state)
        #     norm_action = norm_action.cpu().numpy().reshape(1, -1)

        # ===== Tambahkan inverse transform dari z-score ke dosage =====
        def inverse_transform_action(norm_action, stats_path):
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)

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
        # Ambil aksi acak dari physpol
        idx = np.random.randint(len(physpol))
        physician_action = physpol[idx]  # normalized action

        # Inverse transform dari normalized → raw
        physician_action = inverse_transform_action(physician_action.reshape(1, -1), stats_path) [0]

        print("physpol shape:", np.array(physpol).shape)
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

        # loaded_model.actor.eval()

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

        # Get action dari model: pastikan ini hasil tanh (dalam [-1, 1])
        with torch.no_grad():
            norm_action = loaded_model.get_action(single_state, strategy='mean')  # (100, 2), sudah dalam tanh output
            norm_action = norm_action.reshape(1, -1)


        # 1. LOAD STATS DARI FILE DULU
        stats_path = 'app/./data/action_norm_stats.pkl'

        with open(stats_path, 'rb') as f:
            loaded_stats = pickle.load(f)

        # Pastikan isinya benar (Dictionary)
        print("Stats Loaded:", loaded_stats)
        # Output harusnya: {'mean_log_iv': ..., 'std_log_iv': ..., dll}

        # ===== Tambahkan inverse transform dari z-score ke dosage =====
        def inverse_transform_action(norm_action, stats_path):
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)

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

        raw_action = inverse_transform_action(norm_action, stats_path)

        print(f"Predicted raw action: IV={raw_action[0,0]:.3f} ml, Vaso={raw_action[0,1]:.3f} ug/kg/min")

        return jsonify({
            "recommended_action_model_personalize": aiRecommendation(raw_action[0])
        })

    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 400