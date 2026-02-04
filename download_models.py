#!/usr/bin/env python3
"""Download models from Hugging Face during Docker build
Updated: 2026-02-04 15:42 - Include .npy files for physician policy
"""
import os
import sys
from huggingface_hub import hf_hub_download

REPO_ID = 'alghzlee/sepsis-treatment-model'
LOCAL_DIR = 'app/data'
FILES = [
    'best_agent_ensemble.pt',    # 121.9 MB - Main SAC model
    'best_ae_mimic.pth',          # 0.1 MB - AutoEncoder
    'best_bc_mimic.pth',          # 0.3 MB - Behavior Cloning
    'phys_actionsb.npy',          # 0.4 MB - Physician policy actions
    'agent_actionsb.npy',         # 0.4 MB - Agent policy actions
    'phys_bQ.npy',                # 0.2 MB - Physician Q-values
    'action_norm_stats.pkl',      # <1 KB - Action normalization stats
    'state_norm_stats.pkl'        # <1 KB - State normalization stats
]

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f'[Docker Build] Downloading models from {REPO_ID}...')
    
    for filename in FILES:
        try:
            print(f'  - Downloading {filename}...')
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False
            )
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f'    ✓ {filename} ({size_mb:.1f} MB)')
        except Exception as e:
            print(f'    ✗ Failed: {e}')
            sys.exit(1)
    
    print('[Docker Build] All models downloaded successfully!')

if __name__ == '__main__':
    main()
