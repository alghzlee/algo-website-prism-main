#!/usr/bin/env python3
"""
Upload Model Files ke Hugging Face Hub

Usage:
    1. Install: pip install huggingface_hub
    2. Login: huggingface-cli login
    3. Run: python scripts/upload_to_huggingface.py
"""

import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path

# Configuration
REPO_ID = "alghzlee/sepsis-treatment-model"  # CHANGE THIS!
REPO_TYPE = "model"
MODEL_DIR = "app/data"

# Files to upload
MODEL_FILES = [
    "best_agent_ensemble.pt",
    "best_ae_mimic.pth",
    "best_bc_mimic.pth",
    "phys_actionsb.npy",
    "agent_actionsb.npy",
    "phys_bQ.npy",
    "action_norm_stats.pkl",
    "state_norm_stats.pkl"
]

def main():
    print("=" * 60)
    print("🚀 Uploading Models to Hugging Face Hub")
    print("=" * 60)
    
    # Check if files exist
    print("\n📦 Checking files...")
    for filename in MODEL_FILES:
        filepath = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            print(f"   Please run 'git lfs pull' first!")
            return
        
        file_size = os.path.getsize(filepath)
        # Skip LFS check for .pkl files (they're legitimately small)
        if file_size < 1000 and not filename.endswith('.pkl'):  
            print(f"⚠️  {filename}: {file_size} bytes (looks like LFS pointer!)")
            print(f"   Please run 'git lfs pull' to get actual file")
            return
        else:
            print(f"✅ {filename}: {file_size / 1024 / 1024:.1f} MB")
    
    # Initialize API
    api = HfApi()
    
    # Create repository (if not exists)
    print(f"\n📝 Creating/accessing repository: {REPO_ID}")
    try:
        create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True)
        print(f"✅ Repository ready: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        print(f"   Make sure you're logged in: huggingface-cli login")
        return
    
    # Upload files
    print(f"\n📤 Uploading files...")
    for filename in MODEL_FILES:
        filepath = os.path.join(MODEL_DIR, filename)
        print(f"\n  Uploading {filename}...")
        
        try:
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
            )
            print(f"  ✅ {filename} uploaded successfully!")
        except Exception as e:
            print(f"  ❌ Error uploading {filename}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Upload Complete!")
    print("=" * 60)
    print(f"\n📝 Next steps:")
    print(f"1. Update app/routes/predict.py:")
    print(f"   HF_REPO_ID = '{REPO_ID}'")
    print(f"2. Add to requirements.txt:")
    print(f"   huggingface_hub")
    print(f"3. Update .gitignore to exclude model files")
    print(f"4. View your models: https://huggingface.co/{REPO_ID}")
    print()

if __name__ == "__main__":
    main()
