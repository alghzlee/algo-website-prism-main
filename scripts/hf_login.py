#!/usr/bin/env python3
"""
Login to Hugging Face Hub

Usage:
    python3 scripts/hf_login.py
"""

from huggingface_hub import login, HfApi

def main():
    print("=" * 60)
    print("🔐 Hugging Face Login")
    print("=" * 60)
    print("\n📝 Get your token from: https://huggingface.co/settings/tokens")
    print("   (Create a 'Write' token if you don't have one)\n")
    
    token = input("Paste your token here: ").strip()
    
    if not token:
        print("❌ No token provided!")
        return
    
    try:
        print("\n🔄 Logging in...")
        login(token=token, add_to_git_credential=True)
        
        # Test connection
        api = HfApi()
        user_info = api.whoami(token=token)
        
        print(f"\n✅ Login successful!")
        print(f"   Username: {user_info['name']}")
        print(f"   Email: {user_info.get('email', 'N/A')}")
        print(f"\n💾 Token saved to: ~/.cache/huggingface/token")
        print(f"\n📝 Next step: Update REPO_ID in scripts/upload_to_huggingface.py")
        print(f"   REPO_ID = \"{user_info['name']}/sepsis-treatment-model\"")
        
    except Exception as e:
        print(f"\n❌ Login failed: {e}")
        print(f"   Please check your token and try again")

if __name__ == "__main__":
    main()
