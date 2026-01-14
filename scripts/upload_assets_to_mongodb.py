#!/usr/bin/env python3
"""
Script untuk upload assets ke MongoDB Atlas GridFS
Jalankan sekali dari environment lokal sebelum deploy ke Railway
"""
import os
import sys
from pymongo import MongoClient
from gridfs import GridFS
from dotenv import load_dotenv
import certifi

# Load environment variables
load_dotenv()

# Validate environment variables
MONGODB_URL = os.getenv('MONGODB_URL')
DBNAME = os.getenv('DB_NAME') or os.getenv('DB_NAME')  # Support both formats

if not MONGODB_URL or not DBNAME:
    print("ERROR: MONGODB_URL and DB_NAME must be set in .env file")
    print("Example .env:")
    print("  MONGODB_URL=mongodb+srv://test:4zJgBz5hASF3YqH@cluster0.65vwu8z.mongodb.net/?appName=Cluster0")
    print("  DB_NAME=ICU")
    sys.exit(1)

# Connect to MongoDB Atlas
print(f"Connecting to MongoDB Atlas...")
try:
    client = MongoClient(MONGODB_URL, tlsCAFile=certifi.where())
    db = client[DBNAME]
    fs = GridFS(db)
    # Test connection
    client.admin.command('ping')
    print(f"✓ Connected to database: {DBNAME}")
except Exception as e:
    print(f"ERROR: Failed to connect to MongoDB Atlas: {e}")
    sys.exit(1)

# Files to upload
files_to_upload = [
    # Images - Assets
    'app/static/src/images/assets/bed-hospital.jpg',
    'app/static/src/images/assets/x-ray.png',
    
    # Images - Profiles
    'app/static/src/images/profiles/profile.jpeg',
    'app/static/src/images/profiles/admin@icu.dev.jpeg',
    'app/static/src/images/profiles/admin@icu.dev.jpg',
    'app/static/src/images/profiles/admin@icu.dev.png',
    
    # ML Models
    'app/data/best_agent_ensemble.pt',
    'app/data/best_ae_mimic.pth',
    'app/data/action_norm_stats.pkl',
    'app/data/state_norm_stats.pkl',
    'app/data/SAC Ensemble State Norm Stats.pkl',
    'app/data/state_kmeans_model.pkl',
    'app/data/patients_kmeans_model.pkl',
    'app/data/pca_model.pkl',
    'app/data/transition_prob.pkl',
    'app/data/phys_actionsb.npy',
    'app/data/phys_bQ.npy',
    
    # CSV Data
    'app/data/df_with_readable_charttime.csv',
    'app/data/selected_data.csv',
    'app/data/sofa_indicators.csv',
    
    # JSON Data
    'app/data/similarity_data_dummy.json',
]

print(f"\nUploading {len(files_to_upload)} files to MongoDB Atlas GridFS...")
print("-" * 60)

uploaded_count = 0
skipped_count = 0
error_count = 0

for file_path in files_to_upload:
    if os.path.exists(file_path):
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        try:
            # Check if already exists
            existing = fs.find_one({'filename': filename})
            if existing:
                fs.delete(existing._id)
                action = "Replaced"
            else:
                action = "Uploaded"
            
            with open(file_path, 'rb') as f:
                fs.put(f, filename=filename)
            
            print(f"  ✓ {action}: {filename} ({file_size / 1024:.1f} KB)")
            uploaded_count += 1
            
        except Exception as e:
            print(f"  ✗ Error uploading {filename}: {e}")
            error_count += 1
    else:
        print(f"  - Skipped (not found): {file_path}")
        skipped_count += 1

print("-" * 60)
print(f"\nSummary:")
print(f"  ✓ Uploaded: {uploaded_count} files")
print(f"  - Skipped:  {skipped_count} files (not found)")
print(f"  ✗ Errors:   {error_count} files")

if uploaded_count > 0:
    print(f"\n✓ Done! Files uploaded to MongoDB Atlas.")
    print(f"  Verify in MongoDB Atlas Dashboard > Collections > fs.files")
else:
    print(f"\n⚠ No files were uploaded.")
