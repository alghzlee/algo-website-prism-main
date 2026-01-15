# DOKUMENTASI LENGKAP
## Optimisasi Docker Image & Implementasi MongoDB Atlas GridFS
### Project: PRISM - ICU Monitoring System
### Tanggal: 15 Januari 2026

---

# DAFTAR ISI

1. [Ringkasan Eksekutif](#1-ringkasan-eksekutif)
2. [Optimisasi Docker Image](#2-optimisasi-docker-image)
3. [Implementasi MongoDB Atlas GridFS](#3-implementasi-mongodb-atlas-gridfs)
4. [Perbaikan Fitur Profile Photo](#4-perbaikan-fitur-profile-photo)
5. [Daftar File yang Dimodifikasi](#5-daftar-file-yang-dimodifikasi)
6. [Panduan Deployment](#6-panduan-deployment)
7. [Troubleshooting](#7-troubleshooting)

---

# 1. RINGKASAN EKSEKUTIF

## 1.1 Latar Belakang

Project PRISM adalah sistem monitoring ICU yang dibangun menggunakan Flask, PyTorch, dan MongoDB. Project ini perlu di-deploy ke Railway, namun menghadapi beberapa kendala:

1. **Ukuran Docker image terlalu besar** (>4GB) - melebihi batas Railway
2. **File statis (gambar) tidak muncul** di Docker container
3. **Profile photo tidak bisa diupload/diubah** di environment production

## 1.2 Hasil yang Dicapai

| Aspek | Sebelum | Sesudah |
|-------|---------|---------|
| Ukuran Docker Image | >4 GB | 424 MB |
| Pengurangan Ukuran | - | ~90% |
| Gambar Statis | Tidak muncul | ✅ Muncul |
| Upload Profile Photo | Error | ✅ Berhasil |
| File Storage | Filesystem lokal | MongoDB Atlas GridFS |

---

# 2. OPTIMISASI DOCKER IMAGE

## 2.1 Analisis Masalah

### Penyebab Ukuran Image Besar:

1. **PyTorch dengan CUDA** (~2.5 GB)
   - Default installation PyTorch menyertakan CUDA dependencies
   - Railway menggunakan CPU-only, sehingga CUDA tidak diperlukan

2. **Single-stage Docker build**
   - Build dependencies tetap ada di final image
   - Tidak ada optimasi layer

3. **File model ter-copy ke image**
   - File `.pkl`, `.pt`, `.npy` ter-copy karena tidak di-exclude

## 2.2 Solusi yang Diimplementasikan

### 2.2.1 Multi-Stage Docker Build

```dockerfile
# ===== BUILD STAGE =====
FROM python:3.11-slim AS builder
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch CPU-only (menghemat ~1.7GB)
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ===== PRODUCTION STAGE =====
FROM python:3.11-slim AS production
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . .

EXPOSE 8000

# Run with gunicorn + eventlet worker
CMD ["gunicorn", "-k", "eventlet", "-w", "1", "-b", "0.0.0.0:8000", "wsgi:app"]
```

### 2.2.2 Update requirements.txt

**Perubahan:**
- Menghapus `torch==2.6.0` (di-install terpisah dengan CPU-only flag)
- Menghapus duplikat `PyJWT`
- Menghapus `Werkzeug==3.1.5` (biarkan Flask memilih versi kompatibel)
- Menambahkan `eventlet==0.38.2` untuk async support
- Menambahkan `certifi>=2023.0.0` untuk SSL MongoDB Atlas

**requirements.txt Final:**
```
Flask==3.1.2
Flask_Cors==5.0.0
Flask_SocketIO==5.5.1
eventlet==0.38.2
joblib==1.4.2
matplotlib==3.10.8
numpy==2.4.1
pandas==2.3.3
pydantic==2.12.5
PyJWT==2.10.1
pymongo==4.16.0
python-dotenv==1.2.1
scikit_learn==1.8.0
scipy==1.17.0
gunicorn==23.0.0
certifi>=2023.0.0
```

### 2.2.3 Update .dockerignore

**File yang di-exclude:**
```
# Version Control
.git
.gitignore
.github/

# Environment
.env
.env.*

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/

# Node.js
node_modules/
app/static/src/
package.json
package-lock.json
tailwind.config.js

# Model files - PENTING!
*.pt
*.pth
*.pkl
*.npy
*.h5
*.onnx
*.safetensors

# Data files
*.csv
*.zip

# IDE
.DS_Store
.idea/
.vscode/

# Documentation
*.md
!README.md
```

### 2.2.4 Update wsgi.py

Menambahkan `eventlet.monkey_patch()` di awal file:

```python
import eventlet
eventlet.monkey_patch()

from app import create_app
from app.extensions import socketio
import os

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001)) 
    socketio.run(app, host='0.0.0.0', port=port, debug=True, allow_unsafe_werkzeug=True)
```

---

# 3. IMPLEMENTASI MONGODB ATLAS GRIDFS

## 3.1 Latar Belakang

Setelah optimisasi Docker, file statis (gambar) tidak muncul karena:
- Folder `app/static/src/` di-exclude oleh `.dockerignore`
- File model (`.pkl`, `.pt`) juga di-exclude

**Solusi:** Menyimpan file statis di MongoDB Atlas menggunakan GridFS.

## 3.2 Arsitektur

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Flask App     │────▶│  Assets Route   │────▶│  MongoDB Atlas  │
│   (Docker)      │     │  /assets/...    │     │  (GridFS)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                   │
                                                   ├── fs.files
                                                   └── fs.chunks
```

## 3.3 File yang Dibuat

### 3.3.1 app/routes/assets.py

Route untuk serve file dari MongoDB GridFS:

```python
from flask import Blueprint, Response, abort, current_app
from gridfs import GridFS
from pymongo import MongoClient
import certifi

assets_ = Blueprint('assets', __name__)

def get_gridfs():
    """Get GridFS instance from MongoDB Atlas"""
    client = MongoClient(
        current_app.config['MONGODB_URL'],
        tlsCAFile=certifi.where()
    )
    db = client[current_app.config['DBNAME']]
    return GridFS(db)

@assets_.route('/assets/images/<path:filename>')
def serve_image(filename):
    """Serve images from MongoDB Atlas GridFS"""
    fs = get_gridfs()
    grid_file = fs.find_one({'filename': filename})
    
    if not grid_file:
        abort(404)
    
    ext = filename.rsplit('.', 1)[-1].lower()
    content_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'svg': 'image/svg+xml',
        'webp': 'image/webp',
    }
    content_type = content_types.get(ext, 'application/octet-stream')
    
    return Response(
        grid_file.read(),
        content_type=content_type,
        headers={'Cache-Control': 'public, max-age=31536000'}
    )
```

### 3.3.2 app/services/file_storage.py

Service untuk operasi GridFS:

```python
from pymongo import MongoClient
from gridfs import GridFS
from flask import current_app
import os
import certifi

def get_gridfs():
    """Get GridFS instance from MongoDB Atlas"""
    client = MongoClient(
        current_app.config['MONGODB_URL'],
        tlsCAFile=certifi.where()
    )
    db = client[current_app.config['DBNAME']]
    return GridFS(db)

def upload_file(file_path, filename=None):
    """Upload file to MongoDB Atlas GridFS"""
    fs = get_gridfs()
    if filename is None:
        filename = os.path.basename(file_path)
    
    with open(file_path, 'rb') as f:
        file_id = fs.put(f, filename=filename)
    
    return file_id

def get_file(filename):
    """Get file from MongoDB Atlas GridFS"""
    fs = get_gridfs()
    return fs.find_one({'filename': filename})

def download_file(filename):
    """Download file content from GridFS"""
    fs = get_gridfs()
    grid_file = fs.find_one({'filename': filename})
    if grid_file:
        return grid_file.read()
    return None
```

### 3.3.3 scripts/upload_assets_to_mongodb.py

Script untuk upload file ke MongoDB Atlas:

```python
#!/usr/bin/env python3
import os
import sys
from pymongo import MongoClient
from gridfs import GridFS
from dotenv import load_dotenv
import certifi

load_dotenv()

MONGODB_URL = os.getenv('MONGODB_URL')
DBNAME = os.getenv('DB_NAME') or os.getenv('DBNAME')

if not MONGODB_URL or not DBNAME:
    print("ERROR: MONGODB_URL and DB_NAME must be set in .env file")
    sys.exit(1)

client = MongoClient(MONGODB_URL, tlsCAFile=certifi.where())
db = client[DBNAME]
fs = GridFS(db)

files_to_upload = [
    # Images
    'app/static/src/images/assets/bed-hospital.jpg',
    'app/static/src/images/assets/x-ray.png',
    'app/static/src/images/profiles/profile.jpeg',
    
    # Models
    'app/data/best_agent_ensemble.pt',
    'app/data/best_ae_mimic.pth',
    'app/data/action_norm_stats.pkl',
    # ... dll
]

for file_path in files_to_upload:
    if os.path.exists(file_path):
        filename = os.path.basename(file_path)
        existing = fs.find_one({'filename': filename})
        if existing:
            fs.delete(existing._id)
        with open(file_path, 'rb') as f:
            fs.put(f, filename=filename)
        print(f"Uploaded: {filename}")
```

## 3.4 Update app/__init__.py

Registrasi assets blueprint dan koneksi MongoDB Atlas dengan SSL:

```python
from flask import Flask
from config import Config
from app.extensions import socketio
from pymongo import MongoClient
from flask_cors import CORS
import certifi

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Connect to MongoDB Atlas with SSL certificate
    client = MongoClient(app.config['MONGODB_URL'], tlsCAFile=certifi.where())
    app.db = client[app.config['DBNAME']]
    
    # Assets route for serving files from MongoDB GridFS
    from .routes.assets import assets_
    app.register_blueprint(assets_)
    
    # ... other blueprints
    
    return app
```

## 3.5 Update Template References

### Sebelum:
```html
<img src="{{url_for('static', filename='src/images/assets/bed-hospital.jpg')}}" />
```

### Sesudah:
```html
<img src="{{url_for('assets.serve_image', filename='bed-hospital.jpg')}}" />
```

**Template yang diupdate:**
- `app/templates/home/bed-selection.html` (8 gambar)
- `app/templates/patient-monitoring/detail.html` (1 gambar)

---

# 4. PERBAIKAN FITUR PROFILE PHOTO

## 4.1 Masalah

1. Profile photo disimpan ke filesystem lokal (`app/./static/src/images/profiles/`)
2. Filesystem lokal tidak tersedia di Docker container
3. Template menggunakan `url_for('static', ...)` yang tidak bisa akses GridFS

## 4.2 Solusi

### 4.2.1 Update app/routes/profile.py

Upload profile photo ke MongoDB GridFS:

```python
from gridfs import GridFS
from pymongo import MongoClient
import certifi

def get_gridfs():
    client = MongoClient(
        current_app.config['MONGODB_URL'],
        tlsCAFile=certifi.where()
    )
    db = client[current_app.config['DBNAME']]
    return GridFS(db)

@profile_.route('/update-profile', methods=["POST"])
@token_required
def update_profile():
    # ... validation code ...
    
    if "filePict" in request.files:
        file = request.files["filePict"]
        if file.filename:
            filename = secure_filename(file.filename)
            extension = filename.split(".")[-1].lower()
            
            # Generate unique filename for GridFS
            gridfs_filename = f"profile_{email.replace('@', '_at_').replace('.', '_')}.{extension}"
            
            fs = get_gridfs()
            
            # Delete existing profile picture
            existing = fs.find_one({'filename': gridfs_filename})
            if existing:
                fs.delete(existing._id)
            
            # Upload new file
            fs.put(file, filename=gridfs_filename)
            
            newDoc["profilePict"] = gridfs_filename
    
    # Update database
    current_app.db.users.update_one({"email": email}, {"$set": newDoc})
```

### 4.2.2 Update Template dengan Fallback

Semua template yang menampilkan profile photo diupdate dengan format:

```html
<img 
  src="{% if user_info.profilePict %}{{ url_for('assets.serve_image', filename=user_info.profilePict) }}{% else %}https://ui-avatars.com/api/?name={{ user_info.username }}&background=0891b2&color=fff&size=150{% endif %}"
  onerror="this.src='https://ui-avatars.com/api/?name={{ user_info.username }}&background=0891b2&color=fff&size=150'"
/>
```

**Keterangan:**
- Jika `profilePict` ada → tampilkan dari GridFS
- Jika tidak ada → tampilkan avatar otomatis dari UI Avatars
- Jika error loading → fallback ke UI Avatars

**Template yang diupdate:**
| Template | Jumlah Lokasi |
|----------|---------------|
| profile.html | 1 |
| bed-selection.html | 2 |
| user-settings.html | 2 |
| navbar.html | 1 |
| **Total** | **6** |

---

# 5. DAFTAR FILE YANG DIMODIFIKASI

## 5.1 File Baru

| File | Deskripsi |
|------|-----------|
| `app/routes/assets.py` | Route untuk serve file dari GridFS |
| `app/services/file_storage.py` | Service helper GridFS |
| `app/services/__init__.py` | Package init |
| `scripts/upload_assets_to_mongodb.py` | Script upload file ke Atlas |

## 5.2 File yang Dimodifikasi

| File | Perubahan |
|------|-----------|
| `Dockerfile` | Multi-stage build, PyTorch CPU-only |
| `requirements.txt` | Hapus torch, tambah eventlet + certifi |
| `.dockerignore` | Tambah pattern untuk model files |
| `wsgi.py` | Tambah eventlet.monkey_patch() |
| `config.py` | Tidak berubah |
| `app/__init__.py` | Register assets blueprint, SSL connection |
| `app/routes/profile.py` | Upload foto ke GridFS |
| `app/templates/home/bed-selection.html` | Update 10 image references |
| `app/templates/patient-monitoring/detail.html` | Update 1 image reference |
| `app/templates/profile/profile.html` | Update profile image |
| `app/templates/profile/user-settings.html` | Update 2 image references |
| `app/templates/component/navbar.html` | Update 1 image reference |

---

# 6. PANDUAN DEPLOYMENT

## 6.1 Prerequisites

1. MongoDB Atlas cluster dengan database `ICU`
2. Railway account
3. Docker Desktop terinstall (untuk testing lokal)

## 6.2 Environment Variables

Pastikan `.env` file berisi:

```
SECRET_KEY=<your-secret-key>
TOKEN_KEY=<your-token-key>
MONGODB_URL=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?appName=Cluster0
DB_NAME=ICU
```

## 6.3 Langkah Deployment

### Step 1: Upload Assets ke MongoDB Atlas

```bash
# Dari folder project
source venv/bin/activate
python scripts/upload_assets_to_mongodb.py
```

### Step 2: Build Docker Image

```bash
docker build --platform linux/amd64 -t algo-prism-test .
```

### Step 3: Test Lokal

```bash
docker run --platform linux/amd64 -p 8000:8000 --env-file .env algo-prism-test
```

Buka `http://localhost:8000/` untuk testing.

### Step 4: Deploy ke Railway

1. Push code ke GitHub repository
2. Connect repository ke Railway
3. Set environment variables di Railway dashboard
4. Deploy!

## 6.4 Konfigurasi MongoDB Atlas

### Whitelist IP Address

1. Buka MongoDB Atlas Dashboard
2. Pergi ke **Network Access** → **IP Access List**
3. Klik **Add IP Address**
4. Pilih **Allow Access from Anywhere** (`0.0.0.0/0`)

### Verifikasi Upload

1. Buka MongoDB Atlas Dashboard
2. Klik **Browse Collections**
3. Cari collection `fs.files` dan `fs.chunks`
4. Pastikan semua file sudah ter-upload

---

# 7. TROUBLESHOOTING

## 7.1 Gambar Tidak Muncul

**Kemungkinan Penyebab:**
1. File belum diupload ke GridFS
2. Config key salah (`DB_NAME` vs `DBNAME`)
3. SSL certificate error

**Solusi:**
```bash
# Test koneksi dari Python
python -c "
from app import create_app
app = create_app()
with app.app_context():
    from app.routes.assets import get_gridfs
    fs = get_gridfs()
    files = list(fs.find())
    print(f'Found {len(files)} files')
"
```

## 7.2 Profile Photo Error

**Kemungkinan Penyebab:**
1. GridFS connection error
2. File size terlalu besar
3. Extension tidak didukung

**Solusi:**
- Check console log untuk error message
- Pastikan file < 16MB

## 7.3 Docker Build Error

**"No space left on device":**
```bash
docker system prune -af
docker builder prune -af
```

**Dependency conflict:**
```bash
# Rebuild tanpa cache
docker build --no-cache --platform linux/amd64 -t algo-prism-test .
```

---

# LAMPIRAN

## A. Struktur Folder Project

```
algo-website-prism-main/
├── app/
│   ├── __init__.py
│   ├── extensions.py
│   ├── data/
│   │   ├── *.pkl, *.pt, *.csv (model files)
│   ├── routes/
│   │   ├── assets.py          ← BARU
│   │   ├── auth.py
│   │   ├── profile.py         ← MODIFIED
│   │   └── ...
│   ├── services/
│   │   ├── __init__.py        ← BARU
│   │   ├── file_storage.py    ← BARU
│   │   └── ...
│   ├── static/
│   │   └── src/images/        ← Di-exclude dari Docker
│   ├── templates/
│   │   ├── component/
│   │   │   └── navbar.html    ← MODIFIED
│   │   ├── home/
│   │   │   └── bed-selection.html ← MODIFIED
│   │   ├── patient-monitoring/
│   │   │   └── detail.html    ← MODIFIED
│   │   └── profile/
│   │       ├── profile.html   ← MODIFIED
│   │       └── user-settings.html ← MODIFIED
│   └── websockets/
├── scripts/
│   └── upload_assets_to_mongodb.py ← BARU
├── .dockerignore              ← MODIFIED
├── .env
├── config.py
├── Dockerfile                 ← MODIFIED
├── requirements.txt           ← MODIFIED
└── wsgi.py                    ← MODIFIED
```

## B. Daftar File di MongoDB GridFS

| Filename | Type | Size |
|----------|------|------|
| bed-hospital.jpg | Image | ~50KB |
| x-ray.png | Image | ~100KB |
| profile.jpeg | Image | ~20KB |
| best_agent_ensemble.pt | Model | 8.4MB |
| best_ae_mimic.pth | Model | 67KB |
| action_norm_stats.pkl | Data | 110B |
| state_norm_stats.pkl | Data | 118B |
| state_kmeans_model.pkl | Model | 282KB |
| patients_kmeans_model.pkl | Model | 67KB |
| pca_model.pkl | Model | 161KB |
| transition_prob.pkl | Data | 8.1MB |
| phys_actionsb.npy | Data | 379KB |
| phys_bQ.npy | Data | 189KB |
| df_with_readable_charttime.csv | Data | 60KB |
| selected_data.csv | Data | 5.5KB |
| sofa_indicators.csv | Data | 4.3KB |
| similarity_data_dummy.json | Data | 12KB |

---

**Dokumen ini dibuat pada 15 Januari 2026**

**Penulis: AI Assistant (Claude)**
