# DOKUMENTASI OPTIMISASI DOCKER IMAGE
## Project: algo-website-prism-main
### Tanggal: 14 Januari 2026

---

## 1. RINGKASAN EKSEKUTIF

Dokumentasi ini menjelaskan proses optimisasi Docker image untuk project algo-website-prism-main agar dapat di-deploy ke Railway. 

**Masalah:** Ukuran Docker image melebihi 4GB  
**Solusi:** Optimisasi konfigurasi Docker dan dependencies  
**Hasil:** Ukuran image berkurang dari >4GB menjadi **424MB** (pengurangan ~90%)

| Metrik | Sebelum | Sesudah |
|--------|---------|---------|
| Ukuran Image | >4GB | 424MB |
| Pengurangan | - | ~90% |

---

## 2. ANALISIS MASALAH

### 2.1 Penyebab Ukuran Image Besar

Setelah analisis mendalam, ditemukan beberapa penyebab utama:

1. **PyTorch (torch==2.6.0):** Default installation termasuk CUDA/GPU dependencies (~2.5GB+). Railway menggunakan CPU-only, sehingga CUDA tidak diperlukan.

2. **scikit-learn, scipy, pandas:** Library machine learning dengan dependencies numerik (~400MB)

3. **matplotlib:** Library plotting (~100MB)

4. **File model (.pt, .pth, .pkl, .npy):** File model yang ter-copy ke dalam image karena tidak di-exclude dengan benar

### 2.2 Konfigurasi Awal (SEBELUM OPTIMISASI)

#### Dockerfile Original:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "-b", "0.0.0.0:8000", "app.app:app"]
```

#### requirements.txt Original:
```
Flask==3.1.2
Flask_Cors==5.0.0
Flask_SocketIO==5.5.1
joblib==1.4.2
matplotlib==3.10.8
numpy==2.4.1
pandas==2.3.3
pydantic==2.12.5
PyJWT==2.10.1
PyJWT==2.10.1          # MASALAH: duplikat
pymongo==4.16.0
python-dotenv==1.2.1
scikit_learn==1.8.0
scipy==1.17.0
torch==2.6.0           # MASALAH: penyebab utama ukuran besar
Werkzeug==3.1.5
gunicorn==23.0.0
```

#### .dockerignore Original:
```
.git
.gitignore
.env
__pycache__/
*.pyc

node_modules/
app/static/src/
package.json
package-lock.json
tailwind.config.js

.venv/
venv/

*.ipynb
*.csv
*.zip
*.pt
*.pkl
```
**MASALAH:** Tidak include `*.npy`, `*.pth`, dan pattern lainnya

---

## 3. SOLUSI YANG DIIMPLEMENTASIKAN

### 3.1 Optimisasi Dockerfile

**Perubahan yang dilakukan:**
- Multi-stage build untuk memisahkan build dan production stage
- Install PyTorch dari CPU-only index (menghemat ~1.7GB)
- Menambahkan eventlet worker untuk Flask-SocketIO support
- Menggunakan virtual environment untuk isolation

#### Dockerfile BARU:
```dockerfile
# ===== BUILD STAGE =====
FROM python:3.11-slim AS builder
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch CPU-only FIRST (dari CPU-only index)
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ===== PRODUCTION STAGE =====
FROM python:3.11-slim AS production
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Expose port for Railway
EXPOSE 8000

# Run with gunicorn
CMD ["gunicorn", "-k", "eventlet", "-w", "1", "-b", "0.0.0.0:8000", "wsgi:app"]
```

---

### 3.2 Optimisasi requirements.txt

**Perubahan yang dilakukan:**
- Menghapus `torch==2.6.0` (di-install terpisah di Dockerfile dengan CPU-only flag)
- Menghapus duplikat `PyJWT==2.10.1`
- Menghapus `Werkzeug==3.1.5` (biarkan Flask memilih versi yang kompatibel)
- Menambahkan `eventlet==0.38.2` untuk Flask-SocketIO async support

#### requirements.txt BARU:
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
```

---

### 3.3 Optimisasi .dockerignore

**Perubahan yang dilakukan:**
- Menambahkan `*.npy` untuk file numpy array
- Menambahkan `*.pth` untuk file PyTorch model
- Menambahkan `*.h5`, `*.onnx`, `*.safetensors` untuk future-proofing
- Menambahkan pattern untuk file IDE dan temporary

#### .dockerignore BARU:
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
*.pyd
.Python
.venv/
venv/
env/
*.egg-info/
*.egg

# Node.js (Tailwind build tools)
node_modules/
app/static/src/
package.json
package-lock.json
tailwind.config.js

# Jupyter & Data files
*.ipynb
*.csv
*.zip

# Model files - SANGAT PENTING!
*.pt
*.pth
*.pkl
*.npy
*.h5
*.onnx
*.safetensors

# IDE & OS
.DS_Store
.idea/
.vscode/
*.swp
*.swo

# Temp files
tempCodeRunnerFile.py
*.tmp
*.log

# Documentation
*.md
!README.md

# Docker/CI
Dockerfile*
docker-compose*
.dockerignore
```

---

### 3.4 Update wsgi.py

**Perubahan:** Menambahkan `eventlet.monkey_patch()` di awal file agar patching terjadi sebelum import library lainnya.

#### wsgi.py BARU:
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

## 4. VERIFIKASI

### 4.1 Build Image
```bash
docker build --platform linux/amd64 -t algo-prism-test .
```

### 4.2 Cek Ukuran Image
```bash
docker images algo-prism-test

# Output:
REPOSITORY        TAG      SIZE
algo-prism-test   latest   424MB
```

### 4.3 Test Run Container
```bash
docker run --platform linux/amd64 -p 8000:8000 --env-file .env algo-prism-test

# Output yang diharapkan:
[INFO] Starting gunicorn 23.0.0
[INFO] Listening at: http://0.0.0.0:8000 (1)
[INFO] Using worker: eventlet
[INFO] Booting worker with pid: 7
```

---

## 5. DEPLOY KE RAILWAY

**Langkah-langkah:**

1. Commit semua perubahan ke repository Git
2. Push ke GitHub/GitLab
3. Connect repository ke Railway
4. Railway akan otomatis detect Dockerfile dan build image
5. Set environment variables di Railway dashboard (dari file .env)
6. Deploy!

**Catatan Penting:**
- Railway menggunakan Linux AMD64
- Jika develop di Mac M1/M2, gunakan `--platform linux/amd64` saat testing lokal
- Environment variables dari .env harus di-set manual di Railway dashboard
- Pastikan MongoDB connection string sudah benar

---

## 6. KESIMPULAN

Optimisasi Docker image berhasil mengurangi ukuran dari lebih dari **4GB** menjadi **424MB** (pengurangan ~90%). 

Perubahan utama yang memberikan dampak terbesar adalah menginstall PyTorch versi CPU-only yang menghemat sekitar 1.7GB.

Dengan ukuran image yang sudah optimal, project siap untuk di-deploy ke Railway.

---

## 7. LAMPIRAN: RINGKASAN FILE YANG DIMODIFIKASI

| No | File | Deskripsi Perubahan |
|----|------|---------------------|
| 1 | Dockerfile | Multi-stage build, PyTorch CPU-only, eventlet worker |
| 2 | requirements.txt | Hapus torch, hapus Werkzeug, tambah eventlet, hapus duplikat |
| 3 | .dockerignore | Tambah pattern untuk model files (*.npy, *.pth, dll) |
| 4 | wsgi.py | Tambah eventlet.monkey_patch() di awal file |

---

**Dokumen ini dibuat pada 14 Januari 2026**
