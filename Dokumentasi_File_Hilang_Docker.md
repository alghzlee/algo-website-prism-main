# DOKUMENTASI ANALISIS FILE YANG HILANG DI DOCKER IMAGE
## Project: algo-website-prism-main
### Tanggal: 14 Januari 2026

---

## 1. RINGKASAN MASALAH

Setelah optimisasi Docker image, beberapa file tidak ter-copy ke dalam container karena di-exclude oleh `.dockerignore`. Hal ini menyebabkan:
- **Gambar tidak muncul** di website
- **Model ML tidak bisa di-load** (berpotensi error saat prediction)
- **Data CSV tidak tersedia** (berpotensi error saat websocket)

---

## 2. HASIL ANALISIS

### 2.1 File Gambar yang Hilang

**Lokasi:** `app/static/src/images/`

**Alasan Hilang:** `.dockerignore` mengandung pattern `app/static/src/` yang meng-exclude seluruh folder tersebut.

**File yang Terpengaruh:**
| File | Lokasi |
|------|--------|
| bed-hospital.jpg | app/static/src/images/assets/ |
| x-ray.png | app/static/src/images/assets/ |
| admin@icu.dev.jpeg | app/static/src/images/profiles/ |
| admin@icu.dev.jpg | app/static/src/images/profiles/ |
| admin@icu.dev.png | app/static/src/images/profiles/ |
| profile.jpeg | app/static/src/images/profiles/ |

**File yang Menggunakan Gambar Ini:**
- `app/templates/home/bed-selection.html` → menggunakan `bed-hospital.jpg`
- `app/templates/component/navbar_dua.html` → menggunakan profile images

---

### 2.2 File Model ML yang Hilang

**Lokasi:** `app/data/`

**Alasan Hilang:** `.dockerignore` mengandung pattern `*.pkl`, `*.pt`, `*.pth`, `*.npy`, `*.csv`

**File yang Terpengaruh:**
| File | Ukuran | Digunakan Oleh |
|------|--------|----------------|
| best_agent_ensemble.pt | 8.4 MB | app/routes/predict.py |
| best_ae_mimic.pth | 67 KB | app/routes/predict.py |
| action_norm_stats.pkl | 110 B | app/routes/predict.py |
| state_norm_stats.pkl | 118 B | app/data/SAC_deepQnet.py |
| state_kmeans_model.pkl | 282 KB | (ML inference) |
| patients_kmeans_model.pkl | 67 KB | (ML inference) |
| pca_model.pkl | 161 KB | (ML inference) |
| transition_prob.pkl | 8.1 MB | (ML inference) |
| phys_actionsb.npy | 379 KB | (ML inference) |
| phys_bQ.npy | 189 KB | (ML inference) |

---

### 2.3 File CSV Data yang Hilang

**Lokasi:** `app/data/`

**File yang Terpengaruh:**
| File | Ukuran | Digunakan Oleh |
|------|--------|----------------|
| df_with_readable_charttime.csv | 60 KB | app/websockets/treatment_recommendation_socket.py |
| selected_data.csv | 5.5 KB | app/websockets/patient_monitoring_socket.py |
| sofa_indicators.csv | 4.3 KB | app/websockets/patient_monitoring_socket.py |

---

## 3. OPSI SOLUSI

### OPSI A: Pindahkan File Statis ke MongoDB Atlas (DIREKOMENDASIKAN untuk Production)

**Kelebihan:**
- Files tidak perlu di-bundle dalam Docker image
- Mudah di-update tanpa rebuild image
- Scalable untuk multiple instances
- **MongoDB Atlas** menyediakan cloud-hosted database yang reliable dan mudah diakses dari Railway

**Catatan Penting untuk MongoDB Atlas:**
- Pastikan connection string menggunakan format: `mongodb+srv://<username>:<password>@<cluster>.mongodb.net/<dbname>`
- Pastikan IP Address Railway sudah di-whitelist di MongoDB Atlas Network Access (atau gunakan `0.0.0.0/0` untuk allow all)
- GridFS tersedia di semua tier MongoDB Atlas termasuk FREE tier (M0)

**Langkah Implementasi:**

#### A.1 Setup MongoDB Atlas GridFS untuk File Storage

GridFS adalah spesifikasi MongoDB untuk menyimpan file besar (>16MB). GridFS bekerja dengan baik di MongoDB Atlas.

```python
# app/services/file_storage.py
from pymongo import MongoClient
from gridfs import GridFS
from flask import current_app
import os
import certifi

def get_gridfs():
    """Get GridFS instance from MongoDB Atlas"""
    # Gunakan certifi untuk SSL certificate (required untuk MongoDB Atlas)
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

**Catatan:** Pastikan `certifi` sudah terinstall di requirements.txt untuk koneksi SSL ke MongoDB Atlas.

#### A.2 Update requirements.txt

Tambahkan `certifi` untuk SSL connection ke MongoDB Atlas:

```
certifi>=2023.0.0
```

#### A.3 Script Upload File ke MongoDB Atlas

Jalankan script ini SEKALI dari environment lokal untuk upload semua file ke MongoDB Atlas:

```python
# scripts/upload_assets_to_mongodb.py
import os
from pymongo import MongoClient
from gridfs import GridFS
from dotenv import load_dotenv
import certifi

load_dotenv()

# Connect to MongoDB Atlas dengan SSL certificate
# Format: mongodb+srv://<username>:<password>@<cluster>.mongodb.net/<dbname>
client = MongoClient(
    os.getenv('MONGODB_URL'),
    tlsCAFile=certifi.where()
)
db = client[os.getenv('DBNAME')]
fs = GridFS(db)

print(f"Connected to MongoDB Atlas: {os.getenv('DBNAME')}")

# Files to upload
files_to_upload = [
    # Images
    'app/static/src/images/assets/bed-hospital.jpg',
    'app/static/src/images/assets/x-ray.png',
    'app/static/src/images/profiles/profile.jpeg',
    'app/static/src/images/profiles/admin@icu.dev.jpeg',
    'app/static/src/images/profiles/admin@icu.dev.jpg',
    'app/static/src/images/profiles/admin@icu.dev.png',
    
    # Models
    'app/data/best_agent_ensemble.pt',
    'app/data/best_ae_mimic.pth',
    'app/data/action_norm_stats.pkl',
    'app/data/state_norm_stats.pkl',
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
]

uploaded_count = 0
for file_path in files_to_upload:
    if os.path.exists(file_path):
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        # Check if already exists
        existing = fs.find_one({'filename': filename})
        if existing:
            fs.delete(existing._id)
            print(f"Replaced: {filename} ({file_size / 1024:.1f} KB)")
        else:
            print(f"Uploading: {filename} ({file_size / 1024:.1f} KB)")
        
        with open(file_path, 'rb') as f:
            fs.put(f, filename=filename)
        uploaded_count += 1
    else:
        print(f"NOT FOUND: {file_path}")

print(f"\nDone! {uploaded_count} files uploaded to MongoDB Atlas.")
print("You can verify in MongoDB Atlas Dashboard > Collections > fs.files")
```

#### A.4 Route untuk Serve Gambar dari MongoDB Atlas

```python
# app/routes/assets.py
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
    
    # Determine content type
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

@assets_.route('/assets/models/<path:filename>')
def serve_model(filename):
    """Serve model files from MongoDB Atlas GridFS"""
    fs = get_gridfs()
    grid_file = fs.find_one({'filename': filename})
    
    if not grid_file:
        abort(404)
    
    return Response(
        grid_file.read(),
        content_type='application/octet-stream'
    )
```

#### A.5 Update Template HTML

Ubah referensi gambar dari static files ke route MongoDB Atlas:

**SEBELUM:**
```html
<img src="{{url_for('static', filename='src/images/assets/bed-hospital.jpg')}}" />
```

**SESUDAH:**
```html
<img src="{{url_for('assets.serve_image', filename='bed-hospital.jpg')}}" />
```

---

### OPSI B: Update .dockerignore (Solusi Cepat)

Jika ingin tetap menggunakan file lokal, update `.dockerignore` untuk TIDAK meng-exclude file yang diperlukan.

**Ubah baris ini di .dockerignore:**

```diff
# SEBELUM
app/static/src/
*.pkl
*.pt
*.pth
*.npy
*.csv

# SESUDAH - Exclude hanya file yang benar-benar tidak perlu
# node_modules dan source CSS saja yang di-exclude
node_modules/
app/static/src/css/
app/static/src/js/

# JANGAN exclude images
# app/static/src/images/ <- HAPUS BARIS INI

# Model files tetap exclude jika pakai MongoDB Atlas
# *.pkl
# *.pt
# etc.
```

**PERINGATAN:** Ini akan menambah ukuran Docker image (~17MB untuk folder app/data/).

---

### OPSI C: Gunakan Cloud Storage (AWS S3/Cloudinary)

Untuk production skala besar, pertimbangkan:
- **AWS S3** untuk file storage
- **Cloudinary** untuk image optimization & CDN
- **Google Cloud Storage** sebagai alternatif

---

## 4. KONFIGURASI MONGODB ATLAS

### 4.1 Whitelist IP Address

Untuk Railway deployment, Anda perlu whitelist IP di MongoDB Atlas:

1. Buka MongoDB Atlas Dashboard
2. Pergi ke **Network Access** > **IP Access List**
3. Klik **Add IP Address**
4. Pilih **Allow Access from Anywhere** (0.0.0.0/0) untuk development
5. Atau tambahkan IP Railway jika diketahui

### 4.2 Connection String Format

Pastikan environment variable `MONGODB_URL` menggunakan format MongoDB Atlas:

```
mongodb+srv://<username>:<password>@<cluster-name>.mongodb.net/<database>?retryWrites=true&w=majority
```

**Contoh:**
```
mongodb+srv://myuser:mypassword@cluster0.abc123.mongodb.net/algo_prism?retryWrites=true&w=majority
```

### 4.3 Verifikasi Upload di MongoDB Atlas

Setelah menjalankan script upload, verifikasi:

1. Buka MongoDB Atlas Dashboard
2. Klik **Browse Collections**
3. Cari collection `fs.files` dan `fs.chunks`
4. Pastikan semua file sudah ter-upload

---

## 5. REKOMENDASI IMPLEMENTASI

Untuk Railway + MongoDB Atlas deployment, saya merekomendasikan:

| Tipe File | Solusi | Alasan |
|-----------|--------|--------|
| Gambar (images) | MongoDB Atlas GridFS | Terintegrasi dengan database yang sudah ada |
| Model ML (.pt, .pkl) | MongoDB Atlas GridFS | Model jarang berubah, mudah di-manage |
| Data CSV | MongoDB Atlas Collection atau GridFS | Bisa pilih sesuai kebutuhan query |

---

## 6. LANGKAH IMPLEMENTASI YANG DISARANKAN

### Langkah 1: Tambah certifi ke requirements.txt
```
certifi>=2023.0.0
```

### Langkah 2: Buat File Storage Service
Buat file `app/services/file_storage.py` dengan code dari bagian A.1

### Langkah 3: Buat Assets Route  
Buat file `app/routes/assets.py` dengan code dari bagian A.4

### Langkah 4: Register Blueprint
Update `app/__init__.py`:
```python
from .routes.assets import assets_
app.register_blueprint(assets_)
```

### Langkah 5: Whitelist IP di MongoDB Atlas
Buka MongoDB Atlas > Network Access > Allow from Anywhere (0.0.0.0/0)

### Langkah 6: Upload Files ke MongoDB Atlas
Jalankan script dari bagian A.3 di environment lokal:
```bash
source venv/bin/activate
python scripts/upload_assets_to_mongodb.py
```

### Langkah 7: Update Template References
Ganti semua `url_for('static', filename='src/images/...')` menjadi `url_for('assets.serve_image', filename='...')`

### Langkah 8: Rebuild & Deploy
```bash
docker build --platform linux/amd64 -t algo-prism-test .
# Push to Railway
```

---

## 7. KESIMPULAN

File yang hilang di Docker image disebabkan oleh konfigurasi `.dockerignore` yang terlalu agresif. Solusi terbaik untuk production dengan **MongoDB Atlas** adalah memindahkan file statis ke GridFS, yang akan:

1. ✅ Mengurangi ukuran Docker image
2. ✅ Memudahkan update file tanpa rebuild
3. ✅ Scalable untuk multiple instances
4. ✅ Centralized file management di MongoDB Atlas
5. ✅ Terintegrasi dengan database yang sudah ada

---

## 8. LAMPIRAN: DAFTAR PERUBAHAN FILE

| No | File | Aksi |
|----|------|------|
| 1 | requirements.txt | UPDATE (tambah certifi) |
| 2 | app/services/file_storage.py | BUAT BARU |
| 3 | app/routes/assets.py | BUAT BARU |
| 4 | app/__init__.py | UPDATE (register blueprint) |
| 5 | scripts/upload_assets_to_mongodb.py | BUAT BARU |
| 6 | app/templates/home/bed-selection.html | UPDATE (image references) |
| 7 | app/templates/component/navbar_dua.html | UPDATE (image references) |

---

**Dokumen ini dibuat pada 14 Januari 2026**
