
# PRISM - Yarsi
Dokumen ini menjelaskan langkah-langkah untuk menjalankan website PRISM menggunakan Python dan virtual environment.

1. Membuat Virtual Environment

Buat virtual environment di direktori proyek:

```python -m venv venv```

2. Mengaktifkan Virtual Environment

Aktifkan virtual environment sesuai sistem operasi:

Windows

```.\venv\Scripts\activate```


macOS / Linux

```source venv/bin/activate```

3. Menginstal Dependency

Instal seluruh library yang dibutuhkan:

```pip install Flask requests pymongo dnspython certifi pyjwt pandas```
```pip install flask-cors gunicorn python-dotenv pydantic flask-socketio```
```pip install scikit-learn==1.6.0```
```pip install torch torchvision torchaudio```

4. Menjalankan Aplikasi

Jalankan aplikasi dengan perintah:

```python app.py```

5. Jika Terjadi Error Path (Windows)

Apabila muncul error terkait path atau direktori kerja, jalankan aplikasi menggunakan path lengkap:

```python "d:/Prototype Web ICU/Prototype Web ICU/app.py"```

Catatan

Pastikan virtual environment aktif sebelum menjalankan aplikasi.

Gunakan Python versi yang kompatibel dengan PyTorch dan scikit-learn.

Port default aplikasi adalah 5000, pastikan tidak digunakan oleh aplikasi lain.