cara menggunakan website

buat folder virtual environment
python -m venv venv

aktifkan virtual environment
windows: .\venv\Scripts\activate
mac: source venv/bin/activate

install beberapa library yang dibutuhkan untuk virtual environment
- pip install Flask requests pymongo dnspython certifi pyjwt pandas
- pip install flask-cors gunicorn python-dotenv pydantic flask-socketio
- pip install scikit-learn==1.6.0
- pip install torch torchvision torchaudio


selanjutnya run file app.py

jika terjadi error maka masukan pada terminal perintah
python "d:/Prototype Web ICU/Prototype Web ICU/app.py"

jika sudah berhasil dan masuk kehalaman login, masuk menggunakan akun developer
email: admin@icu.dev
password: admin123