from app import create_app
from app.extensions import socketio
import os
import subprocess

# Download models on first startup (Railway runtime)
print("Checking model files...")
try:
    subprocess.run(["python", "download_models.py"], check=True)
    print("Model files ready!")
except Exception as e:
    print(f"Warning: Model download issue: {e}")

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    socketio.run(app, host='0.0.0.0', port=port, debug=debug_mode)
