# PRISM - Sepsis Treatment ML Platform
# Updated: 2026-02-04 - Railway deployment with HF model loading

FROM node:18-alpine AS css-builder
WORKDIR /build
COPY package*.json ./
COPY tailwind.config.js ./
COPY app/templates ./app/templates
COPY app/static/src ./app/static/src
RUN npm install
RUN npx tailwindcss -i ./app/static/src/css/input.css -o ./app/static/src/css/output.css --minify

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY config.py .
COPY app.py .
COPY wsgi.py .
COPY download_models.py .

COPY --from=css-builder /build/app/static/src/css/output.css ./app/static/src/css/output.css

# Download ALL model files from Hugging Face during build
# Models are now hosted entirely on HF Hub (no Git LFS)
# Includes: .pt, .pth, .npy, and .pkl files
RUN python download_models.py || echo "WARNING: Model download failed, will retry at runtime"

# Test imports at build time
RUN python -c "import flask; import torch; print('Build test: imports OK')"

# Railway provides PORT env var dynamically (defaults to 8080)
EXPOSE 8080

# Start command - Railway will set PORT env var
CMD gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT --timeout 120 --log-level debug --access-logfile - --error-logfile - wsgi:app