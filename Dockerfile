# PRISM - Sepsis Treatment ML Platform
# Optimized for Railway - Image size < 4GB

FROM node:18-alpine AS css-builder
WORKDIR /build
COPY package*.json ./
COPY tailwind.config.js ./
COPY app/templates ./app/templates
COPY app/static/src/css/input.css ./app/static/src/css/input.css
RUN npm ci
RUN npx tailwindcss -i ./app/static/src/css/input.css -o ./app/static/src/css/output.css --minify

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPYCACHEPREFIX=/tmp

WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy application code (model files excluded via .dockerignore)
COPY app/ ./app/
COPY config.py app.py wsgi.py download_models.py ./

# Copy built CSS from previous stage
COPY --from=css-builder /build/app/static/src/css/output.css ./app/static/src/css/output.css

# Clean up unnecessary files to reduce image size
RUN find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /app -type f -name "*.pyc" -delete && \
    rm -rf /tmp/* /var/tmp/*

# Railway provides PORT env var dynamically (defaults to 8080)
EXPOSE 8080

# Start command - Railway will set PORT env var
CMD gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT --timeout 120 --log-level debug --access-logfile - --error-logfile - wsgi:app