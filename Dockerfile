# =============================================================================
# PRISM Web Application - Docker Configuration for Railway
# Multi-stage build: Node (TailwindCSS) → Python (Flask + ML)
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build TailwindCSS
# -----------------------------------------------------------------------------
FROM node:18-alpine AS css-builder

WORKDIR /build

# Copy only what's needed for npm install and CSS build
COPY package*.json ./
COPY tailwind.config.js ./
COPY app/templates ./app/templates
COPY app/static/src ./app/static/src

# Install dependencies and build CSS
RUN npm install
RUN npx tailwindcss -i ./app/static/src/css/input.css -o ./app/static/dist/css/output.css --minify

# -----------------------------------------------------------------------------
# Stage 2: Python Application
# -----------------------------------------------------------------------------
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY config.py .
COPY app.py .
COPY wsgi.py .

# Copy built CSS from Stage 1
COPY --from=css-builder /build/app/static/dist/css/output.css ./app/static/dist/css/output.css

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose port (Railway will override with $PORT)
EXPOSE ${PORT:-5001}

# Health check for Railway
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5001}/sign-in || exit 1

# Run with gunicorn + eventlet for WebSocket support
CMD ["sh", "-c", "gunicorn --worker-class eventlet -w 1 -b 0.0.0.0:${PORT:-5001} --timeout 120 --keep-alive 5 --log-level info wsgi:app"]