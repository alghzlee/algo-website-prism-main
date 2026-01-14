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