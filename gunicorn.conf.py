# Gunicorn Configuration for Railway Deployment
# This file takes precedence over Procfile arguments

import os

# Worker Configuration
workers = 1  # Railway free tier: single worker
worker_class = 'eventlet'  # Required for Socket.IO
worker_connections = 1000
threads = 1

# Timeout Configuration (CRITICAL for model loading)
timeout = 300  # 5 minutes - prevent worker timeout during PyTorch model loading
graceful_timeout = 30  # 30 seconds for clean shutdown
keepalive = 2

# Binding
bind = f"0.0.0.0:{os.getenv('PORT', '8080')}"

# Logging
loglevel = 'debug'
accesslog = '-'  # stdout
errorlog = '-'   # stderr
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process Naming
proc_name = 'prism-api'

# Server Mechanics
preload_app = False  # Don't preload - let each worker load on demand
reload = False
daemon = False

# Security
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# Directory
chdir = '/app'

# Environment Check Hook
def on_starting(server):
    """Print environment configuration on Gunicorn startup"""
    print("=" * 50)
    print("PRISM CONFIG - Environment Check")
    print("=" * 50)
    print(f"SECRET_KEY: {'SET' if os.getenv('SECRET_KEY') else 'MISSING'}")
    print(f"TOKEN_KEY: {'SET' if os.getenv('TOKEN_KEY') else 'MISSING'}")
    print(f"MONGODB_URL: {'SET' if os.getenv('MONGODB_URL') else 'MISSING'}")
    print(f"DB_NAME: {'SET' if os.getenv('DBNAME') else 'MISSING'}")
    print(f"PORT: {os.getenv('PORT', '5001')}")
    print("=" * 50)
