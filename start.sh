#!/bin/bash
set -eu

export PYTHONUNBUFFERED=true

VIRTUALENV=".data/venv"

# Memastikan virtual environment ada
if [ ! -d $VIRTUALENV ]; then
  python3 -m venv $VIRTUALENV
fi

# Install pip jika belum ada
if [ ! -f $VIRTUALENV/bin/pip ]; then
  curl --silent --show-error --retry 5 https://bootstrap.pypa.io/get-pip.py | $VIRTUALENV/bin/python
fi

# Install dependencies
$VIRTUALENV/bin/pip install --no-cache-dir -r requirements.txt

# Build Tailwind CSS jika ada
if [ -f tailwind.config.js ]; then
  echo "Building Tailwind CSS..."
  npx tailwindcss -i ./app/static/src/css/input.css -o ./app/static/dist/css/output.css --watch &
fi

# Jalankan aplikasi dengan python di virtual environment
$VIRTUALENV/bin/python app.py
