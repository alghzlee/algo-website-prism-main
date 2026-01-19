# 🚀 Railway Deployment Guide - PRISM

Panduan lengkap untuk deploy aplikasi PRISM ke Railway menggunakan Docker.

## Prerequisites

- Akun [Railway](https://railway.app)
- Repository di GitHub
- MongoDB Atlas database (sudah dikonfigurasi)

---

## Quick Deploy

### 1. Connect Repository

1. Buka [Railway Dashboard](https://railway.app/dashboard)
2. Klik **"New Project"** → **"Deploy from GitHub repo"**
3. Pilih repository `algo-website-prism-main`
4. Railway akan otomatis mendeteksi `Dockerfile`

### 2. Configure Environment Variables

Di Railway Dashboard → Project → **Variables**, tambahkan:

| Variable | Value | Cara Generate |
|----------|-------|---------------|
| `SECRET_KEY` | 32+ karakter random | `openssl rand -hex 32` |
| `TOKEN_KEY` | 16+ karakter random | `openssl rand -hex 16` |
| `MONGODB_URL` | MongoDB Atlas URI | Dari MongoDB Atlas Dashboard |
| `DB_NAME` | Nama database | Contoh: `prism_db` |

> **Note**: Railway otomatis set `PORT` environment variable

### 3. Deploy

Railway otomatis build dan deploy saat:
- Push ke branch `main`
- Atau klik **"Deploy"** di dashboard

---

## File Configuration

### `railway.json`
```json
{
  "build": { "builder": "DOCKERFILE" },
  "deploy": {
    "healthcheckPath": "/sign-in",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

### `Dockerfile`
- **Stage 1**: Build TailwindCSS dengan Node.js
- **Stage 2**: Python app dengan Gunicorn + Eventlet
- Health check enabled
- Non-root user (security)

---

## Monitoring

### Health Check
- Endpoint: `/sign-in`
- Interval: 30s
- Timeout: 10s

### Logs
```bash
# Via Railway CLI
railway logs

# Atau di Dashboard → Project → Deployments → View Logs
```

---

## Troubleshooting

### Build Failed
1. Cek logs di Railway Dashboard
2. Pastikan semua dependencies di `requirements.txt` valid
3. Verifikasi `Dockerfile` syntax

### App Crash / Restart Loop
1. Cek environment variables sudah lengkap
2. Pastikan MongoDB Atlas whitelist IP: `0.0.0.0/0` (allow all)
3. Cek koneksi MongoDB di logs

### WebSocket Not Working
- Pastikan menggunakan `--worker-class eventlet`
- Railway support WebSocket secara default

---

## Local Testing (Docker)

```bash
# Build image
docker build -t prism-app .

# Run dengan .env file
docker run -p 5001:5001 --env-file .env prism-app

# Akses di http://localhost:5001
```

---

## Estimated Build Time

| Stage | Time |
|-------|------|
| CSS Build (Node) | ~30s |
| Python Dependencies | ~2-3 min |
| **Total** | ~3-4 min |

---

## Resources

- [Railway Docs](https://docs.railway.app)
- [Railway CLI](https://docs.railway.app/develop/cli)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
