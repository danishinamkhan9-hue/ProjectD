#!/usr/bin/env bash
set -e

PORT=${PORT:-8501}

echo "[*] Launching ProjectD FastAPI Backend on 0.0.0.0:8000..."
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Allow sufficient time for DeBERTa / CLIP model weights to load into RAM
echo "[*] Waiting 15s for model weights to initialize..."
sleep 15

echo "[*] Launching ProjectD Demo Dashboard on public port ${PORT}..."
streamlit run demo_dashboard.py \
    --server.port "${PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false
