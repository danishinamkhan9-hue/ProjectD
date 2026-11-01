#!/usr/bin/env bash
set -e

# Fallback PORT if not provided by cloud platform
PORT=${PORT:-8501}

echo "[*] Launching ProjectD FastAPI Backend on 127.0.0.1:8000..."
uvicorn app:app --host 127.0.0.1 --port 8000 &

# Wait briefly for FastAPI to initialize model weights
sleep 5

echo "[*] Launching ProjectD Demo Dashboard on public port ${PORT}..."
streamlit run demo_dashboard.py \
    --server.port "${PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false