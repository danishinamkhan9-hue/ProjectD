#!/usr/bin/env bash
set -e

# Render passes PORT (e.g. 10000) for the public-facing HTTP interface
PUBLIC_PORT=${PORT:-8501}

echo "[*] Launching ProjectD FastAPI Backend on internal port 8000..."
# Override PORT locally for uvicorn so it never attempts to bind to 10000
PORT=8000 uvicorn app:app --host 0.0.0.0 --port 8000 &

# Give Uvicorn 3 seconds to bind its socket and begin async model warm-up
sleep 3

echo "[*] Launching Streamlit Dashboard on public port ${PUBLIC_PORT}..."
streamlit run demo_dashboard.py \
    --server.port "${PUBLIC_PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false
