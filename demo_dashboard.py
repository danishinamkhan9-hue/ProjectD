"""
ProjectD: Interactive B2B Demo Dashboard
Visual testbench for multimodal taste-reranking and closed-loop feedback modeling.
"""
from __future__ import annotations

import json
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="ProjectD | Real-Time Taste Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E24;
        border: 1px solid #33333F;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        background-color: #2E3856;
        color: #79FFE1;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ ProjectD: Multimodal Taste Reranker & Feedback Loop")
st.caption("B2B SaaS Demonstration Engine | Resolving Underdetermination via DCN-v2 & Bayesian Telemetry")

# =====================================================================
# SIDEBAR: TENANT CONFIG & SESSION STATUS
# =====================================================================
with st.sidebar:
    st.header("Tenant Control")
    user_id = st.text_input("Active User ID", value="fashion_startup_user")
    
    # Health Probe
    try:
        health_resp = requests.get(f"{API_BASE}/healthz", timeout=1.0)
        if health_resp.status_code == 200:
            st.success("Core Engine API: Online (Port 8000)")
        else:
            st.error("Core Engine API: Unhealthy")
    except Exception:
        st.error("Core Engine API: Offline. Run `uvicorn app:app`.")

    st.markdown("---")
    st.subheader("Engine Hyperparameters")
    st.markdown("""
    - **Backbone**: `microsoft/deberta-v3-small` (Fine-Tuned)
    - **Vision Encoder**: `CLIP ViT-B/32`
    - **Manifold Dim**: 32-dim Normalized Latent Space
    - **Feature Crossing**: 3-Layer DCN-v2
    - **Prior**: Dirichlet-Multinomial Grounding
    """)

# Session State Cache
if "last_session_id" not in st.session_state:
    st.session_state.last_session_id = None
if "last_ranked" not in st.session_state:
    st.session_state.last_ranked = []
if "priors_history" not in st.session_state:
    st.session_state.priors_history = None

# =====================================================================
# STAGE 1: MULTIMODAL CANDIDATE INGESTION & RERANKING
# =====================================================================
st.subheader("1. Candidate Evaluation & DCN-v2 Ranking")

col1, col2 = st.columns([1, 1])

candidate_presets = [
    {
        "id": "ITEM_A",
        "text": "Oversized minimalist vintage washed denim jacket",
        "affinity_tag": "streetwear_aesthetic",
        "image_url": "https://images.unsplash.com/photo-1576995853123-5a10305d93c0"
    },
    {
        "id": "ITEM_B",
        "text": "Tailored double-breasted formal charcoal blazer",
        "affinity_tag": "formal_wear",
        "image_url": "https://images.unsplash.com/photo-1594938298603-c8148c4dae35"
    },
    {
        "id": "ITEM_C",
        "text": "Technical waterproof utility cargo jacket with reflective trim",
        "affinity_tag": "techwear_functional",
        "image_url": "https://images.unsplash.com/photo-1548883354-7622d03aca27"
    }
]

with col1:
    st.markdown("#### Input Candidates")
    st.json(candidate_presets, expanded=False)
    
    if st.button("Execute Multimodal Rerank", type="primary", use_container_width=True):
        with st.spinner("Crossing features through DCN-v2..."):
            payload = {"user_id": user_id, "candidates": candidate_presets}
            try:
                res = requests.post(f"{API_BASE}/v1/rerank", json=payload, timeout=8.0)
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.last_session_id = data["session_id"]
                    st.session_state.last_ranked = data["ranked_candidates"]
                    st.success(f"Generated Session: `{data['session_id']}`")
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")

with col2:
    st.markdown("#### Real-Time Ranking Output")
    if st.session_state.last_ranked:
        for idx, item in enumerate(st.session_state.last_ranked, 1):
            with st.container():
                c_img, c_info = st.columns([1, 3])
                with c_img:
                    if item.get("image_url"):
                        st.image(item["image_url"], use_container_width=True)
                with c_info:
                    st.markdown(f"**Rank {idx}: [{item['id']}]** `{item['affinity_tag']}`")
                    st.write(item["text"])
                    st.markdown(f"**Blended Score**: `{item['blended_score']:.4f}` | **DCN-v2 Responsiveness**: `{item['dcn_responsiveness']:.4f}` | **Bayesian Weight**: `{item['prior_weight']:.4f}`")
                st.divider()
    else:
        st.info("Click 'Execute Multimodal Rerank' to compute rankings.")

# =====================================================================
# STAGE 2: CLOSED-LOOP INTERACTION & REAL-TIME PREFERENCE UPDATE
# =====================================================================
st.markdown("---")
st.subheader("2. Closed-Loop Telemetry & Live Preference Update")

if not st.session_state.last_session_id:
    st.warning("Please execute a rerank request in Stage 1 above before simulating feedback.")
else:
    fb_col1, fb_col2 = st.columns([1, 1])

    with fb_col1:
        st.markdown(f"#### Active Session: `{st.session_state.last_session_id}`")
        
        candidate_ids = [c["id"] for c in st.session_state.last_ranked]
        chosen_id = st.selectbox("Selected Candidate Emitted to User", options=candidate_ids)
        chosen_obj = next(c for c in st.session_state.last_ranked if c["id"] == chosen_id)

        feedback_scenario = st.radio(
            "Select Simulated User Response Scenario",
            [
                "Satisfaction (User loves it, copies link)",
                "Explicit Correction (User rejects style, demands another)",
                "Passive Abandonment (Low dwell time, fast bounce)",
                "Happy-but-Wrong Conflict (Polite words, immediate bounce)"
            ]
        )

        # Set scenario telemetry values
        if "Satisfaction" in feedback_scenario:
            agent_text = chosen_obj["text"]
            user_text = "Perfect, this is exactly what I was looking for!"
            dwell = 5600.0
            tokens = 12
            action = "copy"
        elif "Correction" in feedback_scenario:
            agent_text = chosen_obj["text"]
            user_text = "No, that's not what I like at all. Stop showing me formal clothes, I wear casual street clothes."
            dwell = 2200.0
            tokens = 16
            action = "none"
        elif "Abandonment" in feedback_scenario:
            agent_text = chosen_obj["text"]
            user_text = "nevermind, forget it."
            dwell = 800.0
            tokens = 4
            action = "cancel"
        else:
            agent_text = chosen_obj["text"]
            user_text = "Thanks so much, great help!"
            dwell = 650.0
            tokens = 10
            action = "cancel"

        st.text_area("User Utterance", value=user_text, height=70, key="disp_user_text")
        st.caption(f"Simulated Telemetry: Dwell = {dwell}ms | Action = '{action}' | Token Count = {tokens}")

        if st.button("Submit Feedback & Trigger Online Gradient Step", type="primary", use_container_width=True):
            with st.spinner("Processing feedback through DeBERTa & updating DCN-v2..."):
                fb_payload = {
                    "session_id": st.session_state.last_session_id,
                    "selected_candidate_id": chosen_id,
                    "agent_response": agent_text,
                    "user_followup": user_text,
                    "telemetry": {
                        "dwell_time_ms": dwell,
                        "response_token_count": tokens,
                        "user_action": action,
                        "rephrase_similarity": 0.0
                    }
                }
                try:
                    fb_res = requests.post(f"{API_BASE}/v1/feedback", json=fb_payload, timeout=8.0)
                    if fb_res.status_code == 200:
                        st.session_state.fb_result = fb_res.json()
                        st.session_state.priors_history = st.session_state.fb_result["updated_priors"]
                        st.success("Online Model Update Succeeded!")
                    else:
                        st.error(f"Error {fb_res.status_code}: {fb_res.text}")
                except Exception as e:
                    st.error(f"Feedback call failed: {e}")

    with fb_col2:
        st.markdown("#### Engine Adaptation & Bayesian Shift")
        if "fb_result" in st.session_state:
            res = st.session_state.fb_result
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Effective Satisfaction (y)", f"{res['effective_satisfaction']:.4f}")
            m2.metric("Neural Mode", res["dissatisfaction_mode"])
            m3.metric("BCE Loss", f"{res['bce_loss']:.4f}")

            st.info(f"Pipeline Action: **{res['action_decision']}**")

            st.markdown("##### Updated Dirichlet Prior Distribution:")
            st.bar_chart(res["updated_priors"])
            st.caption("The prior distribution has updated in-memory for this user. Re-executing Stage 1 will immediately reflect this behavioral adaptation.")
        else:
            st.info("Submit feedback to view real-time model adaptation and prior shift.")