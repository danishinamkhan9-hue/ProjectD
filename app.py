"""
ProjectD: Production B2B REST API Service
Exposes:
  - POST /v1/rerank    : Multimodal reranking (Text + Vision via CLIP) via DCN-v2
  - POST /v1/feedback  : Evaluates outcome via ProjectD & executes online DCN-v2 update
  - GET  /healthz      : Liveness and model readiness probe
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from multimodal_embedder import MultimodalEmbedder
import numpy as np
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# =====================================================================
# 1. CORE NEURAL & DISAMBIGUATION ENGINE (DCN-v2 + DeBERTa)
# =====================================================================

class DissatisfactionMode(str, Enum):
    NONE = "NONE"
    REPHRASE = "REPHRASE"
    CORRECTION = "CORRECTION"
    ABANDONMENT = "ABANDONMENT"
    DRIFT = "DRIFT"


class CrossNetworkV2(nn.Module):
    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
            for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim, 1))
            for _ in range(num_layers)
        ])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        xl = x0
        x0_col = x0.unsqueeze(-1)
        for l in range(self.num_layers):
            xl_col = xl.unsqueeze(-1)
            mapped = torch.matmul(self.weights[l], xl_col) + self.biases[l]
            xl = (x0_col * mapped).squeeze(-1) + xl
        return xl


class DCNv2TasteReranker(nn.Module):
    def __init__(self, embedding_dim: int = 32, num_cross_layers: int = 3, deep_layers: List[int] = [64, 32]):
        super().__init__()
        input_dim = embedding_dim * 2
        self.cross_net = CrossNetworkV2(input_dim, num_layers=num_cross_layers)

        layers = []
        prev_dim = input_dim
        for hidden_dim in deep_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        self.deep_net = nn.Sequential(*layers)

        combined_dim = input_dim + deep_layers[-1]
        self.head = nn.Linear(combined_dim, 1)

    def forward(self, user_vector: torch.Tensor, item_vector: torch.Tensor) -> torch.Tensor:
        x0 = torch.cat([user_vector, item_vector], dim=-1)
        cross_out = self.cross_net(x0)
        deep_out = self.deep_net(x0)
        stacked = torch.cat([cross_out, deep_out], dim=-1)
        return torch.sigmoid(self.head(stacked))


class DeBERTaFeedbackScorer:
    def __init__(self, model_identifier: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_identifier, num_labels=5)
        self.model.eval()

        self.id2label = {
            0: DissatisfactionMode.NONE,
            1: DissatisfactionMode.REPHRASE,
            2: DissatisfactionMode.CORRECTION,
            3: DissatisfactionMode.ABANDONMENT,
            4: DissatisfactionMode.DRIFT
        }

    def score(self, agent_response: str, user_followup: str) -> Dict[str, Any]:
        if not user_followup or user_followup.strip() == "":
            return {"score": 0.5, "confidence": 0.1, "mode": DissatisfactionMode.NONE.value}

        inputs = self.tokenizer(
            agent_response,
            user_followup,
            return_tensors="pt",
            truncation="longest_first",
            max_length=256,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0].cpu().numpy()

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        pred_idx = int(np.argmax(probs))
        pred_mode = self.id2label.get(pred_idx, DissatisfactionMode.NONE)
        confidence = float(probs[pred_idx])

        score = (
            probs[0] * 1.0 +
            (probs[4] if len(probs) > 4 else 0.0) * 0.3 +
            (probs[1] if len(probs) > 1 else 0.0) * 0.2 +
            (probs[2] if len(probs) > 2 else 0.0) * 0.05 +
            (probs[3] if len(probs) > 3 else 0.0) * 0.0
        )
        return {
            "score": float(np.clip(score, 0.0, 1.0)),
            "confidence": confidence,
            "mode": pred_mode.value
        }


# =====================================================================
# 2. STATE MANAGER
# =====================================================================

class EngineState:
    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim
        self.user_centroids: Dict[str, np.ndarray] = {}
        self.user_priors: Dict[str, Dict[str, float]] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    def get_or_create_user(self, user_id: str) -> Tuple[np.ndarray, Dict[str, float]]:
        if user_id not in self.user_centroids:
            vec = np.random.randn(self.embedding_dim).astype(np.float32)
            self.user_centroids[user_id] = vec / np.linalg.norm(vec)
            self.user_priors[user_id] = {
                "refund_seeker": 2.5,
                "replacement_needed": 2.5,
                "tracking_inquiry": 2.5,
                "frustrated_venting": 2.5
            }
        return self.user_centroids[user_id], self.user_priors[user_id]


# =====================================================================
# 3. PYDANTIC SCHEMAS
# =====================================================================

class CandidateDTO(BaseModel):
    id: str = Field(..., example="C1")
    text: str = Field(..., example="Priority shipping delivery option")
    affinity_tag: str = Field(..., example="replacement_needed")
    image_url: Optional[str] = Field(None, example="https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d")
    embedding: Optional[List[float]] = Field(None, description="32-dim vector. Auto-computed via CLIP if omitted.")


class RerankRequest(BaseModel):
    user_id: str = Field(..., example="startup_user_42")
    candidates: List[CandidateDTO]
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RankedCandidateDTO(BaseModel):
    id: str
    text: str
    affinity_tag: str
    image_url: Optional[str] = None
    blended_score: float
    dcn_responsiveness: float
    prior_weight: float


class RerankResponse(BaseModel):
    session_id: str
    user_id: str
    ranked_candidates: List[RankedCandidateDTO]


class TelemetryDTO(BaseModel):
    dwell_time_ms: float = Field(..., example=5800.0)
    response_token_count: int = Field(..., example=14)
    user_action: str = Field(..., example="copy")
    rephrase_similarity: float = Field(default=0.0)


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., example="sess_12345")
    selected_candidate_id: str = Field(..., example="C1")
    agent_response: str = Field(..., example="Priority shipping delivery option")
    user_followup: str = Field(..., example="Perfect, thank you so much!")
    telemetry: TelemetryDTO


class FeedbackResponse(BaseModel):
    session_id: str
    effective_satisfaction: float
    dissatisfaction_mode: str
    action_decision: str
    bce_loss: float
    updated_priors: Dict[str, float]


# =====================================================================
# 4. FASTAPI LIFESPAN & APPLICATION
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[*] ProjectD Server Starting: Loading Models...")
    current_dir = Path(__file__).resolve().parent
    finetuned_dir = current_dir / "deberta-feedback-finetuned"
    
    model_target = str(finetuned_dir) if (finetuned_dir / "config.json").exists() else "microsoft/deberta-v3-small"

    app.state.embedding_dim = 32
    app.state.reranker = DCNv2TasteReranker(
        embedding_dim=app.state.embedding_dim,
        num_cross_layers=3,
        deep_layers=[64, 32]
    )
    app.state.deberta = DeBERTaFeedbackScorer(model_target)
    app.state.embedder = MultimodalEmbedder(target_dim=app.state.embedding_dim)
    app.state.optimizer = torch.optim.AdamW(app.state.reranker.parameters(), lr=0.02, weight_decay=1e-4)
    app.state.bce_loss = nn.BCELoss()
    app.state.engine = EngineState(embedding_dim=app.state.embedding_dim)
    
    print(f"[✓] ProjectD Multimodal Backend Online. Checkpoint: {model_target}")
    yield
    print("[*] ProjectD Server Shutting Down...")


app = FastAPI(
    title="ProjectD Multimodal Taste Reranker & Feedback Modeling API",
    description="Multimodal (Text + Vision) B2B Engine using DCN-v2 and closed-loop Bayesian telemetry.",
    version="1.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy", "service": "projectd-multimodal-reranker", "version": "1.1.0"}


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank_candidates(payload: RerankRequest):
    if not payload.candidates:
        raise HTTPException(status_code=400, detail="Candidate list cannot be empty")

    user_vec, user_alphas = app.state.engine.get_or_create_user(payload.user_id)
    total_alpha = sum(user_alphas.values())

    item_vectors = []
    candidates_map = {}
    for c in payload.candidates:
        if c.embedding is not None and len(c.embedding) == app.state.embedding_dim:
            v_norm = np.array(c.embedding, dtype=np.float32)
        else:
            v_norm = await asyncio.to_thread(
                app.state.embedder.embed_candidate,
                text=c.text,
                image_url=c.image_url
            )
        item_vectors.append(v_norm)
        candidates_map[c.id] = {"item": c, "vector": v_norm}

    def _compute_dcn_scores():
        app.state.reranker.eval()
        u_tensor = torch.tensor(user_vec, dtype=torch.float32).unsqueeze(0).repeat(len(payload.candidates), 1)
        i_tensor = torch.tensor(np.array(item_vectors), dtype=torch.float32)
        with torch.no_grad():
            return app.state.reranker(u_tensor, i_tensor).squeeze(-1).numpy()

    dcn_preds = await asyncio.to_thread(_compute_dcn_scores)

    ranked_output = []
    for idx, c in enumerate(payload.candidates):
        dcn_score = float(dcn_preds[idx])
        prior_weight = float(user_alphas.get(c.affinity_tag, 1.0) / total_alpha)
        blended = (0.7 * dcn_score) + (0.3 * prior_weight)

        ranked_output.append(
            RankedCandidateDTO(
                id=c.id,
                text=c.text,
                affinity_tag=c.affinity_tag,
                image_url=c.image_url,
                blended_score=round(blended, 4),
                dcn_responsiveness=round(dcn_score, 4),
                prior_weight=round(prior_weight, 4)
            )
        )

    ranked_output.sort(key=lambda x: x.blended_score, reverse=True)

    session_id = f"sess_{uuid.uuid4().hex[:10]}"
    app.state.engine.active_sessions[session_id] = {
        "user_id": payload.user_id,
        "candidates": candidates_map,
        "user_vector": user_vec
    }

    return RerankResponse(
        session_id=session_id,
        user_id=payload.user_id,
        ranked_candidates=ranked_output
    )


@app.post("/v1/feedback", response_model=FeedbackResponse)
async def submit_feedback(payload: FeedbackRequest):
    session = app.state.engine.active_sessions.get(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {payload.session_id} not found or expired")

    selected_data = session["candidates"].get(payload.selected_candidate_id)
    if not selected_data:
        raise HTTPException(status_code=400, detail=f"Candidate ID {payload.selected_candidate_id} was not in this session")

    cand_item: CandidateDTO = selected_data["item"]
    cand_vector: np.ndarray = selected_data["vector"]
    user_id: str = session["user_id"]
    user_vec: np.ndarray = session["user_vector"]

    deberta_res = await asyncio.to_thread(
        app.state.deberta.score,
        payload.agent_response,
        payload.user_followup
    )

    t = payload.telemetry
    expected_dwell = max(500, t.response_token_count * 180)
    dwell_ratio = t.dwell_time_ms / expected_dwell

    telemetry_score = 0.5
    telemetry_conf = 0.6
    if dwell_ratio < 0.25 and t.user_action in ["cancel", "none"]:
        telemetry_score = 0.1
        telemetry_conf = 0.85
    elif t.user_action == "copy" or dwell_ratio >= 0.8:
        telemetry_score = 0.95
        telemetry_conf = 0.90
    elif t.rephrase_similarity > 0.85:
        telemetry_score = 0.2
        telemetry_conf = 0.80

    if deberta_res["score"] > 0.70 and telemetry_score < 0.30:
        reconciled_ground_truth = 0.20
        action = "ESCALATE_FOR_AUDIT"
    else:
        total_conf = deberta_res["confidence"] + telemetry_conf
        reconciled_ground_truth = (
            (deberta_res["score"] * deberta_res["confidence"]) +
            (telemetry_score * telemetry_conf)
        ) / max(total_conf, 1e-6)

        if reconciled_ground_truth >= 0.70:
            action = "ACCEPT_AND_REINFORCE"
        elif reconciled_ground_truth <= 0.35:
            action = "PENALIZE_AND_DOWNWEIGHT"
        else:
            action = "FLAG_FOR_REVIEW"

    user_alphas = app.state.engine.user_priors[user_id]
    tag = cand_item.affinity_tag
    if reconciled_ground_truth > 0.60:
        user_alphas[tag] = user_alphas.get(tag, 1.0) + (2.0 * reconciled_ground_truth)
    else:
        user_alphas[tag] = max(0.5, user_alphas.get(tag, 1.0) - (2.0 * (1.0 - reconciled_ground_truth)))

    def _train_step():
        app.state.reranker.train()
        app.state.optimizer.zero_grad()
        u_t = torch.tensor(user_vec, dtype=torch.float32).unsqueeze(0)
        c_t = torch.tensor(cand_vector, dtype=torch.float32).unsqueeze(0)
        target_t = torch.tensor([[reconciled_ground_truth]], dtype=torch.float32)

        pred = app.state.reranker(u_t, c_t)
        loss = app.state.bce_loss(pred, target_t)
        loss.backward()
        app.state.optimizer.step()
        return float(loss.item())

    loss_val = await asyncio.to_thread(_train_step)

    total_alpha = sum(user_alphas.values())
    priors_normalized = {k: round(v / total_alpha, 4) for k, v in user_alphas.items()}

    return FeedbackResponse(
        session_id=payload.session_id,
        effective_satisfaction=round(float(reconciled_ground_truth), 4),
        dissatisfaction_mode=deberta_res["mode"],
        action_decision=action,
        bce_loss=round(loss_val, 4),
        updated_priors=priors_normalized
    )