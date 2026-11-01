"""
ProjectD: Closed-Loop Taste Reranker & Feedback Modeling Architecture
Integrates:
1. DCN-v2 (Deep & Cross Network v2) Taste Reranker
2. Candidate Selection & Semantic User Centroid Space
3. ProjectD Neural Feedback & Telemetry Scorer (DeBERTa + Bayesian Grounding)
4. Online Gradient Update / Preference Reward Optimization Loop
"""

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =====================================================================
# 1. CORE SCHEMAS & DATA CONTRACTS
# =====================================================================

class DissatisfactionMode(str, Enum):
    NONE = "NONE"
    REPHRASE = "REPHRASE"
    CORRECTION = "CORRECTION"
    ABANDONMENT = "ABANDONMENT"
    DRIFT = "DRIFT"


@dataclass
class TelemetryPayload:
    dwell_time_ms: float
    response_token_count: int
    user_action: str  # "copy", "click_external", "cancel", "none"
    rephrase_similarity: float = 0.0


@dataclass
class CandidateItem:
    id: str
    text: str
    affinity_tag: str
    semantic_embedding: Optional[np.ndarray] = None


@dataclass
class ScorerResult:
    scorer_name: str
    satisfaction_score: float  # [0.0, 1.0]
    confidence: float          # [0.0, 1.0]
    detected_mode: DissatisfactionMode
    metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# 2. DCN-v2 (DEEP & CROSS NETWORK V2) TASTE RERANKER
# =====================================================================

class CrossNetworkV2(nn.Module):
    """
    Explicit Multi-Order Feature Crossing (Matrix Multiplication Mode).
    Formula: x_{l+1} = x_0 ⊙ (W_l * x_l + b_l) + x_l
    Discovers high-order feature attention correlations across user & item dimensions.
    """
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
        # x0 shape: (batch_size, input_dim)
        xl = x0
        x0_col = x0.unsqueeze(-1)  # (batch_size, input_dim, 1)

        for l in range(self.num_layers):
            # xl_col: (batch_size, input_dim, 1)
            xl_col = xl.unsqueeze(-1)
            # W * xl + b
            mapped = torch.matmul(self.weights[l], xl_col) + self.biases[l]
            # Element-wise product with original x0, plus residual connection
            xl = (x0_col * mapped).squeeze(-1) + xl

        return xl


class DCNv2TasteReranker(nn.Module):
    """
    DCN-v2: Parallel Deep & Cross Network.
    Uses LayerNorm instead of BatchNorm to support online streaming updates (batch_size=1).
    """
    def __init__(self, embedding_dim: int = 64, num_cross_layers: int = 3, deep_layers: List[int] = [128, 64]):
        super().__init__()
        input_dim = embedding_dim * 2  # user_vector concatenated with item_vector

        # Cross Path
        self.cross_net = CrossNetworkV2(input_dim, num_layers=num_cross_layers)

        # Deep Path (MLP) with LayerNorm
        layers = []
        prev_dim = input_dim
        for hidden_dim in deep_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # Replaced BatchNorm1d with LayerNorm
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        self.deep_net = nn.Sequential(*layers)

        # Stacking Combination Layer -> Responsiveness Logit
        combined_dim = input_dim + deep_layers[-1]
        self.head = nn.Linear(combined_dim, 1)

    def forward(self, user_vector: torch.Tensor, item_vector: torch.Tensor) -> torch.Tensor:
        x0 = torch.cat([user_vector, item_vector], dim=-1)

        cross_out = self.cross_net(x0)
        deep_out = self.deep_net(x0)

        stacked = torch.cat([cross_out, deep_out], dim=-1)
        logits = self.head(stacked)
        return torch.sigmoid(logits)   # Predicted responsiveness P(Engagement / Success)


# =====================================================================
# 3. PROJECTD FEEDBACK & TELEMETRY SCORING PIPELINE
# =====================================================================

class DeBERTaFeedbackScorer:
    """
    Evaluates response-followup pairs using the fine-tuned DeBERTa model.
    """
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

    def score(self, agent_response: str, user_followup: str) -> ScorerResult:
        if not user_followup or user_followup.strip() == "":
            return ScorerResult("deberta_scorer", 0.5, 0.1, DissatisfactionMode.NONE)

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

        # Continuous satisfaction calculation
        score = (
            probs[0] * 1.0 +
            (probs[4] if len(probs) > 4 else 0.0) * 0.3 +
            (probs[1] if len(probs) > 1 else 0.0) * 0.2 +
            (probs[2] if len(probs) > 2 else 0.0) * 0.05 +
            (probs[3] if len(probs) > 3 else 0.0) * 0.0
        )

        return ScorerResult(
            scorer_name="deberta_scorer",
            satisfaction_score=float(np.clip(score, 0.0, 1.0)),
            confidence=confidence,
            detected_mode=pred_mode,
            metadata={"probabilities": {self.id2label[i].value: round(float(probs[i]), 4) for i in range(min(5, len(probs)))}}
        )


class BehavioralTelemetryScorer:
    """
    Evaluates physical behavioral friction: dwell ratio, copy events, rapid abandons.
    """
    def score(self, telemetry: TelemetryPayload) -> ScorerResult:
        expected_dwell_ms = max(500, telemetry.response_token_count * 180)
        dwell_ratio = telemetry.dwell_time_ms / expected_dwell_ms

        score = 0.5
        mode = DissatisfactionMode.NONE
        confidence = 0.6

        if dwell_ratio < 0.25 and telemetry.user_action in ["cancel", "none"]:
            score = 0.1
            mode = DissatisfactionMode.ABANDONMENT
            confidence = 0.85
        elif telemetry.user_action == "copy" or dwell_ratio >= 0.8:
            score = 0.95
            mode = DissatisfactionMode.NONE
            confidence = 0.9
        elif telemetry.rephrase_similarity > 0.85:
            score = 0.2
            mode = DissatisfactionMode.REPHRASE
            confidence = 0.8

        return ScorerResult(
            scorer_name="behavioral_telemetry",
            satisfaction_score=score,
            confidence=confidence,
            detected_mode=mode,
            metadata={"dwell_ratio": round(dwell_ratio, 2), "action": telemetry.user_action}
        )


class BayesianHypothesisManager:
    """
    Maintains Dirichlet-Multinomial prior over user taste/intents.
    """
    def __init__(self, initial_priors: Dict[str, float]):
        self.alpha_counts = {k: v * 10.0 for k, v in initial_priors.items()}

    def get_prior_bias(self, affinity_tag: str) -> float:
        total = sum(self.alpha_counts.values())
        return self.alpha_counts.get(affinity_tag, 1.0) / total

    def update_priors(self, selected_tag: str, satisfaction_score: float):
        learning_rate = 2.0
        if satisfaction_score > 0.6:
            self.alpha_counts[selected_tag] = self.alpha_counts.get(selected_tag, 1.0) + (learning_rate * satisfaction_score)
        else:
            penalty = learning_rate * (1.0 - satisfaction_score)
            self.alpha_counts[selected_tag] = max(0.5, self.alpha_counts.get(selected_tag, 1.0) - penalty)


# =====================================================================
# 4. ORCHESTRATOR: RE-RANKER TO FEEDBACK LOOP WITH ONLINE LEARNING
# =====================================================================

class ClosedLoopTasteSystem:
    def __init__(
        self,
        reranker: DCNv2TasteReranker,
        deberta_scorer: DeBERTaFeedbackScorer,
        telemetry_scorer: BehavioralTelemetryScorer,
        bayes_manager: BayesianHypothesisManager,
        embedding_dim: int = 64,
        learning_rate: float = 0.01
    ):
        self.reranker = reranker
        self.deberta_scorer = deberta_scorer
        self.telemetry_scorer = telemetry_scorer
        self.bayes_manager = bayes_manager
        self.embedding_dim = embedding_dim

        # Online optimizer for the DCN-v2 network
        self.optimizer = torch.optim.AdamW(self.reranker.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = nn.BCELoss()

    def rank_candidates(self, user_taste_vector: np.ndarray, candidates: List[CandidateItem]) -> List[Tuple[CandidateItem, float]]:
        """
        Takes user taste vector and candidate items, runs DCN-v2 forward pass,
        blends neural responsiveness with Bayesian intent priors, and returns ranked pairs.
        """
        self.reranker.eval()

        u_tensor = torch.tensor(user_taste_vector, dtype=torch.float32).unsqueeze(0).repeat(len(candidates), 1)
        item_tensors = torch.tensor(
            np.array([c.semantic_embedding for c in candidates]),
            dtype=torch.float32
        )

        with torch.no_grad():
            responsiveness_scores = self.reranker(u_tensor, item_tensors).squeeze(-1).numpy()

        scored_candidates = []
        for i, candidate in enumerate(candidates):
            dcn_score = float(responsiveness_scores[i])
            prior_bias = self.bayes_manager.get_prior_bias(candidate.affinity_tag)
            
            # Hybrid rank score: 70% DCN-v2 latent attention + 30% Bayesian historical prior
            blended_score = (0.7 * dcn_score) + (0.3 * prior_bias)
            scored_candidates.append((candidate, blended_score))

        # Sort descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates

    def process_turn_and_update(
        self,
        user_taste_vector: np.ndarray,
        chosen_candidate: CandidateItem,
        agent_response: str,
        user_followup: str,
        telemetry: TelemetryPayload
    ) -> Dict[str, Any]:
        """
        Evaluates emitted response via ProjectD, derives ground-truth target y,
        and executes an online backpropagation step on DCN-v2.
        """
        # 1. ProjectD Scorer Evaluations
        deberta_res = self.deberta_scorer.score(agent_response, user_followup)
        telemetry_res = self.telemetry_scorer.score(telemetry)

        # 2. Conflict Reconciliation (Happy-but-wrong detection)
        conflict_detected = False
        if deberta_res.satisfaction_score > 0.7 and telemetry_res.satisfaction_score < 0.3:
            conflict_detected = True
            reconciled_ground_truth = 0.20
            action = "ESCALATE_FOR_AUDIT"
        else:
            total_conf = deberta_res.confidence + telemetry_res.confidence
            reconciled_ground_truth = (
                (deberta_res.satisfaction_score * deberta_res.confidence) +
                (telemetry_res.satisfaction_score * telemetry_res.confidence)
            ) / max(total_conf, 1e-6)

            if reconciled_ground_truth >= 0.7:
                action = "ACCEPT_AND_REINFORCE"
            elif reconciled_ground_truth <= 0.35:
                action = "PENALIZE_AND_DOWNWEIGHT"
            else:
                action = "FLAG_FOR_REVIEW"

        # 3. Update Bayesian Prior Distribution
        self.bayes_manager.update_priors(chosen_candidate.affinity_tag, reconciled_ground_truth)

        # 4. ONLINE LEARNING STEP ON DCN-v2
        # Train DCN-v2 to predict this reconciled satisfaction score for this (user, candidate) pair
        self.reranker.train()
        self.optimizer.zero_grad()

        u_tensor = torch.tensor(user_taste_vector, dtype=torch.float32).unsqueeze(0)
        c_tensor = torch.tensor(chosen_candidate.semantic_embedding, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor([[reconciled_ground_truth]], dtype=torch.float32)

        predicted_responsiveness = self.reranker(u_tensor, c_tensor)
        loss = self.criterion(predicted_responsiveness, target_tensor)
        loss.backward()
        self.optimizer.step()

        return {
            "predicted_dcn_score_before": round(float(predicted_responsiveness.item()), 4),
            "reconciled_ground_truth": round(float(reconciled_ground_truth), 4),
            "loss": round(float(loss.item()), 4),
            "action": action,
            "conflict_detected": conflict_detected,
            "deberta_mode": deberta_res.detected_mode.value,
            "deberta_score": round(deberta_res.satisfaction_score, 4),
            "telemetry_score": round(telemetry_res.satisfaction_score, 4),
            "updated_priors": {
                k: round(v / sum(self.bayes_manager.alpha_counts.values()), 4)
                for k, v in self.bayes_manager.alpha_counts.items()
            }
        }


# =====================================================================
# 5. EXECUTION DEMONSTRATION & END-TO-END VERIFICATION
# =====================================================================

if __name__ == "__main__":
    print("===================================================================")
    print(" PROJECTD: CLOSED-LOOP DCN-v2 TASTE RERANKER + FEEDBACK PIPELINE  ")
    print("===================================================================")

    current_dir = Path(__file__).resolve().parent
    finetuned_dir = current_dir / "deberta-feedback-finetuned"
    
    # Model fallback check
    if finetuned_dir.exists() and (finetuned_dir / "config.json").exists():
        model_target = str(finetuned_dir)
        print(f"[*] Found fine-tuned DeBERTa model: {model_target}")
    else:
        model_target = "microsoft/deberta-v3-small"
        print(f"[*] Defaulting to base model: {model_target}")

    EMBEDDING_DIM = 32

    # Instantiate Sub-systems
    print("[*] Initializing DCN-v2 Taste Reranker...")
    reranker = DCNv2TasteReranker(embedding_dim=EMBEDDING_DIM, num_cross_layers=3, deep_layers=[64, 32])

    print("[*] Initializing ProjectD Feedback Scorers...")
    deberta = DeBERTaFeedbackScorer(model_target)
    telemetry_scorer = BehavioralTelemetryScorer()

    initial_priors = {
        "refund_seeker": 0.25,
        "replacement_needed": 0.25,
        "tracking_inquiry": 0.25,
        "frustrated_venting": 0.25
    }
    bayes_manager = BayesianHypothesisManager(initial_priors)

    system = ClosedLoopTasteSystem(
        reranker=reranker,
        deberta_scorer=deberta,
        telemetry_scorer=telemetry_scorer,
        bayes_manager=bayes_manager,
        embedding_dim=EMBEDDING_DIM,
        learning_rate=0.02
    )

    # -------------------------------------------------------------
    # SIMULATION SCENARIO
    # User asks: "My order has not arrived yet."
    # -------------------------------------------------------------
    np.random.seed(42)

    # Synthetic User Semantic Centroid (representing user's historical taste vector)
    user_taste_centroid = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    user_taste_centroid /= np.linalg.norm(user_taste_centroid)

    # Synthetic Candidate Pool (Candidates generated by Planner / RAG)
    candidates = [
        CandidateItem("C1", "Issue immediate refund of $89.90", "refund_seeker", np.random.randn(EMBEDDING_DIM).astype(np.float32)),
        CandidateItem("C2", "Ship priority overnight replacement item", "replacement_needed", np.random.randn(EMBEDDING_DIM).astype(np.float32)),
        CandidateItem("C3", "Provide live delivery GPS driver tracking link", "tracking_inquiry", np.random.randn(EMBEDDING_DIM).astype(np.float32)),
        CandidateItem("C4", "Offer empathetic validation statement", "frustrated_venting", np.random.randn(EMBEDDING_DIM).astype(np.float32)),
    ]
    for c in candidates:
        c.semantic_embedding /= np.linalg.norm(c.semantic_embedding)

    print("\n-------------------------------------------------------------")
    print("STAGE 1: DCN-v2 TASTE RERANKER INFERENCE")
    print("-------------------------------------------------------------")
    ranked = system.rank_candidates(user_taste_centroid, candidates)
    for rank, (cand, score) in enumerate(ranked, 1):
        print(f"  Rank {rank}: [{cand.id}] ({cand.affinity_tag:<18}) Blended Score = {score:.4f} | Text: '{cand.text}'")

    # The Top Ranked candidate is selected by the agent
    selected_candidate, initial_pred_score = ranked[0]
    print(f"\n[*] Agent Executes Highest Ranked Candidate: [{selected_candidate.id}]")

    # -------------------------------------------------------------
    # STAGE 2: USER INTERACTION & TELEMETRY OBSERVATION
    # -------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print("STAGE 2: EXECUTION & USER REACTION")
    print("-------------------------------------------------------------")
    agent_emitted_msg = selected_candidate.text

    # Case A: User wanted tracking, but agent issued refund (Correction event)
    if selected_candidate.affinity_tag == "refund_seeker":
        user_reaction = "Wait, do not refund my money! I still want the item, where is the driver?"
        telemetry = TelemetryPayload(dwell_time_ms=1600, response_token_count=14, user_action="none")
    else:
        user_reaction = "Perfect, thank you so much!"
        telemetry = TelemetryPayload(dwell_time_ms=5800, response_token_count=14, user_action="copy")

    print(f"  Agent Response: \"{agent_emitted_msg}\"")
    print(f"  User Followup : \"{user_reaction}\"")
    print(f"  User Telemetry: Dwell = {telemetry.dwell_time_ms}ms, Action = '{telemetry.user_action}'")

    # -------------------------------------------------------------
    # STAGE 3: PROJECTD EVALUATION & ONLINE PREFERENCE UPDATE
    # -------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print("STAGE 3: PROJECTD SCORING & DCN-v2 PREFERENCE GRADIENT UPDATE")
    print("-------------------------------------------------------------")
    step_results = system.process_turn_and_update(
        user_taste_vector=user_taste_centroid,
        chosen_candidate=selected_candidate,
        agent_response=agent_emitted_msg,
        user_followup=user_reaction,
        telemetry=telemetry
    )

    print(f"  ProjectD Neural Mode        : {step_results['deberta_mode']}")
    print(f"  ProjectD DeBERTa Score      : {step_results['deberta_score']}")
    print(f"  ProjectD Telemetry Score    : {step_results['telemetry_score']}")
    print(f"  Reconciled Ground-Truth (y) : {step_results['reconciled_ground_truth']}")
    print(f"  DCN-v2 Initial Prediction   : {step_results['predicted_dcn_score_before']}")
    print(f"  Online Loss (BCE)           : {step_results['loss']}")
    print(f"  Pipeline Decision Action    : {step_results['action']}")
    print(f"  Updated Bayesian Priors     : {step_results['updated_priors']}")

    # -------------------------------------------------------------
    # STAGE 4: POST-UPDATE RE-RANKING CHECK
    # -------------------------------------------------------------
    print("\n-------------------------------------------------------------")
    print("STAGE 4: VERIFYING CLOSED-LOOP ADAPTATION")
    print("-------------------------------------------------------------")
    reranked_post = system.rank_candidates(user_taste_centroid, candidates)
    print("[*] Re-ranking the exact same candidate pool after 1 gradient step:")
    for rank, (cand, score) in enumerate(reranked_post, 1):
        print(f"  Rank {rank}: [{cand.id}] ({cand.affinity_tag:<18}) New Blended Score = {score:.4f}")

    print("\n[✓] Closed-loop cycle executed successfully.")