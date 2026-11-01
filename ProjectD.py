"""
ProjectD: Feedback Modeling & Hypothesis Disambiguation Engine
Dual-Backend: Evaluates fine-tuned weights or ONNX Runtime graph.
"""

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ==========================================
# 1. CORE DATA CONTRACTS & SCHEMAS
# ==========================================

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
class Hypothesis:
    id: str
    description: str
    prior_probability: float
    affinity_tag: str


@dataclass
class ScorerResult:
    scorer_name: str
    satisfaction_score: float  # [0.0, 1.0]
    confidence: float          # [0.0, 1.0]
    detected_mode: DissatisfactionMode
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==========================================
# 2. SIGNAL SCORERS (DUAL PYTORCH / ONNX)
# ==========================================

class DeBERTaFeedbackScorer:
    """
    Evaluates response-followup pairs using the trained neural classification head.
    Supports native PyTorch checkpoints and ONNX Runtime graphs.
    """
    def __init__(self, model_identifier: str):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier, use_fast=False)
        self.use_onnx = False

        onnx_file = Path(model_identifier) / "model.onnx"
        if onnx_file.exists():
            import onnxruntime as ort
            print(f"[*] Running with ONNX Runtime backend: {onnx_file}")
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 4
            self.session = ort.InferenceSession(str(onnx_file), sess_options=session_options, providers=["CPUExecutionProvider"])
            self.use_onnx = True
        else:
            import torch
            from transformers import AutoModelForSequenceClassification
            print(f"[*] Loading fine-tuned PyTorch checkpoint ({model_identifier})...")
            self.torch = torch
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
            return ScorerResult(
                scorer_name="deberta_scorer",
                satisfaction_score=0.5,
                confidence=0.1,
                detected_mode=DissatisfactionMode.NONE
            )

        if self.use_onnx:
            inputs = self.tokenizer(
                agent_response,
                user_followup,
                return_tensors="np",
                truncation="longest_first",
                max_length=256,
                padding=True
            )
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            logits = self.session.run(None, ort_inputs)[0][0]
        else:
            inputs = self.tokenizer(
                agent_response,
                user_followup,
                return_tensors="pt",
                truncation="longest_first",
                max_length=256,
                padding=True
            )
            with self.torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0].cpu().numpy()

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        pred_idx = int(np.argmax(probs))
        pred_mode = self.id2label.get(pred_idx, DissatisfactionMode.NONE)
        confidence = float(probs[pred_idx])

        # ProjectD satisfaction score formulation:
        # NONE (0) = 1.0, DRIFT (4) = 0.3, REPHRASE (1) = 0.2, CORRECTION (2) = 0.05, ABANDONMENT (3) = 0.0
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
    Scores user dwell time, actions, and implicit friction.
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


# ==========================================
# 3. DISAMBIGUATION & TASTE PRIOR ENGINE
# ==========================================

class HypothesisDisambiguator:
    """
    Maintains a learned Dirichlet-Multinomial Bayesian prior over hypotheses[cite: 1].
    """
    def __init__(self, initial_priors: Dict[str, float]):
        self.alpha_counts = {k: v * 10.0 for k, v in initial_priors.items()}

    def get_ranked_hypotheses(self, candidates: List[Hypothesis]) -> List[Hypothesis]:
        total_weight = sum(self.alpha_counts.get(c.affinity_tag, 1.0) for c in candidates)
        for c in candidates:
            c.prior_probability = self.alpha_counts.get(c.affinity_tag, 1.0) / total_weight
        return sorted(candidates, key=lambda x: x.prior_probability, reverse=True)

    def update_priors(self, selected_tag: str, satisfaction_score: float):
        learning_rate = 2.0
        if satisfaction_score > 0.6:
            self.alpha_counts[selected_tag] = self.alpha_counts.get(selected_tag, 1.0) + (learning_rate * satisfaction_score)
        else:
            penalty = learning_rate * (1.0 - satisfaction_score)
            self.alpha_counts[selected_tag] = max(0.5, self.alpha_counts.get(selected_tag, 1.0) - penalty)


# ==========================================
# 4. PROJECTD FEEDBACK PIPELINE
# ==========================================

class ProjectDFeedbackPipeline:
    def __init__(
        self,
        deberta_scorer: DeBERTaFeedbackScorer,
        telemetry_scorer: BehavioralTelemetryScorer,
        disambiguator: HypothesisDisambiguator
    ):
        self.deberta_scorer = deberta_scorer
        self.telemetry_scorer = telemetry_scorer
        self.disambiguator = disambiguator

    def evaluate_turn(
        self,
        chosen_hypothesis: Hypothesis,
        agent_response: str,
        user_followup: str,
        telemetry: TelemetryPayload
    ) -> Dict[str, Any]:
        
        deberta_res = self.deberta_scorer.score(agent_response, user_followup)
        telemetry_res = self.telemetry_scorer.score(telemetry)

        conflict_detected = False
        if deberta_res.satisfaction_score > 0.7 and telemetry_res.satisfaction_score < 0.3:
            conflict_detected = True
            effective_score = 0.25
            action = "ESCALATE_FOR_AUDIT"
        else:
            total_conf = deberta_res.confidence + telemetry_res.confidence
            effective_score = (
                (deberta_res.satisfaction_score * deberta_res.confidence) +
                (telemetry_res.satisfaction_score * telemetry_res.confidence)
            ) / max(total_conf, 1e-6)

            if effective_score >= 0.7:
                action = "ACCEPT_AND_REINFORCE"
            elif effective_score <= 0.35:
                action = "PENALIZE_AND_DOWNWEIGHT"
            else:
                action = "FLAG_FOR_REVIEW"

        self.disambiguator.update_priors(chosen_hypothesis.affinity_tag, effective_score)

        return {
            "effective_satisfaction": round(effective_score, 4),
            "action": action,
            "conflict_detected": conflict_detected,
            "deberta_result": {
                "score": deberta_res.satisfaction_score,
                "mode": deberta_res.detected_mode.value,
                "confidence": deberta_res.confidence
            },
            "telemetry_result": {
                "score": telemetry_res.satisfaction_score,
                "mode": telemetry_res.detected_mode.value,
                "confidence": telemetry_res.confidence
            },
            "updated_hypothesis_priors": {
                k: round(v / sum(self.disambiguator.alpha_counts.values()), 4)
                for k, v in self.disambiguator.alpha_counts.items()
            }
        }


# ==========================================
# 5. DEMONSTRATION & EXECUTION ENTRYPOINT
# ==========================================

if __name__ == "__main__":
    print("--- Initializing ProjectD Feedback Modeling Engine ---")
    
    current_dir = Path(__file__).resolve().parent
    finetuned_dir = current_dir / "deberta-feedback-finetuned"
    local_onnx_dir = current_dir / "deberta-feedback-onnx"

    # Priority 1: Fine-tuned checkpoint from train_projectd_scorer.py
    if finetuned_dir.exists() and (finetuned_dir / "config.json").exists():
        model_target = str(finetuned_dir)
        print(f"[*] Targeting fine-tuned weights at: {model_target}")
    # Priority 2: Custom ONNX model if available
    elif local_onnx_dir.exists() and (local_onnx_dir / "model.onnx").exists():
        model_target = str(local_onnx_dir)
        print(f"[*] Targeting local ONNX directory at: {model_target}")
    # Priority 3: Fallback base model
    else:
        model_target = "microsoft/deberta-v3-small"
        print(f"[*] Fallback to base model: {model_target}")

    deberta = DeBERTaFeedbackScorer(model_target)
    telemetry_scorer = BehavioralTelemetryScorer()
    
    initial_priors = {
        "refund_seeker": 0.25,
        "replacement_needed": 0.25,
        "tracking_inquiry": 0.25,
        "frustrated_venting": 0.25
    }
    disambiguator = HypothesisDisambiguator(initial_priors)
    pipeline = ProjectDFeedbackPipeline(deberta, telemetry_scorer, disambiguator)

    candidate_hypotheses = [
        Hypothesis("H1", "Issue immediate refund", 0.0, "refund_seeker"),
        Hypothesis("H2", "Ship priority replacement", 0.0, "replacement_needed"),
        Hypothesis("H3", "Supply real-time GPS tracking link", 0.0, "tracking_inquiry"),
        Hypothesis("H4", "Acknowledge frustration with empathy statement", 0.0, "frustrated_venting")
    ]

    print("\n--- Step A: Initial Hypothesis Priors ---")
    ranked = disambiguator.get_ranked_hypotheses(candidate_hypotheses)
    for h in ranked:
        print(f"  [{h.id}] {h.description:<50} P(H)={h.prior_probability:.3f}")

    # SIMULATION TURN 1: User says order didn't arrive, agent assumes refund (H1) -> User corrects
    chosen = candidate_hypotheses[0]
    agent_msg = "I have initiated a full refund of $89.90 back to your card."
    user_reaction = "Wait, don't refund it! I just wanted to know when it is coming."
    user_telemetry = TelemetryPayload(
        dwell_time_ms=1800,
        response_token_count=15,
        user_action="none",
        rephrase_similarity=0.2
    )

    print("\n--- Step B: Evaluating Turn 1 Feedback ---")
    eval_result = pipeline.evaluate_turn(chosen, agent_msg, user_reaction, user_telemetry)
    print(f"Satisfaction Score : {eval_result['effective_satisfaction']}")
    print(f"Action Taken       : {eval_result['action']}")
    print(f"Dissatisfaction    : {eval_result['deberta_result']['mode']}")
    print(f"Updated Priors     : {eval_result['updated_hypothesis_priors']}")

    # SIMULATION TURN 2: Agent adapts to H3 (Tracking info) -> User accepts
    chosen_next = candidate_hypotheses[2]
    agent_msg_2 = "Your driver is 3 stops away. Here is the tracking URL: https://track.delivery/123"
    user_reaction_2 = "Perfect, thank you so much!"
    user_telemetry_2 = TelemetryPayload(
        dwell_time_ms=6200,
        response_token_count=16,
        user_action="copy"
    )

    print("\n--- Step C: Evaluating Turn 2 Feedback ---")
    eval_result_2 = pipeline.evaluate_turn(chosen_next, agent_msg_2, user_reaction_2, user_telemetry_2)
    print(f"Satisfaction Score : {eval_result_2['effective_satisfaction']}")
    print(f"Action Taken       : {eval_result_2['action']}")
    print(f"Dissatisfaction    : {eval_result_2['deberta_result']['mode']}")
    print(f"Updated Priors     : {eval_result_2['updated_hypothesis_priors']}")