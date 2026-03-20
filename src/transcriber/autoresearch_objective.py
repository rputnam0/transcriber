from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional


@dataclass(frozen=True)
class ObjectiveWeights:
    session61_matched_accuracy: float
    session61_accuracy: float
    short_slice_matched_accuracy: float
    session22_accuracy: float
    session22_matched_accuracy: float


@dataclass(frozen=True)
class DevAcceptanceGate:
    min_score_gain: float
    min_session61_matched_accuracy_gain: float
    max_session22_accuracy_regression: float


@dataclass(frozen=True)
class FinalGate:
    max_mean_accuracy_regression: float
    max_mean_matched_accuracy_regression: float


def _session_metric(
    metrics: Mapping[str, Mapping[str, object]],
    session_name: str,
    metric_name: str,
) -> float:
    session_metrics = dict(metrics.get(session_name) or {})
    value = session_metrics.get(metric_name)
    return float(value) if isinstance(value, (int, float)) else 0.0


def compute_dev_score(
    metrics: Mapping[str, Mapping[str, object]],
    *,
    weights: ObjectiveWeights,
) -> float:
    return (
        weights.session61_matched_accuracy
        * _session_metric(metrics, "Session61", "mean_matched_accuracy")
        + weights.session61_accuracy * _session_metric(metrics, "Session61", "mean_accuracy")
        + weights.short_slice_matched_accuracy
        * _session_metric(metrics, "short_segment_slice", "mean_matched_accuracy")
        + weights.session22_accuracy * _session_metric(metrics, "Session22", "mean_accuracy")
        + weights.session22_matched_accuracy
        * _session_metric(metrics, "Session22", "mean_matched_accuracy")
    )


def evaluate_dev_candidate(
    candidate_metrics: Mapping[str, Mapping[str, object]],
    *,
    champion_metrics: Mapping[str, Mapping[str, object]],
    weights: ObjectiveWeights,
    gate: DevAcceptanceGate,
) -> Dict[str, object]:
    candidate_score = compute_dev_score(candidate_metrics, weights=weights)
    champion_score = compute_dev_score(champion_metrics, weights=weights)
    session61_gain = _session_metric(
        candidate_metrics, "Session61", "mean_matched_accuracy"
    ) - _session_metric(champion_metrics, "Session61", "mean_matched_accuracy")
    session22_accuracy_delta = _session_metric(
        candidate_metrics, "Session22", "mean_accuracy"
    ) - _session_metric(champion_metrics, "Session22", "mean_accuracy")

    accepted = bool(
        candidate_score >= champion_score + gate.min_score_gain
        and session61_gain >= gate.min_session61_matched_accuracy_gain
        and session22_accuracy_delta >= -gate.max_session22_accuracy_regression
    )
    return {
        "accepted": accepted,
        "candidate_score": candidate_score,
        "champion_score": champion_score,
        "score_gain": candidate_score - champion_score,
        "session61_matched_accuracy_gain": session61_gain,
        "session22_accuracy_delta": session22_accuracy_delta,
    }


def summarize_final_metrics(
    metrics: Mapping[str, Mapping[str, object]],
) -> Dict[str, float]:
    session_names = list(metrics.keys())
    if not session_names:
        return {
            "mean_accuracy": 0.0,
            "mean_matched_accuracy": 0.0,
        }
    accuracy_values = [
        _session_metric(metrics, session_name, "mean_accuracy") for session_name in session_names
    ]
    matched_values = [
        _session_metric(metrics, session_name, "mean_matched_accuracy")
        for session_name in session_names
    ]
    return {
        "mean_accuracy": sum(accuracy_values) / len(accuracy_values),
        "mean_matched_accuracy": sum(matched_values) / len(matched_values),
    }


def evaluate_final_gate(
    candidate_metrics: Mapping[str, Mapping[str, object]],
    *,
    champion_metrics: Mapping[str, Mapping[str, object]],
    gate: FinalGate,
) -> Dict[str, object]:
    candidate_summary = summarize_final_metrics(candidate_metrics)
    champion_summary = summarize_final_metrics(champion_metrics)
    accuracy_delta = (
        float(candidate_summary["mean_accuracy"]) - float(champion_summary["mean_accuracy"])
    )
    matched_delta = (
        float(candidate_summary["mean_matched_accuracy"])
        - float(champion_summary["mean_matched_accuracy"])
    )
    accepted = bool(
        accuracy_delta >= -gate.max_mean_accuracy_regression
        and matched_delta >= -gate.max_mean_matched_accuracy_regression
    )
    return {
        "accepted": accepted,
        "candidate_summary": candidate_summary,
        "champion_summary": champion_summary,
        "mean_accuracy_delta": accuracy_delta,
        "mean_matched_accuracy_delta": matched_delta,
    }


def load_objective_weights(raw: Optional[Mapping[str, object]]) -> ObjectiveWeights:
    payload = dict(raw or {})
    return ObjectiveWeights(
        session61_matched_accuracy=float(payload.get("session61_matched_accuracy", 0.40)),
        session61_accuracy=float(payload.get("session61_accuracy", 0.25)),
        short_slice_matched_accuracy=float(payload.get("short_slice_matched_accuracy", 0.20)),
        session22_accuracy=float(payload.get("session22_accuracy", 0.10)),
        session22_matched_accuracy=float(payload.get("session22_matched_accuracy", 0.05)),
    )


def load_dev_acceptance_gate(raw: Optional[Mapping[str, object]]) -> DevAcceptanceGate:
    payload = dict(raw or {})
    return DevAcceptanceGate(
        min_score_gain=float(payload.get("min_score_gain", 0.003)),
        min_session61_matched_accuracy_gain=float(
            payload.get("min_session61_matched_accuracy_gain", 0.005)
        ),
        max_session22_accuracy_regression=float(
            payload.get("max_session22_accuracy_regression", 0.010)
        ),
    )


def load_final_gate(raw: Optional[Mapping[str, object]]) -> FinalGate:
    payload = dict(raw or {})
    return FinalGate(
        max_mean_accuracy_regression=float(payload.get("max_mean_accuracy_regression", 0.005)),
        max_mean_matched_accuracy_regression=float(
            payload.get("max_mean_matched_accuracy_regression", 0.005)
        ),
    )

