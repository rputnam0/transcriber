from __future__ import annotations

from transcriber.autoresearch_objective import (
    DevAcceptanceGate,
    FinalGate,
    ObjectiveWeights,
    compute_dev_score,
    evaluate_dev_candidate,
    evaluate_final_gate,
)


def _metrics(
    *,
    session22_accuracy: float,
    session22_matched_accuracy: float,
    session61_accuracy: float,
    session61_matched_accuracy: float,
    short_slice_matched_accuracy: float,
) -> dict[str, dict[str, float]]:
    return {
        "Session22": {
            "mean_accuracy": session22_accuracy,
            "mean_matched_accuracy": session22_matched_accuracy,
        },
        "Session61": {
            "mean_accuracy": session61_accuracy,
            "mean_matched_accuracy": session61_matched_accuracy,
        },
        "short_segment_slice": {
            "mean_matched_accuracy": short_slice_matched_accuracy,
        },
    }


def test_compute_dev_score_uses_balanced_composite_weights():
    weights = ObjectiveWeights(0.40, 0.25, 0.20, 0.10, 0.05)
    metrics = _metrics(
        session22_accuracy=0.75,
        session22_matched_accuracy=0.77,
        session61_accuracy=0.65,
        session61_matched_accuracy=0.67,
        short_slice_matched_accuracy=0.69,
    )

    score = compute_dev_score(metrics, weights=weights)

    expected = (0.40 * 0.67) + (0.25 * 0.65) + (0.20 * 0.69) + (0.10 * 0.75) + (0.05 * 0.77)
    assert score == expected


def test_dev_acceptance_requires_score_gain_session61_gain_and_session22_guardrail():
    weights = ObjectiveWeights(0.40, 0.25, 0.20, 0.10, 0.05)
    gate = DevAcceptanceGate(
        min_score_gain=0.003,
        min_session61_matched_accuracy_gain=0.005,
        max_session22_accuracy_regression=0.010,
    )
    champion = _metrics(
        session22_accuracy=0.7627,
        session22_matched_accuracy=0.7770,
        session61_accuracy=0.6333,
        session61_matched_accuracy=0.6515,
        short_slice_matched_accuracy=0.6916,
    )

    accepted_candidate = _metrics(
        session22_accuracy=0.7530,
        session22_matched_accuracy=0.7680,
        session61_accuracy=0.6500,
        session61_matched_accuracy=0.6687,
        short_slice_matched_accuracy=0.6872,
    )
    accepted = evaluate_dev_candidate(
        accepted_candidate,
        champion_metrics=champion,
        weights=weights,
        gate=gate,
    )
    assert accepted["accepted"] is True

    rejected_candidate = _metrics(
        session22_accuracy=0.7490,
        session22_matched_accuracy=0.7660,
        session61_accuracy=0.6500,
        session61_matched_accuracy=0.6687,
        short_slice_matched_accuracy=0.6872,
    )
    rejected = evaluate_dev_candidate(
        rejected_candidate,
        champion_metrics=champion,
        weights=weights,
        gate=gate,
    )
    assert rejected["accepted"] is False


def test_final_gate_rejects_mean_regression_beyond_budget():
    gate = FinalGate(
        max_mean_accuracy_regression=0.005,
        max_mean_matched_accuracy_regression=0.005,
    )
    champion = {
        "Session58": {"mean_accuracy": 0.60, "mean_matched_accuracy": 0.70},
        "Session52": {"mean_accuracy": 0.58, "mean_matched_accuracy": 0.68},
        "Session48": {"mean_accuracy": 0.62, "mean_matched_accuracy": 0.72},
    }
    candidate = {
        "Session58": {"mean_accuracy": 0.58, "mean_matched_accuracy": 0.69},
        "Session52": {"mean_accuracy": 0.57, "mean_matched_accuracy": 0.67},
        "Session48": {"mean_accuracy": 0.60, "mean_matched_accuracy": 0.70},
    }

    result = evaluate_final_gate(candidate, champion_metrics=champion, gate=gate)

    assert result["accepted"] is False
    assert result["mean_accuracy_delta"] < -0.005
