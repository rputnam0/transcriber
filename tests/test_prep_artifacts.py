from __future__ import annotations

import json

import numpy as np

from transcriber.prep_artifacts import (
    StageMetricsLogger,
    append_dataset,
    artifact_is_reusable,
    build_artifact_manifest,
    build_audio_quality_metrics,
    build_coverage_report,
    build_stage_manifest,
    quality_rejection_reason,
    save_manifest,
    stage_manifest_is_reusable,
)
from transcriber.segment_classifier import (
    ClassifierDataset,
    balance_classifier_dataset,
    merge_classifier_datasets,
    relabel_classifier_dataset_sources,
)


def _dataset(
    embeddings: list[list[float]],
    *,
    labels: list[str],
    domains: list[str],
    sources: list[str],
    sessions: list[str],
    durations: list[float] | None = None,
    dominant_shares: list[float] | None = None,
    active_speakers: list[int] | None = None,
) -> ClassifierDataset:
    count = len(labels)
    return ClassifierDataset(
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=list(labels),
        domains=list(domains),
        sources=list(sources),
        sessions=list(sessions),
        durations=np.asarray(durations or ([1.0] * count), dtype=np.float32),
        dominant_shares=np.asarray(dominant_shares or ([0.9] * count), dtype=np.float32),
        top1_powers=np.full(count, 0.3, dtype=np.float32),
        top2_powers=np.full(count, 0.05, dtype=np.float32),
        active_speakers=np.asarray(active_speakers or ([1] * count), dtype=np.int32),
    )


def test_artifact_is_reusable_enforces_manifest_and_diarizer(tmp_path):
    manifest_path = tmp_path / "dataset_manifest.json"

    reusable, reason = artifact_is_reusable(
        manifest_path,
        artifact_type="dataset",
        diarization_model="pyannote/speaker-diarization-community-1",
    )

    assert reusable is False
    assert reason == "missing_manifest"

    reusable, reason = artifact_is_reusable(
        manifest_path,
        artifact_type="dataset",
        diarization_model="pyannote/speaker-diarization-community-1",
        allow_legacy_reuse=True,
    )

    assert reusable is True
    assert reason == "missing_manifest"

    manifest = build_artifact_manifest(
        artifact_type="dataset",
        diarization_model="pyannote/speaker-diarization-community-1",
        source_sessions=["Session_10"],
        input_file_identities=[],
        build_params={"variant": "mixed_raw"},
        parent_artifacts=[],
        git_commit="deadbeef",
    )
    save_manifest(manifest_path, manifest)

    reusable, reason = artifact_is_reusable(
        manifest_path,
        artifact_type="dataset",
        diarization_model="pyannote/speaker-diarization-community-1",
        artifact_id=str(manifest["artifact_id"]),
    )

    assert reusable is True
    assert reason == "ok"

    reusable, reason = artifact_is_reusable(
        manifest_path,
        artifact_type="dataset",
        diarization_model="pyannote/speaker-diarization-legacy",
    )

    assert reusable is False
    assert reason == "diarization_model_mismatch"


def test_quality_rejection_reason_flags_clipping_silence_and_non_finite():
    clipped = build_audio_quality_metrics(np.ones(400, dtype=np.float32), sample_rate=100)
    silent = build_audio_quality_metrics(np.zeros(400, dtype=np.float32), sample_rate=100)
    non_finite = build_audio_quality_metrics(
        np.asarray([0.1, np.nan, 0.2], dtype=np.float32),
        sample_rate=100,
    )

    assert quality_rejection_reason(clipped, min_duration=0.1, max_duration=10.0) == "clipping"
    assert quality_rejection_reason(silent, min_duration=0.1, max_duration=10.0) == "silence"
    assert quality_rejection_reason(non_finite, min_duration=0.1, max_duration=10.0) == "non_finite"


def test_build_coverage_report_warns_for_low_sessions_and_low_samples():
    dataset = _dataset(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.0, 1.0],
        ],
        labels=["Alice", "Alice", "Alice", "Alice", "Bob"],
        domains=["mixed"] * 5,
        sources=["mixed_raw"] * 5,
        sessions=["Session_01", "Session_02", "Session_03", "Session_04", "Session_01"],
    )

    report = build_coverage_report(dataset)
    warnings = {(item["speaker"], item["kind"]) for item in report["warnings"]}

    assert ("Bob", "low_session_count") in warnings
    assert ("Bob", "low_sample_count") in warnings


def test_append_dataset_preserves_balanced_base_and_adds_hard_negatives():
    bank = _dataset(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        labels=["Alice", "Alice", "Bob", "Bob"],
        domains=["bank"] * 4,
        sources=["bank"] * 4,
        sessions=["bank"] * 4,
    )
    raw = _dataset(
        [[1.0, 0.2], [0.2, 1.0]],
        labels=["Alice", "Bob"],
        domains=["mixed"] * 2,
        sources=["mixed_raw"] * 2,
        sessions=["Session_10"] * 2,
    )
    aug = _dataset(
        [[1.0, 0.3], [1.0, 0.4], [0.3, 1.0], [0.4, 1.0]],
        labels=["Alice", "Alice", "Bob", "Bob"],
        domains=["mixed_aug"] * 4,
        sources=["light_x1"] * 4,
        sessions=["Session_10"] * 4,
        active_speakers=[2, 2, 2, 2],
    )
    hard_negative = _dataset(
        [[0.6, 0.4], [0.45, 0.55]],
        labels=["Alice", "Bob"],
        domains=["hard_negative"] * 2,
        sources=["hard_negative"] * 2,
        sessions=["Session_61", "Session_61"],
        dominant_shares=[0.62, 0.61],
        active_speakers=[2, 2],
    )

    merged = merge_classifier_datasets(
        [
            relabel_classifier_dataset_sources(bank, "bank"),
            raw,
            relabel_classifier_dataset_sources(aug, "mixed_aug_total"),
        ]
    )
    balanced, _summary = balance_classifier_dataset(
        merged,
        target_speakers=["Alice", "Bob"],
        max_samples_per_cell=10,
    )
    combined = append_dataset(balanced, hard_negative)

    assert balanced.sources.count("bank") == 4
    assert balanced.sources.count("mixed_raw") == 2
    assert balanced.sources.count("mixed_aug_total") == 2
    assert combined.sources.count("bank") == 4
    assert combined.sources.count("mixed_raw") == 2
    assert combined.sources.count("mixed_aug_total") == 2
    assert combined.sources.count("hard_negative") == 2
    assert combined.sources[-2:] == ["hard_negative", "hard_negative"]


def test_stage_manifest_reuse_requires_matching_signature_and_outputs(tmp_path):
    required_path = tmp_path / "artifact.txt"
    required_path.write_text("ok", encoding="utf-8")
    manifest_path = tmp_path / "stage.json"
    manifest = build_stage_manifest(
        stage="mixed_base",
        stage_signature={"artifact_id": "abc123", "version": 1},
        outputs={"artifact": str(required_path)},
        required_paths=[str(required_path)],
        parent_stages=["bank"],
        git_commit="deadbeef",
    )
    save_manifest(manifest_path, manifest)

    reusable, loaded, reason = stage_manifest_is_reusable(
        manifest_path,
        stage="mixed_base",
        stage_signature={"artifact_id": "abc123", "version": 1},
    )
    assert reusable is True
    assert loaded is not None
    assert reason == "ok"

    required_path.unlink()
    reusable, _loaded, reason = stage_manifest_is_reusable(
        manifest_path,
        stage="mixed_base",
        stage_signature={"artifact_id": "abc123", "version": 1},
    )
    assert reusable is False
    assert reason == "missing_required_output"


def test_stage_metrics_logger_emits_jsonl_records(tmp_path):
    log_path = tmp_path / "stage_metrics.jsonl"
    logger = StageMetricsLogger(log_path)
    logger.log(stage="bank", status="stage_started", cache_hit=False)
    logger.bind(stage="bank", variant="mixed_raw")(
        status="window_cache_hit",
        session="Session_10",
        cache_hit=True,
        elapsed_seconds=1.25,
        extra={"window_index": 1},
    )

    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert records[0]["stage"] == "bank"
    assert records[1]["variant"] == "mixed_raw"
    assert records[1]["session"] == "Session_10"
    assert records[1]["window_index"] == 1
