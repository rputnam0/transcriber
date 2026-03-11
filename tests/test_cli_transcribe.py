from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Keep environment predictable for run_transcribe tests."""
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)


def _fake_segments_payload() -> Dict[str, Any]:
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello world",
                "speaker": "SPEAKER_00",
            }
        ],
        "extra": {"note": "metadata that should be ignored"},
    }


def test_run_transcribe_normalises_segment_payload(monkeypatch, tmp_path):
    """Ensure dict-based segment payloads are converted before consolidation."""
    from transcriber import cli as cli_mod
    from transcriber.transcript_pipeline import TranscriptPipelineResult

    fake_input = tmp_path / "input.zip"
    fake_input.write_text("dummy")

    def fake_gather_inputs(path: str) -> Tuple[List[str], None]:
        assert path == str(fake_input)
        return ([str(tmp_path / "track1.wav")], None)

    def fake_save_outputs(
        *,
        base_stem: str,
        output_dir: str,
        per_file_segments: List[Tuple[str, List[Dict[str, Any]]]],
        consolidated_pairs: List[Tuple[str, str, str]],
        diar_by_file: Dict[str, List[Dict[str, Any]]] | None,
        exclusive_diar_by_file: Dict[str, List[Dict[str, Any]]] | None,
        write_srt_file: bool,
        write_jsonl_file: bool,
    ) -> Path:
        # Expect normalised tuples with list-of-dict segments
        assert per_file_segments and len(per_file_segments[0]) == 2
        filename, segments = per_file_segments[0]
        assert filename.endswith("track1.wav")
        assert isinstance(segments, list)
        assert segments[0]["text"] == "hello world"
        # consolidated data should match the normalised structure
        assert consolidated_pairs[0][2] == "hello world"
        if diar_by_file:
            diar_entry = diar_by_file[filename][0]
            assert diar_entry["speaker"] == "SPEAKER_00"
        return Path(output_dir)

    def fake_transcribe_with_faster_pipeline(*args: Any, **kwargs: Any):
        segments = _fake_segments_payload()["segments"]
        diar = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
        return TranscriptPipelineResult(
            segments=segments,
            diarization_segments=diar,
            exclusive_diarization_segments=diar,
            speaker_embeddings={},
            metadata={},
        )

    monkeypatch.setattr(cli_mod, "gather_inputs", fake_gather_inputs)
    monkeypatch.setattr(cli_mod, "save_outputs", fake_save_outputs)
    monkeypatch.setattr(cli_mod, "cleanup_tmp", lambda *_args: None)
    monkeypatch.setattr(
        "transcriber.transcript_pipeline.transcribe_with_faster_pipeline",
        fake_transcribe_with_faster_pipeline,
    )
    monkeypatch.setattr("transcriber.diarization._detect_device", lambda: "cpu")
    monkeypatch.setattr(cli_mod, "_ensure_cuda_libs_on_path", lambda: None)
    monkeypatch.setattr(cli_mod, "_preload_cudnn_libs", lambda: None)

    cli_mod.run_transcribe(
        input_path=str(fake_input),
        backend="faster",
        model_name="tiny",
        compute_type="int8",
        batch_size=4,
        output_dir=str(tmp_path / "outputs"),
        hf_cache_root=None,
        speaker_bank_root=None,
        write_srt=False,
        write_jsonl=False,
        auto_batch=False,
        speaker_bank_config=None,
    )


def test_aggregate_segment_label_candidates_uses_full_candidate_evidence():
    from transcriber.cli import _aggregate_segment_label_candidates

    seg_matches = {
        0: {
            "accepted": True,
            "match": {"speaker": "Dungeon Master", "score": 0.91},
            "candidates": [
                {"speaker": "Dungeon Master", "score": 0.91, "cluster_id": "dm0", "distance": 0.1},
                {
                    "speaker": "Cletus Cobbington",
                    "score": 0.20,
                    "cluster_id": "cl0",
                    "distance": 0.8,
                },
            ],
        },
        1: {
            "accepted": True,
            "match": {"speaker": "Dungeon Master", "score": 0.89},
            "candidates": [
                {"speaker": "Dungeon Master", "score": 0.89, "cluster_id": "dm0", "distance": 0.1},
                {
                    "speaker": "Cletus Cobbington",
                    "score": 0.22,
                    "cluster_id": "cl0",
                    "distance": 0.8,
                },
            ],
        },
        2: {
            "accepted": False,
            "match": None,
            "candidates": [
                {
                    "speaker": "Cletus Cobbington",
                    "score": 0.59,
                    "cluster_id": "cl1",
                    "distance": 0.2,
                },
                {"speaker": "Dungeon Master", "score": 0.10, "cluster_id": "dm0", "distance": 0.9},
            ],
        },
        3: {
            "accepted": False,
            "match": None,
            "candidates": [
                {
                    "speaker": "Cletus Cobbington",
                    "score": 0.60,
                    "cluster_id": "cl1",
                    "distance": 0.2,
                },
                {"speaker": "Dungeon Master", "score": 0.11, "cluster_id": "dm0", "distance": 0.9},
            ],
        },
        4: {
            "accepted": False,
            "match": None,
            "candidates": [
                {
                    "speaker": "Cletus Cobbington",
                    "score": 0.58,
                    "cluster_id": "cl1",
                    "distance": 0.2,
                },
                {"speaker": "Dungeon Master", "score": 0.09, "cluster_id": "dm0", "distance": 0.9},
            ],
        },
        5: {
            "accepted": False,
            "match": None,
            "candidates": [
                {
                    "speaker": "Cletus Cobbington",
                    "score": 0.61,
                    "cluster_id": "cl1",
                    "distance": 0.2,
                },
                {"speaker": "Dungeon Master", "score": 0.10, "cluster_id": "dm0", "distance": 0.9},
            ],
        },
    }

    selection, stats = _aggregate_segment_label_candidates(
        [0, 1, 2, 3, 4, 5],
        seg_matches,
        aggregation="mean",
        threshold=0.45,
        margin_required=0.03,
        min_segments_per_label=3,
    )

    assert selection is not None
    assert selection["speaker"] == "Cletus Cobbington"
    assert selection["score_max"] == pytest.approx(0.61)
    assert stats["segments_embedded"] == 6
    assert stats["segments_matched"] == 2
    assert stats["means"]["Cletus Cobbington"] > stats["means"]["Dungeon Master"]


def test_aggregate_segment_label_candidates_respects_threshold():
    from transcriber.cli import _aggregate_segment_label_candidates

    seg_matches = {
        0: {
            "accepted": False,
            "match": None,
            "candidates": [
                {"speaker": "Dungeon Master", "score": 0.61, "cluster_id": "dm0", "distance": 0.3},
                {
                    "speaker": "Cletus Cobbington",
                    "score": 0.35,
                    "cluster_id": "cl0",
                    "distance": 0.7,
                },
            ],
        },
        1: {
            "accepted": False,
            "match": None,
            "candidates": [
                {"speaker": "Dungeon Master", "score": 0.60, "cluster_id": "dm0", "distance": 0.3},
                {
                    "speaker": "Cletus Cobbington",
                    "score": 0.34,
                    "cluster_id": "cl0",
                    "distance": 0.7,
                },
            ],
        },
        2: {
            "accepted": False,
            "match": None,
            "candidates": [
                {"speaker": "Dungeon Master", "score": 0.59, "cluster_id": "dm0", "distance": 0.3},
                {
                    "speaker": "Cletus Cobbington",
                    "score": 0.33,
                    "cluster_id": "cl0",
                    "distance": 0.7,
                },
            ],
        },
    }

    selection, stats = _aggregate_segment_label_candidates(
        [0, 1, 2],
        seg_matches,
        aggregation="mean",
        threshold=0.65,
        margin_required=0.05,
        min_segments_per_label=2,
    )

    assert selection is None
    assert stats["score_metric"] == pytest.approx(0.6)


def test_resolve_speaker_bank_paths_accepts_speaker_bank_directory_root(tmp_path):
    from transcriber.cli import _resolve_speaker_bank_paths
    from transcriber.speaker_bank import SpeakerBankConfig

    base_root = tmp_path / "root"
    profile_dir = base_root / "speaker_bank" / "demo"
    profile_dir.mkdir(parents=True)

    root, profile, resolved_profile_dir = _resolve_speaker_bank_paths(
        SpeakerBankConfig(path="demo"),
        str(base_root / "speaker_bank"),
        None,
    )

    assert root == base_root / "speaker_bank"
    assert profile == "demo"
    assert resolved_profile_dir == profile_dir


def test_run_transcribe_extracts_segment_embeddings_per_raw_label(monkeypatch, tmp_path):
    from transcriber import cli as cli_mod
    from transcriber.diarization import SegmentEmbeddingResult
    from transcriber.speaker_bank import SpeakerBankConfig
    from transcriber.transcript_pipeline import TranscriptPipelineResult

    fake_input = tmp_path / "input.m4a"
    fake_input.write_text("dummy")
    seen_label_sets: List[set[str]] = []
    seen_as_norm_args: List[Tuple[bool, int]] = []

    def fake_gather_inputs(path: str) -> Tuple[List[str], None]:
        assert path == str(fake_input)
        return ([str(fake_input)], None)

    def fake_transcribe_with_faster_pipeline(*args: Any, **kwargs: Any):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "first", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "second", "speaker": "SPEAKER_01"},
            {"start": 2.0, "end": 3.0, "text": "third", "speaker": "SPEAKER_00"},
            {"start": 3.0, "end": 4.0, "text": "fourth", "speaker": "SPEAKER_01"},
        ]
        speaker_embeddings = {
            "SPEAKER_00": np.array([1.0, 0.0], dtype=np.float32),
            "SPEAKER_01": np.array([0.0, 1.0], dtype=np.float32),
        }
        diar = [dict(item) for item in segments]
        return TranscriptPipelineResult(
            segments=segments,
            diarization_segments=diar,
            exclusive_diarization_segments=diar,
            speaker_embeddings=speaker_embeddings,
            metadata={},
        )

    def fake_extract_embeddings_for_segments(
        audio_path: str,
        segments: List[Tuple[float, float, str]],
        hf_token: str | None,
        **kwargs: Any,
    ):
        labels = {label for _, _, label in segments}
        seen_label_sets.append(labels)
        assert len(labels) == 1
        label = next(iter(labels))
        vector = (
            np.array([1.0, 0.0], dtype=np.float32)
            if label == "SPEAKER_00"
            else np.array([0.0, 1.0], dtype=np.float32)
        )
        results = [
            SegmentEmbeddingResult(
                speaker=label,
                start=float(start),
                end=float(end),
                index=index,
                embedding=vector,
            )
            for index, (start, end, label) in enumerate(segments)
        ]
        return results, {"embedded": len(results), "skipped": 0, "total": len(results)}

    class FakeSpeakerBank:
        def __init__(self, *args: Any, **kwargs: Any):
            self.profile = "default"
            self.is_empty = False

        def summary(self) -> Dict[str, Any]:
            return {
                "profile": "default",
                "speakers": ["Alice", "Bob"],
                "entries": 2,
                "clusters": {},
            }

        def score_candidates(
            self,
            embedding: np.ndarray,
            *,
            radius_factor: float = 2.5,
            as_norm_enabled: bool = False,
            as_norm_cohort_size: int = 50,
        ):
            seen_as_norm_args.append((as_norm_enabled, as_norm_cohort_size))
            if float(embedding[0]) > float(embedding[1]):
                return [
                    {
                        "speaker": "Alice",
                        "cluster_id": "alice",
                        "score": 0.95,
                        "raw_score": 0.95,
                        "distance": 0.05,
                        "source": "prototype",
                    },
                    {
                        "speaker": "Bob",
                        "cluster_id": "bob",
                        "score": 0.20,
                        "raw_score": 0.20,
                        "distance": 0.80,
                        "source": "prototype",
                    },
                ]
            return [
                {
                    "speaker": "Bob",
                    "cluster_id": "bob",
                    "score": 0.96,
                    "raw_score": 0.96,
                    "distance": 0.04,
                    "source": "prototype",
                },
                {
                    "speaker": "Alice",
                    "cluster_id": "alice",
                    "score": 0.22,
                    "raw_score": 0.22,
                    "distance": 0.78,
                    "source": "prototype",
                },
            ]

        def match(
            self,
            embedding: np.ndarray,
            *,
            threshold: float,
            radius_factor: float = 2.5,
            margin: float = 0.0,
            as_norm_enabled: bool = False,
            as_norm_cohort_size: int = 50,
        ) -> Dict[str, Any] | None:
            candidates = self.score_candidates(
                embedding,
                radius_factor=radius_factor,
                as_norm_enabled=as_norm_enabled,
                as_norm_cohort_size=as_norm_cohort_size,
            )
            top1 = candidates[0]
            top2 = candidates[1]["score"]
            if top1["score"] < threshold or (top1["score"] - top2) < margin:
                return None
            return {
                **top1,
                "margin": top1["score"] - top2,
                "second_best": top2,
            }

    def fake_save_outputs(
        *,
        base_stem: str,
        output_dir: str,
        per_file_segments: List[Tuple[str, List[Dict[str, Any]]]],
        consolidated_pairs: List[Tuple[str, str, str]],
        diar_by_file: Dict[str, List[Dict[str, Any]]] | None,
        exclusive_diar_by_file: Dict[str, List[Dict[str, Any]]] | None,
        write_srt_file: bool,
        write_jsonl_file: bool,
    ) -> Path:
        speakers = [segment["speaker"] for _, segments in per_file_segments for segment in segments]
        assert speakers == ["Alice", "Bob", "Alice", "Bob"]
        return Path(output_dir)

    monkeypatch.setattr(cli_mod, "gather_inputs", fake_gather_inputs)
    monkeypatch.setattr(cli_mod, "save_outputs", fake_save_outputs)
    monkeypatch.setattr(cli_mod, "cleanup_tmp", lambda *_args: None)
    monkeypatch.setattr(cli_mod, "SpeakerBank", FakeSpeakerBank)
    monkeypatch.setattr(
        "transcriber.transcript_pipeline.transcribe_with_faster_pipeline",
        fake_transcribe_with_faster_pipeline,
    )
    monkeypatch.setattr(
        "transcriber.diarization.extract_embeddings_for_segments",
        fake_extract_embeddings_for_segments,
    )
    monkeypatch.setattr("transcriber.diarization._detect_device", lambda: "cpu")
    monkeypatch.setattr(cli_mod, "_ensure_cuda_libs_on_path", lambda: None)
    monkeypatch.setattr(cli_mod, "_preload_cudnn_libs", lambda: None)

    cli_mod.run_transcribe(
        input_path=str(fake_input),
        backend="faster",
        model_name="tiny",
        compute_type="int8",
        batch_size=4,
        output_dir=str(tmp_path / "outputs"),
        hf_cache_root=None,
        speaker_bank_root=str(tmp_path / "speaker_bank_root"),
        write_srt=False,
        write_jsonl=False,
        auto_batch=False,
        speaker_bank_config=SpeakerBankConfig(
            enabled=True,
            use_existing=True,
            emit_pca=False,
            match_per_segment=True,
            min_segments_per_label=1,
            threshold=0.5,
            scoring_margin=0.1,
            scoring_as_norm_enabled=True,
            scoring_as_norm_cohort_size=7,
        ),
    )

    assert seen_label_sets == [{"SPEAKER_00"}, {"SPEAKER_01"}]
    assert seen_as_norm_args
    assert set(seen_as_norm_args) == {(True, 7)}


def test_resolve_speaker_bank_settings_reads_classifier_config():
    from transcriber.cli import _resolve_speaker_bank_settings

    cfg = {
        "speaker_bank": {
            "path": "latest_eval",
            "classifier": {
                "model": "knn",
                "n_neighbors": 11,
                "c": 2.0,
                "min_confidence": 0.25,
                "min_margin": 0.14,
                "input_paths": ["/mnt/g/My Drive/DND/Audio"],
                "transcript_roots": ["/mnt/g/My Drive/DND/Transcripts"],
            },
        }
    }

    speaker_bank_config, train_only = _resolve_speaker_bank_settings(cfg, SimpleNamespace())

    assert train_only is None
    assert speaker_bank_config is not None
    assert speaker_bank_config.path == "latest_eval"
    assert speaker_bank_config.classifier_model == "knn"
    assert speaker_bank_config.classifier_n_neighbors == 11
    assert speaker_bank_config.classifier_c == pytest.approx(2.0)
    assert speaker_bank_config.classifier_min_confidence == pytest.approx(0.25)
    assert speaker_bank_config.classifier_min_margin == pytest.approx(0.14)
    assert speaker_bank_config.classifier_input_paths == ["/mnt/g/My Drive/DND/Audio"]
    assert speaker_bank_config.classifier_transcript_roots == ["/mnt/g/My Drive/DND/Transcripts"]


def test_run_transcribe_segment_classifier_overrides_bank_when_confident(monkeypatch, tmp_path):
    from transcriber import cli as cli_mod
    from transcriber.diarization import SegmentEmbeddingResult
    from transcriber.speaker_bank import SpeakerBankConfig
    from transcriber.transcript_pipeline import TranscriptPipelineResult

    fake_input = tmp_path / "input.m4a"
    fake_input.write_text("dummy")

    def fake_gather_inputs(path: str) -> Tuple[List[str], None]:
        assert path == str(fake_input)
        return ([str(fake_input)], None)

    def fake_transcribe_with_faster_pipeline(*args: Any, **kwargs: Any):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "one", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "two", "speaker": "SPEAKER_01"},
        ]
        speaker_embeddings = {
            "SPEAKER_00": np.array([1.0, 0.0], dtype=np.float32),
            "SPEAKER_01": np.array([0.0, 1.0], dtype=np.float32),
        }
        diar = [dict(item) for item in segments]
        return TranscriptPipelineResult(
            segments=segments,
            diarization_segments=diar,
            exclusive_diarization_segments=diar,
            speaker_embeddings=speaker_embeddings,
            metadata={},
        )

    def fake_extract_embeddings_for_segments(
        audio_path: str,
        segments: List[Tuple[float, float, str]],
        hf_token: str | None,
        **kwargs: Any,
    ):
        results = [
            SegmentEmbeddingResult(
                speaker=label,
                start=float(start),
                end=float(end),
                index=index,
                embedding=(
                    np.array([1.0, 0.0], dtype=np.float32)
                    if label == "SPEAKER_00"
                    else np.array([0.0, 1.0], dtype=np.float32)
                ),
            )
            for index, (start, end, label) in enumerate(segments)
        ]
        return results, {"embedded": len(results), "skipped": 0, "total": len(results)}

    class FakeSpeakerBank:
        def __init__(self, *args: Any, **kwargs: Any):
            self.is_empty = False

        def summary(self) -> Dict[str, Any]:
            return {"profile": "default", "speakers": ["BankAlice", "BankBob"], "entries": 2}

        def score_candidates(self, embedding: np.ndarray, **kwargs: Any):
            if float(embedding[0]) > float(embedding[1]):
                return [
                    {
                        "speaker": "BankAlice",
                        "cluster_id": "bank-a",
                        "score": 0.92,
                        "raw_score": 0.92,
                        "distance": 0.08,
                        "source": "prototype",
                    }
                ]
            return [
                {
                    "speaker": "BankBob",
                    "cluster_id": "bank-b",
                    "score": 0.93,
                    "raw_score": 0.93,
                    "distance": 0.07,
                    "source": "prototype",
                }
            ]

        def match(self, embedding: np.ndarray, **kwargs: Any) -> Dict[str, Any] | None:
            candidate = self.score_candidates(embedding)[0]
            return {**candidate, "margin": candidate["score"], "second_best": 0.0}

    class FakeSegmentClassifier:
        def summary(self) -> Dict[str, Any]:
            return {"samples": 10, "speakers": {"ClassifierAlice": 5, "BankBob": 5}}

        def score_candidates(self, embedding: np.ndarray) -> List[Dict[str, Any]]:
            if float(embedding[0]) > float(embedding[1]):
                return [
                    {"speaker": "ClassifierAlice", "score": 0.80, "source": "segment_classifier"},
                    {"speaker": "BankAlice", "score": 0.20, "source": "segment_classifier"},
                ]
            return [
                {"speaker": "BankBob", "score": 0.52, "source": "segment_classifier"},
                {"speaker": "ClassifierAlice", "score": 0.48, "source": "segment_classifier"},
            ]

        def predict(
            self,
            embedding: np.ndarray,
            *,
            min_confidence: float,
            min_margin: float,
        ):
            if float(embedding[0]) > float(embedding[1]):
                from transcriber.segment_classifier import SegmentClassifierPrediction

                return SegmentClassifierPrediction(
                    speaker="ClassifierAlice",
                    score=0.80,
                    margin=0.60,
                    second_best=0.20,
                    candidates=self.score_candidates(embedding),
                )
            return None

    def fake_save_outputs(
        *,
        base_stem: str,
        output_dir: str,
        per_file_segments: List[Tuple[str, List[Dict[str, Any]]]],
        consolidated_pairs: List[Tuple[str, str, str]],
        diar_by_file: Dict[str, List[Dict[str, Any]]] | None,
        exclusive_diar_by_file: Dict[str, List[Dict[str, Any]]] | None,
        write_srt_file: bool,
        write_jsonl_file: bool,
    ) -> Path:
        speakers = [segment["speaker"] for _, segments in per_file_segments for segment in segments]
        assert speakers == ["ClassifierAlice", "BankBob"]
        assert per_file_segments[0][1][0]["speaker_match_source"] == "segment_classifier"
        assert per_file_segments[0][1][1]["speaker_match_source"] == "segment_prototype"
        return Path(output_dir)

    monkeypatch.setattr(cli_mod, "gather_inputs", fake_gather_inputs)
    monkeypatch.setattr(cli_mod, "save_outputs", fake_save_outputs)
    monkeypatch.setattr(cli_mod, "cleanup_tmp", lambda *_args: None)
    monkeypatch.setattr(cli_mod, "SpeakerBank", FakeSpeakerBank)
    monkeypatch.setattr(
        cli_mod, "load_segment_classifier", lambda *_args, **_kwargs: FakeSegmentClassifier()
    )
    monkeypatch.setattr(
        "transcriber.transcript_pipeline.transcribe_with_faster_pipeline",
        fake_transcribe_with_faster_pipeline,
    )
    monkeypatch.setattr(
        "transcriber.diarization.extract_embeddings_for_segments",
        fake_extract_embeddings_for_segments,
    )
    monkeypatch.setattr("transcriber.diarization._detect_device", lambda: "cpu")
    monkeypatch.setattr(cli_mod, "_ensure_cuda_libs_on_path", lambda: None)
    monkeypatch.setattr(cli_mod, "_preload_cudnn_libs", lambda: None)

    cli_mod.run_transcribe(
        input_path=str(fake_input),
        backend="faster",
        model_name="tiny",
        compute_type="int8",
        batch_size=4,
        output_dir=str(tmp_path / "outputs"),
        hf_cache_root=None,
        speaker_bank_root=str(tmp_path / "speaker_bank_root"),
        write_srt=False,
        write_jsonl=False,
        auto_batch=False,
        speaker_bank_config=SpeakerBankConfig(
            enabled=True,
            use_existing=True,
            emit_pca=False,
            match_per_segment=True,
            min_segments_per_label=1,
            threshold=0.5,
            scoring_margin=0.1,
        ),
    )


def test_run_transcribe_faster_writes_exclusive_diarization(monkeypatch, tmp_path):
    from transcriber import cli as cli_mod
    from transcriber.transcript_pipeline import TranscriptPipelineResult

    fake_input = tmp_path / "session.m4a"
    fake_input.write_text("dummy")

    def fake_gather_inputs(path: str) -> Tuple[List[str], None]:
        assert path == str(fake_input)
        return ([str(fake_input)], None)

    def fake_pipeline(*args: Any, **kwargs: Any) -> TranscriptPipelineResult:
        return TranscriptPipelineResult(
            segments=[
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello world",
                    "speaker": "SPEAKER_00",
                    "speaker_raw": "SPEAKER_00",
                    "words": [
                        {
                            "word": "hello",
                            "start": 0.0,
                            "end": 0.4,
                            "speaker": "SPEAKER_00",
                            "speaker_raw": "SPEAKER_00",
                        }
                    ],
                }
            ],
            diarization_segments=[{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
            exclusive_diarization_segments=[
                {"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                {"start": 0.5, "end": 1.0, "speaker": "SPEAKER_01"},
            ],
            speaker_embeddings={},
            metadata={},
        )

    def fake_save_outputs(
        *,
        base_stem: str,
        output_dir: str,
        per_file_segments: List[Tuple[str, List[Dict[str, Any]]]],
        consolidated_pairs: List[Tuple[str, str, str]],
        diar_by_file: Dict[str, List[Dict[str, Any]]] | None,
        exclusive_diar_by_file: Dict[str, List[Dict[str, Any]]] | None,
        write_srt_file: bool,
        write_jsonl_file: bool,
    ) -> Path:
        assert base_stem == "session"
        assert per_file_segments[0][1][0]["speaker_raw"] == "SPEAKER_00"
        assert diar_by_file is not None
        assert exclusive_diar_by_file is not None
        assert len(exclusive_diar_by_file[per_file_segments[0][0]]) == 2
        return Path(output_dir)

    monkeypatch.setattr(cli_mod, "gather_inputs", fake_gather_inputs)
    monkeypatch.setattr(cli_mod, "save_outputs", fake_save_outputs)
    monkeypatch.setattr(cli_mod, "cleanup_tmp", lambda *_args: None)
    monkeypatch.setattr(
        "transcriber.transcript_pipeline.transcribe_with_faster_pipeline",
        fake_pipeline,
    )
    monkeypatch.setattr("transcriber.diarization._detect_device", lambda: "cpu")
    monkeypatch.setattr(cli_mod, "_ensure_cuda_libs_on_path", lambda: None)
    monkeypatch.setattr(cli_mod, "_preload_cudnn_libs", lambda: None)

    cli_mod.run_transcribe(
        input_path=str(fake_input),
        backend="faster",
        model_name="tiny",
        compute_type="int8",
        batch_size=2,
        output_dir=str(tmp_path / "outputs"),
        hf_cache_root=None,
        speaker_bank_root=None,
        write_srt=False,
        write_jsonl=False,
        auto_batch=False,
        speaker_bank_config=None,
    )


def test_run_transcribe_faster_applies_speaker_bank_with_backend_device(monkeypatch, tmp_path):
    from transcriber import cli as cli_mod
    from transcriber.speaker_bank import SpeakerBankConfig
    from transcriber.transcript_pipeline import TranscriptPipelineResult
    from transcriber.diarization import SegmentEmbeddingResult

    fake_input = tmp_path / "session.m4a"
    fake_input.write_text("dummy")
    seen_force_devices: List[str | None] = []

    def fake_gather_inputs(path: str) -> Tuple[List[str], None]:
        assert path == str(fake_input)
        return ([str(fake_input)], None)

    def fake_pipeline(*args: Any, **kwargs: Any) -> TranscriptPipelineResult:
        return TranscriptPipelineResult(
            segments=[
                {"start": 0.0, "end": 1.0, "text": "one", "speaker": "SPEAKER_00"},
                {"start": 1.0, "end": 2.0, "text": "two", "speaker": "SPEAKER_01"},
                {"start": 2.0, "end": 3.0, "text": "three", "speaker": "SPEAKER_00"},
                {"start": 3.0, "end": 4.0, "text": "four", "speaker": "SPEAKER_01"},
            ],
            diarization_segments=[{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}],
            exclusive_diarization_segments=[],
            speaker_embeddings={
                "SPEAKER_00": np.array([1.0, 0.0], dtype=np.float32),
                "SPEAKER_01": np.array([0.0, 1.0], dtype=np.float32),
            },
            metadata={},
        )

    def fake_extract_embeddings_for_segments(
        audio_path: str,
        segments: List[Tuple[float, float, str]],
        hf_token: str | None,
        **kwargs: Any,
    ):
        seen_force_devices.append(kwargs.get("force_device"))
        results = [
            SegmentEmbeddingResult(
                speaker=label,
                start=float(start),
                end=float(end),
                index=index,
                embedding=(
                    np.array([1.0, 0.0], dtype=np.float32)
                    if label == "SPEAKER_00"
                    else np.array([0.0, 1.0], dtype=np.float32)
                ),
            )
            for index, (start, end, label) in enumerate(segments)
        ]
        return results, {"embedded": len(results), "skipped": 0, "total": len(results)}

    class FakeSpeakerBank:
        def __init__(self, *args: Any, **kwargs: Any):
            self.is_empty = False

        def summary(self) -> Dict[str, Any]:
            return {"profile": "default", "speakers": ["Alice", "Bob"], "entries": 2}

        def score_candidates(self, embedding: np.ndarray, **kwargs: Any):
            if float(embedding[0]) > float(embedding[1]):
                return [
                    {
                        "speaker": "Alice",
                        "cluster_id": "alice",
                        "score": 0.95,
                        "raw_score": 0.95,
                        "distance": 0.05,
                        "source": "prototype",
                    },
                    {
                        "speaker": "Bob",
                        "cluster_id": "bob",
                        "score": 0.15,
                        "raw_score": 0.15,
                        "distance": 0.85,
                        "source": "prototype",
                    },
                ]
            return [
                {
                    "speaker": "Bob",
                    "cluster_id": "bob",
                    "score": 0.96,
                    "raw_score": 0.96,
                    "distance": 0.04,
                    "source": "prototype",
                },
                {
                    "speaker": "Alice",
                    "cluster_id": "alice",
                    "score": 0.14,
                    "raw_score": 0.14,
                    "distance": 0.86,
                    "source": "prototype",
                },
            ]

        def match(self, embedding: np.ndarray, **kwargs: Any) -> Dict[str, Any] | None:
            candidate = self.score_candidates(embedding)[0]
            return {**candidate, "margin": candidate["score"], "second_best": 0.0}

    def fake_save_outputs(
        *,
        base_stem: str,
        output_dir: str,
        per_file_segments: List[Tuple[str, List[Dict[str, Any]]]],
        consolidated_pairs: List[Tuple[str, str, str]],
        diar_by_file: Dict[str, List[Dict[str, Any]]] | None,
        exclusive_diar_by_file: Dict[str, List[Dict[str, Any]]] | None,
        write_srt_file: bool,
        write_jsonl_file: bool,
    ) -> Path:
        speakers = [segment["speaker"] for _, segments in per_file_segments for segment in segments]
        assert speakers == ["Alice", "Bob", "Alice", "Bob"]
        return Path(output_dir)

    monkeypatch.setattr(cli_mod, "gather_inputs", fake_gather_inputs)
    monkeypatch.setattr(cli_mod, "save_outputs", fake_save_outputs)
    monkeypatch.setattr(cli_mod, "cleanup_tmp", lambda *_args: None)
    monkeypatch.setattr(cli_mod, "SpeakerBank", FakeSpeakerBank)
    monkeypatch.setattr(
        "transcriber.transcript_pipeline.transcribe_with_faster_pipeline",
        fake_pipeline,
    )
    monkeypatch.setattr(
        "transcriber.diarization.extract_embeddings_for_segments",
        fake_extract_embeddings_for_segments,
    )
    monkeypatch.setattr("transcriber.diarization._detect_device", lambda: "cpu")
    monkeypatch.setattr(cli_mod, "_ensure_cuda_libs_on_path", lambda: None)
    monkeypatch.setattr(cli_mod, "_preload_cudnn_libs", lambda: None)

    cli_mod.run_transcribe(
        input_path=str(fake_input),
        backend="faster",
        model_name="tiny",
        compute_type="int8",
        batch_size=2,
        output_dir=str(tmp_path / "outputs"),
        hf_cache_root=None,
        speaker_bank_root=str(tmp_path / "speaker_bank_root"),
        write_srt=False,
        write_jsonl=False,
        auto_batch=False,
        speaker_bank_config=SpeakerBankConfig(
            enabled=True,
            use_existing=True,
            emit_pca=False,
            match_per_segment=True,
            min_segments_per_label=1,
            threshold=0.5,
            scoring_margin=0.0,
        ),
    )

    assert seen_force_devices == ["cpu", "cpu"]


def test_run_transcribe_train_from_stems_persists_speaker_bank(monkeypatch, tmp_path):
    from transcriber import cli as cli_mod
    from transcriber.speaker_bank import SpeakerBankConfig
    from transcriber.transcript_pipeline import TranscriptPipelineResult

    fake_input = tmp_path / "session.zip"
    fake_input.write_text("dummy")
    persisted: Dict[str, Any] = {"entries": [], "save_calls": 0}

    def fake_gather_inputs(path: str) -> Tuple[List[str], str]:
        assert path == str(fake_input)
        return (
            [str(tmp_path / "track1.wav"), str(tmp_path / "track2.wav")],
            str(tmp_path / "unzipped"),
        )

    def fake_pipeline(audio_path: str, *args: Any, **kwargs: Any) -> TranscriptPipelineResult:
        stem = Path(audio_path).stem
        vector = (
            np.array([1.0, 0.0], dtype=np.float32)
            if stem == "track1"
            else np.array([0.0, 1.0], dtype=np.float32)
        )
        return TranscriptPipelineResult(
            segments=[{"start": 0.0, "end": 1.0, "text": stem, "speaker": None}],
            diarization_segments=[],
            exclusive_diarization_segments=[],
            speaker_embeddings={f"{stem}_speaker": vector},
            metadata={},
        )

    class FakeSpeakerBank:
        def __init__(self, *args: Any, **kwargs: Any):
            self.is_empty = False

        def summary(self) -> Dict[str, Any]:
            return {"profile": "default", "speakers": [], "entries": len(persisted["entries"])}

        def extend(self, entries):
            persisted["entries"].extend(list(entries))

        def save(self) -> None:
            persisted["save_calls"] += 1

        def render_pca(self, output_path: Path):
            return None

    def fake_save_outputs(**kwargs: Any) -> Path:
        out_dir = tmp_path / "outputs" / "session"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    monkeypatch.setattr(cli_mod, "gather_inputs", fake_gather_inputs)
    monkeypatch.setattr(cli_mod, "save_outputs", fake_save_outputs)
    monkeypatch.setattr(cli_mod, "cleanup_tmp", lambda *_args: None)
    monkeypatch.setattr(cli_mod, "SpeakerBank", FakeSpeakerBank)
    monkeypatch.setattr(cli_mod, "load_segment_classifier", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "transcriber.transcript_pipeline.transcribe_with_faster_pipeline",
        fake_pipeline,
    )
    monkeypatch.setattr("transcriber.diarization._detect_device", lambda: "cpu")
    monkeypatch.setattr(cli_mod, "_ensure_cuda_libs_on_path", lambda: None)
    monkeypatch.setattr(cli_mod, "_preload_cudnn_libs", lambda: None)

    cli_mod.run_transcribe(
        input_path=str(fake_input),
        backend="faster",
        model_name="tiny",
        compute_type="int8",
        batch_size=2,
        output_dir=str(tmp_path / "outputs"),
        hf_cache_root=None,
        speaker_bank_root=str(tmp_path / "speaker_bank_root"),
        write_srt=False,
        write_jsonl=False,
        auto_batch=False,
        speaker_bank_config=SpeakerBankConfig(
            enabled=True,
            use_existing=False,
            train_from_stems=True,
            emit_pca=False,
        ),
    )

    assert persisted["save_calls"] == 1
    assert [entry[0] for entry in persisted["entries"]] == ["track1", "track2"]


def test_run_transcribe_multi_file_mapping_preserves_bank_labels(monkeypatch, tmp_path):
    from transcriber import cli as cli_mod
    from transcriber.transcript_pipeline import TranscriptPipelineResult

    track1 = tmp_path / "track1.wav"
    track2 = tmp_path / "track2.wav"
    track1.write_text("dummy")
    track2.write_text("dummy")
    captured: Dict[str, Any] = {}

    def fake_gather_inputs(path: str) -> Tuple[List[str], None]:
        assert path == str(tmp_path)
        return ([str(track1), str(track2)], None)

    def fake_load_yaml_or_json(path: str | None) -> Dict[str, str]:
        if path == "mapping.yaml":
            return {"track1": "Mapped One", "track2": "Mapped Two"}
        return {}

    def fake_pipeline(audio_path: str, *args: Any, **kwargs: Any) -> TranscriptPipelineResult:
        speaker = "Bank Alice" if Path(audio_path).stem == "track1" else "Bank Bob"
        source = "segment_classifier" if speaker == "Bank Alice" else "label_vector"
        return TranscriptPipelineResult(
            segments=[
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "speaker": speaker,
                    "speaker_match_source": source,
                }
            ],
            diarization_segments=[],
            exclusive_diarization_segments=[],
            speaker_embeddings={},
            metadata={},
        )

    def fake_save_outputs(**kwargs: Any) -> Path:
        captured["per_file_segments"] = kwargs["per_file_segments"]
        out_dir = tmp_path / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    monkeypatch.setattr(cli_mod, "gather_inputs", fake_gather_inputs)
    monkeypatch.setattr(cli_mod, "_load_yaml_or_json", fake_load_yaml_or_json)
    monkeypatch.setattr(cli_mod, "save_outputs", fake_save_outputs)
    monkeypatch.setattr(cli_mod, "cleanup_tmp", lambda *_args: None)
    monkeypatch.setattr(
        "transcriber.transcript_pipeline.transcribe_with_faster_pipeline",
        fake_pipeline,
    )
    monkeypatch.setattr("transcriber.diarization._detect_device", lambda: "cpu")
    monkeypatch.setattr(cli_mod, "_ensure_cuda_libs_on_path", lambda: None)
    monkeypatch.setattr(cli_mod, "_preload_cudnn_libs", lambda: None)

    cli_mod.run_transcribe(
        input_path=str(tmp_path),
        backend="faster",
        model_name="tiny",
        compute_type="int8",
        batch_size=2,
        output_dir=str(tmp_path / "outputs"),
        speaker_mapping_path="mapping.yaml",
        hf_cache_root=None,
        speaker_bank_root=None,
        write_srt=False,
        write_jsonl=False,
        auto_batch=False,
        speaker_bank_config=None,
    )

    speakers = [segments[0]["speaker"] for _, segments in captured["per_file_segments"]]
    assert speakers == ["Bank Alice", "Bank Bob"]
