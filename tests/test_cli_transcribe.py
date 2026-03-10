from __future__ import annotations

import sys
from pathlib import Path
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

    def fake_transcribe_with_whisperx(*args: Any, **kwargs: Any):
        segments = _fake_segments_payload()
        diar = {"segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]}
        speaker_embeddings: Dict[str, Any] = {}
        return segments, diar, speaker_embeddings

    monkeypatch.setattr(cli_mod, "gather_inputs", fake_gather_inputs)
    monkeypatch.setattr(cli_mod, "save_outputs", fake_save_outputs)
    monkeypatch.setattr(cli_mod, "cleanup_tmp", lambda *_args: None)
    monkeypatch.setattr("transcriber.whisperx_backend.transcribe_with_whisperx", fake_transcribe_with_whisperx)
    monkeypatch.setattr("transcriber.whisperx_backend._detect_device", lambda: "cpu")
    monkeypatch.setattr(cli_mod, "_ensure_cuda_libs_on_path", lambda: None)
    monkeypatch.setattr(cli_mod, "_preload_cudnn_libs", lambda: None)

    cli_mod.run_transcribe(
        input_path=str(fake_input),
        backend="whisperx",
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
                {"speaker": "Cletus Cobbington", "score": 0.20, "cluster_id": "cl0", "distance": 0.8},
            ],
        },
        1: {
            "accepted": True,
            "match": {"speaker": "Dungeon Master", "score": 0.89},
            "candidates": [
                {"speaker": "Dungeon Master", "score": 0.89, "cluster_id": "dm0", "distance": 0.1},
                {"speaker": "Cletus Cobbington", "score": 0.22, "cluster_id": "cl0", "distance": 0.8},
            ],
        },
        2: {
            "accepted": False,
            "match": None,
            "candidates": [
                {"speaker": "Cletus Cobbington", "score": 0.59, "cluster_id": "cl1", "distance": 0.2},
                {"speaker": "Dungeon Master", "score": 0.10, "cluster_id": "dm0", "distance": 0.9},
            ],
        },
        3: {
            "accepted": False,
            "match": None,
            "candidates": [
                {"speaker": "Cletus Cobbington", "score": 0.60, "cluster_id": "cl1", "distance": 0.2},
                {"speaker": "Dungeon Master", "score": 0.11, "cluster_id": "dm0", "distance": 0.9},
            ],
        },
        4: {
            "accepted": False,
            "match": None,
            "candidates": [
                {"speaker": "Cletus Cobbington", "score": 0.58, "cluster_id": "cl1", "distance": 0.2},
                {"speaker": "Dungeon Master", "score": 0.09, "cluster_id": "dm0", "distance": 0.9},
            ],
        },
        5: {
            "accepted": False,
            "match": None,
            "candidates": [
                {"speaker": "Cletus Cobbington", "score": 0.61, "cluster_id": "cl1", "distance": 0.2},
                {"speaker": "Dungeon Master", "score": 0.10, "cluster_id": "dm0", "distance": 0.9},
            ],
        },
    }

    selection, stats = _aggregate_segment_label_candidates(
        [0, 1, 2, 3, 4, 5],
        seg_matches,
        aggregation="mean",
        margin_required=0.03,
        min_segments_per_label=3,
    )

    assert selection is not None
    assert selection["speaker"] == "Cletus Cobbington"
    assert selection["score_max"] == pytest.approx(0.61)
    assert stats["segments_embedded"] == 6
    assert stats["segments_matched"] == 2
    assert stats["means"]["Cletus Cobbington"] > stats["means"]["Dungeon Master"]


def test_run_transcribe_extracts_segment_embeddings_per_raw_label(monkeypatch, tmp_path):
    from transcriber import cli as cli_mod
    from transcriber.speaker_bank import SpeakerBankConfig
    from transcriber.whisperx_backend import SegmentEmbeddingResult

    fake_input = tmp_path / "input.m4a"
    fake_input.write_text("dummy")
    seen_label_sets: List[set[str]] = []

    def fake_gather_inputs(path: str) -> Tuple[List[str], None]:
        assert path == str(fake_input)
        return ([str(fake_input)], None)

    def fake_transcribe_with_whisperx(*args: Any, **kwargs: Any):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "first", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "second", "speaker": "SPEAKER_01"},
            {"start": 2.0, "end": 3.0, "text": "third", "speaker": "SPEAKER_00"},
            {"start": 3.0, "end": 4.0, "text": "fourth", "speaker": "SPEAKER_01"},
        ]
        diar = {"segments": [dict(item) for item in segments]}
        speaker_embeddings = {
            "SPEAKER_00": np.array([1.0, 0.0], dtype=np.float32),
            "SPEAKER_01": np.array([0.0, 1.0], dtype=np.float32),
        }
        return segments, diar, speaker_embeddings

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
            return {"profile": "default", "speakers": ["Alice", "Bob"], "entries": 2, "clusters": {}}

        def score_candidates(self, embedding: np.ndarray, *, radius_factor: float = 2.5):
            if float(embedding[0]) > float(embedding[1]):
                return [
                    {
                        "speaker": "Alice",
                        "cluster_id": "alice",
                        "score": 0.95,
                        "distance": 0.05,
                        "source": "prototype",
                    },
                    {
                        "speaker": "Bob",
                        "cluster_id": "bob",
                        "score": 0.20,
                        "distance": 0.80,
                        "source": "prototype",
                    },
                ]
            return [
                {
                    "speaker": "Bob",
                    "cluster_id": "bob",
                    "score": 0.96,
                    "distance": 0.04,
                    "source": "prototype",
                },
                {
                    "speaker": "Alice",
                    "cluster_id": "alice",
                    "score": 0.22,
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
        ) -> Dict[str, Any] | None:
            candidates = self.score_candidates(embedding, radius_factor=radius_factor)
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
    monkeypatch.setattr("transcriber.whisperx_backend.transcribe_with_whisperx", fake_transcribe_with_whisperx)
    monkeypatch.setattr(
        "transcriber.whisperx_backend.extract_embeddings_for_segments",
        fake_extract_embeddings_for_segments,
    )
    monkeypatch.setattr("transcriber.whisperx_backend._detect_device", lambda: "cpu")
    monkeypatch.setattr(cli_mod, "_ensure_cuda_libs_on_path", lambda: None)
    monkeypatch.setattr(cli_mod, "_preload_cudnn_libs", lambda: None)

    cli_mod.run_transcribe(
        input_path=str(fake_input),
        backend="whisperx",
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

    assert seen_label_sets == [{"SPEAKER_00"}, {"SPEAKER_01"}]
