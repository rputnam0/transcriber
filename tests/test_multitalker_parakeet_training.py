from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_multitalker_parakeet_dataset import (  # noqa: E402
    group_speaker_words,
    resolve_audio_path,
)
from train_multitalker_parakeet_adapter import (  # noqa: E402
    build_training_mapping,
    configure_trainable_parameters,
)


def test_group_speaker_words_preserves_overlap_between_speakers() -> None:
    words = [
        {"speaker": "Alice", "start": 0.2, "end": 0.5, "text": "hello"},
        {"speaker": "Bob", "start": 0.4, "end": 0.7, "text": "yes"},
        {"speaker": "Alice", "start": 0.6, "end": 0.9, "text": "there"},
    ]

    spans = group_speaker_words(
        words,
        max_gap_seconds=0.2,
        collar_seconds=0.1,
        clip_duration=2.0,
    )

    assert [span["speaker"] for span in spans] == ["Alice", "Bob"]
    assert [span["text"] for span in spans] == ["hello there", "yes"]
    assert [span["start"] for span in spans] == pytest.approx([0.1, 0.3])
    assert [span["end"] for span in spans] == pytest.approx([1.0, 0.8])


def test_group_speaker_words_preserves_source_utterance_order() -> None:
    words = [
        {
            "speaker": "Alice",
            "start": 2.7,
            "end": 3.2,
            "text": "fifteen",
            "source_span_start": 151.0,
            "source_span_end": 153.0,
        },
        {
            "speaker": "Alice",
            "start": 2.8,
            "end": 3.0,
            "text": "it",
            "source_span_start": 153.0,
            "source_span_end": 155.0,
        },
        {
            "speaker": "Alice",
            "start": 3.2,
            "end": 3.5,
            "text": "last",
            "source_span_start": 151.0,
            "source_span_end": 153.0,
        },
        {
            "speaker": "Alice",
            "start": 3.2,
            "end": 3.4,
            "text": "was",
            "source_span_start": 153.0,
            "source_span_end": 155.0,
        },
    ]

    spans = group_speaker_words(
        words,
        max_gap_seconds=0.2,
        collar_seconds=0.0,
        clip_duration=30.0,
        clip_offset=150.0,
    )

    assert [span["text"] for span in spans] == ["fifteen last", "it was"]
    assert [span["start"] for span in spans] == pytest.approx([1.0, 3.0])
    assert [span["end"] for span in spans] == pytest.approx([3.0, 5.0])


def test_resolve_audio_path_uses_manifest_materialized_audio(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "outputs" / "dataset"
    audio_path = dataset_dir / "audio" / "clip.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.touch()
    dataset_jsonl = dataset_dir / "dataset.jsonl"

    resolved = resolve_audio_path(
        "outputs/dataset/audio/clip.wav",
        dataset_jsonl=dataset_jsonl,
        audio_root=None,
    )

    assert resolved == audio_path.resolve()


def test_configure_trainable_parameters_selects_kernels_and_decoder() -> None:
    import torch

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.spk_kernels = torch.nn.Linear(2, 2)
            self.bg_spk_kernels = torch.nn.Linear(2, 2)
            self.decoder = torch.nn.Linear(2, 2)
            self.joint = torch.nn.Linear(2, 2)

    model = Model()
    summary = configure_trainable_parameters(model, "kernels-decoder")

    assert not model.encoder.weight.requires_grad
    assert model.spk_kernels.weight.requires_grad
    assert model.bg_spk_kernels.weight.requires_grad
    assert model.decoder.weight.requires_grad
    assert model.joint.weight.requires_grad
    assert summary["trainable_parameters"] < summary["total_parameters"]


def test_configure_trainable_parameters_can_select_decoder_and_encoder_tail() -> None:
    import torch

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
            self.decoder = torch.nn.Linear(2, 2)
            self.joint = torch.nn.Linear(2, 2)
            self.spk_kernels = torch.nn.Linear(2, 2)

    model = TinyModel()
    summary = configure_trainable_parameters(model, "asr-tail-2")

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {
        "encoder.layers.2.weight",
        "encoder.layers.2.bias",
        "encoder.layers.3.weight",
        "encoder.layers.3.bias",
        "decoder.weight",
        "decoder.bias",
        "joint.weight",
        "joint.bias",
    }
    assert summary["encoder_tail_layers"] == 2


def test_build_training_config_uses_native_lhotse_cuts(tmp_path: Path) -> None:
    cuts_path = tmp_path / "cuts.jsonl.gz"

    config = build_training_mapping(
        {"manifest_filepath": "legacy.jsonl", "batch_duration": 1200},
        cuts_path=cuts_path,
        batch_size=1,
        num_workers=0,
        max_duration=31.0,
    )

    assert config["manifest_filepath"] is None
    assert config["cuts_path"] == str(cuts_path.resolve())
    assert "batch_duration" not in config
