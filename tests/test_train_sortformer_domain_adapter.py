from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from train_sortformer_domain_adapter import build_diarization_data_config  # noqa: E402


def test_build_diarization_data_config_overrides_training_shape(tmp_path):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("", encoding="utf-8")
    base = {
        "manifest_filepath": "old.jsonl",
        "sample_rate": 8000,
        "num_spks": 4,
        "session_len_sec": 180,
        "batch_size": 4,
        "num_workers": 16,
        "pin_memory": True,
        "shuffle": False,
        "use_lhotse": True,
        "use_bucketing": True,
        "soft_label_thres": 0.7,
        "soft_targets": True,
    }

    config = build_diarization_data_config(
        base,
        manifest_path=manifest,
        sample_rate=16000,
        num_speakers=8,
        session_len_seconds=30.0,
        batch_size=1,
        num_workers=2,
        shuffle=True,
    )

    assert config["manifest_filepath"] == str(manifest.resolve())
    assert config["sample_rate"] == 16000
    assert config["num_spks"] == 8
    assert config["session_len_sec"] == 30.0
    assert config["batch_size"] == 1
    assert config["num_workers"] == 2
    assert config["pin_memory"] is False
    assert config["shuffle"] is True
    assert config["use_lhotse"] is False
    assert config["use_bucketing"] is False
    assert config["soft_label_thres"] == 0.7
    assert config["soft_targets"] is True
