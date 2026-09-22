from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import torch
from torch import nn

from train_ps4_activity_word_owner import (
    _evaluate_full_groups,
    _group_rows,
    _sha256,
)
from train_sequence_tsvad_word_owner_baseline import _summarize_groups
from train_tsvad_word_owner_baseline import _write_jsonl
from train_usef_tse_domain_adapter import ForcedReferenceIndex, _load_rows


DEFAULT_REVISION = "eefb734ba3dd10bd3566ecc578db1a5cf7e1c83d"
DEFAULT_CHECKPOINT = "checkpoints/V3/OptimizerStep960.pth"
MODEL_ARGS = {
    "hidden_channels": 256,
    "n_head": 4,
    "emb_dim": 128,
    "emb_ks": 1,
    "emb_hs": 1,
    "num_layers": 6,
}


class StreamingUsefTpActivity(nn.Module):
    """Expose the pretrained USEF-TP personal-VAD logits as an activity model."""

    frame_hop_seconds = 0.008

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, mixture: torch.Tensor, enrollment: torch.Tensor) -> torch.Tensor:
        required = ("stft", "encoder", "cmha", "separator", "pvad_decoder")
        if all(hasattr(self.model, name) for name in required):
            mix_spectrum = self.model.stft(mixture.unsqueeze(1))[-1]
            enrollment_spectrum = self.model.stft(enrollment.unsqueeze(1))[-1]
            mix_features = (
                torch.cat([mix_spectrum.real, mix_spectrum.imag], dim=1)
                .permute(0, 1, 3, 2)
                .contiguous()
            )
            enrollment_features = (
                torch.cat([enrollment_spectrum.real, enrollment_spectrum.imag], dim=1)
                .permute(0, 1, 3, 2)
                .contiguous()
            )
            encoded_mix = self.model.encoder(mix_features)
            encoded_enrollment = self.model.encoder(enrollment_features)
            speaker_features = self.model.cmha(encoded_mix, encoded_enrollment)
            separated = self.model.separator(torch.cat([encoded_mix, speaker_features], dim=1))
            return self.model.pvad_decoder(separated).squeeze(1)
        _, logits = self.model(mixture, enrollment)
        return logits.squeeze(1)

    def set_trainable_mode(self, training: bool) -> None:
        self.model.train(training)


class IdentityHead(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits


def load_model(
    *, source_root: Path, checkpoint_path: Path, device: torch.device
) -> tuple[StreamingUsefTpActivity, dict[str, object]]:
    sys.path.insert(0, str(source_root.resolve()))
    try:
        model_class = importlib.import_module("model_streaming_usef_tp").Streaming_USEF_TP
    finally:
        sys.path.pop(0)
    model = model_class(**MODEL_ARGS)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state: Mapping[str, torch.Tensor] = checkpoint.get("model", checkpoint)
    response = model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return StreamingUsefTpActivity(model), {
        "missing_keys": list(response.missing_keys),
        "unexpected_keys": list(response.unexpected_keys),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a pretrained streaming USEF-TP personal-VAD on known speakers."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--forced-reference-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--sessions")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-full-eval-groups", type=int, default=0)
    parser.add_argument("--inference-chunk-seconds", type=float, default=15.0)
    parser.add_argument("--enrollment-seconds", type=float, default=10.0)
    parser.add_argument("--candidate-batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path or args.source_root / DEFAULT_CHECKPOINT
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    sessions = {value.strip() for value in str(args.sessions or "").split(",") if value.strip()}
    rows = _load_rows(
        args.manifest,
        {str(args.split)},
        sessions=sessions or None,
        max_rows=args.max_rows,
    )
    groups = _group_rows(rows)
    if not groups:
        raise RuntimeError(f"No {args.split} groups were found")

    device = torch.device(args.device)
    extractor, load_summary = load_model(
        source_root=args.source_root,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    references = ForcedReferenceIndex(args.forced_reference_jsonl)
    eval_args = SimpleNamespace(
        max_full_eval_groups=int(args.max_full_eval_groups),
        inference_chunk_seconds=float(args.inference_chunk_seconds),
        enrollment_seconds=float(args.enrollment_seconds),
        sample_rate=8000,
        inference_candidate_batch_size=int(args.candidate_batch_size),
    )
    group_results, word_records = _evaluate_full_groups(
        groups,
        references=references,
        manifest_dir=args.manifest.resolve().parent,
        extractor=extractor,
        head=IdentityHead().to(device),
        device=device,
        args=eval_args,
    )
    summary = {
        "model": "streaming-usef-tp-personal-vad",
        "source_repo": "VMoorjani/Streaming-USEF-TP",
        "source_revision": DEFAULT_REVISION,
        "license": None,
        "license_status": "undeclared-research-feasibility-only",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "model_args": MODEL_ARGS,
        "load": load_summary,
        "split": str(args.split),
        "groups": len(group_results),
        "sample_rate": 8000,
        "frame_hop_seconds": extractor.frame_hop_seconds,
        "result": _summarize_groups(group_results),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "streaming_usef_tp_groups.jsonl", group_results)
    _write_jsonl(args.output_dir / "streaming_usef_tp_words.jsonl", word_records)
    (args.output_dir / "streaming_usef_tp_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
