from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

from build_sortformer_mask_cutset import match_probability_speakers
from score_sortformer_oracle_diarization import (
    DEFAULT_SORTFORMER_MODEL,
    _apply_streaming_config,
    _load_sortformer_model,
)


def _run_key(cut_id: object) -> tuple[int, int]:
    match = re.match(r"session_(\d+)_w(\d+)_c(\d+)", str(cut_id or ""))
    if not match:
        raise ValueError(f"Cannot infer contiguous run from cut id {cut_id!r}")
    return int(match.group(1)), int(match.group(2))


def _cut_offset(cut_id: object) -> int:
    match = re.match(r"session_\d+_w\d+_c(\d+)", str(cut_id or ""))
    if not match:
        raise ValueError(f"Cannot infer cut offset from {cut_id!r}")
    return int(match.group(1))


def group_contiguous_cuts(cuts: Sequence[object]) -> list[list[object]]:
    grouped: dict[tuple[int, int], list[object]] = defaultdict(list)
    for cut in cuts:
        grouped[_run_key(cut.id)].append(cut)
    runs = []
    for key in sorted(grouped):
        ordered = sorted(grouped[key], key=lambda cut: _cut_offset(cut.id))
        current = []
        for cut in ordered:
            expected_offset = (
                _cut_offset(current[-1].id) + int(round(float(current[-1].duration) * 1000))
                if current
                else None
            )
            if current and _cut_offset(cut.id) != expected_offset:
                runs.append(current)
                current = []
            current.append(cut)
        if current:
            runs.append(current)
    return runs


def split_probabilities(
    probabilities: np.ndarray,
    sample_counts: Sequence[int],
) -> list[np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"Expected [frames, speakers] probabilities, got {values.shape}")
    total_samples = sum(sample_counts)
    if total_samples <= 0:
        raise ValueError("Cannot split probabilities across empty audio")
    cumulative = np.cumsum([0, *sample_counts], dtype=np.float64)
    boundaries = np.rint(cumulative / total_samples * values.shape[0]).astype(int)
    boundaries[0] = 0
    boundaries[-1] = values.shape[0]
    return [values[first:last] for first, last in zip(boundaries, boundaries[1:])]


def _load_cut_audio(cut: object) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(
        cut.recording.sources[0].source,
        start=int(round(cut.start * cut.sampling_rate)),
        frames=cut.num_samples,
        dtype="float32",
        always_2d=True,
    )
    return np.asarray(samples.mean(axis=1), dtype=np.float32), int(sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run streaming Sortformer across contiguous mono cuts without resetting slots."
    )
    parser.add_argument("--input-cuts", type=Path, required=True)
    parser.add_argument("--output-cuts", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_SORTFORMER_MODEL)
    parser.add_argument("--restore-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-len", type=int, default=340)
    parser.add_argument("--right-context", type=int, default=40)
    parser.add_argument("--fifo-len", type=int, default=40)
    parser.add_argument("--spkcache-update-period", type=int, default=340)
    parser.add_argument("--spkcache-len", type=int, default=188)
    args = parser.parse_args()

    from lhotse import CutSet
    from lhotse.utils import fastcopy

    model = _load_sortformer_model(
        model_name=args.model_name,
        restore_path=args.restore_path,
        device=args.device,
    )
    streaming_config = _apply_streaming_config(
        model,
        chunk_len=args.chunk_len,
        right_context=args.right_context,
        fifo_len=args.fifo_len,
        spkcache_update_period=args.spkcache_update_period,
        spkcache_len=args.spkcache_len,
    )
    runs = group_contiguous_cuts(list(CutSet.from_file(args.input_cuts)))
    probability_dir = args.output_cuts.parent / (
        args.output_cuts.name.removesuffix(".jsonl.gz") + "_probabilities"
    )
    probability_dir.mkdir(parents=True, exist_ok=True)
    output = []
    metrics = []
    run_summaries = []
    for run_index, run in enumerate(runs):
        waves = []
        sample_rate = None
        for cut in run:
            wave, cut_rate = _load_cut_audio(cut)
            if sample_rate is not None and cut_rate != sample_rate:
                raise ValueError("Contiguous cuts have inconsistent sample rates")
            sample_rate = cut_rate
            waves.append(wave)
        mixture = np.concatenate(waves)
        _segments, raw_probabilities = model.diarize(
            audio=[mixture],
            sample_rate=int(sample_rate),
            batch_size=args.batch_size,
            include_tensor_outputs=True,
            num_workers=0,
            verbose=False,
        )
        probabilities = raw_probabilities[0].squeeze(0).detach().cpu().numpy()
        cut_probabilities = split_probabilities(probabilities, [len(wave) for wave in waves])
        run_summaries.append(
            {
                "run_index": run_index,
                "first_cut_id": run[0].id,
                "last_cut_id": run[-1].id,
                "cut_count": len(run),
                "duration": len(mixture) / int(sample_rate),
                "probability_frames": int(probabilities.shape[0]),
            }
        )
        for cut, values in zip(run, cut_probabilities, strict=True):
            mapping, cut_metrics = match_probability_speakers(
                cut.supervisions,
                values,
                duration=float(cut.duration),
            )
            probability_path = probability_dir / f"{cut.id}.npy"
            np.save(probability_path, values, allow_pickle=False)
            custom = dict(cut.custom or {})
            custom.update(
                {
                    "activity_mask_source": "mono-sortformer-contiguous-soft",
                    "sortformer_probabilities_path": str(probability_path.resolve()),
                    "sortformer_slot_to_speaker": dict(mapping),
                    "sortformer_contiguous_run": run_index,
                    "oracle_slot_mapping_is_diagnostic_only": True,
                }
            )
            output.append(fastcopy(cut, id=f"{cut.id}-mask-sortformer", custom=custom))
            metrics.append(cut_metrics)
        print(f"processed contiguous run {run_index + 1}/{len(runs)}", flush=True)

    args.output_cuts.parent.mkdir(parents=True, exist_ok=True)
    CutSet.from_cuts(output).to_file(args.output_cuts)
    summary = {
        "input_cuts": str(args.input_cuts),
        "output_cuts": str(args.output_cuts),
        "output_cut_count": len(output),
        "contiguous_run_count": len(runs),
        "model_name": args.model_name,
        "restore_path": str(args.restore_path) if args.restore_path else None,
        "mean_reference_speaker_recall": float(
            np.mean([item["reference_speaker_recall"] for item in metrics])
        ),
        "mean_matched_f1": float(np.mean([item["mean_matched_f1"] for item in metrics])),
        "streaming_config": streaming_config,
        "probability_dir": str(probability_dir),
        "runs": run_summaries,
    }
    args.output_cuts.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
