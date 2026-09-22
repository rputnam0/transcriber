from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf

from score_pyannote_oracle_diarization import _score_group, _summarize
from train_tsvad_word_owner_baseline import (
    _group_manifest_rows,
    _load_reference_groups,
    _materialized_path,
    _read_jsonl,
    _split_values,
    _write_jsonl,
)


DEFAULT_SORTFORMER_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    wave = np.asarray(audio, dtype=np.float32)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    return np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0), int(sample_rate)


def _parse_sortformer_segments(lines: Sequence[object]) -> list[dict]:
    turns = []
    for raw in lines:
        if isinstance(raw, Mapping):
            start = raw.get("start")
            end = raw.get("end")
            speaker = raw.get("speaker")
        else:
            parts = str(raw).split()
            if len(parts) < 3:
                continue
            start, end, speaker = parts[:3]
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if end_f <= start_f:
            continue
        turns.append(
            {
                "start": start_f,
                "end": end_f,
                "speaker": str(speaker or "unknown"),
            }
        )
    turns.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    return turns


def _load_sortformer_model(
    *,
    model_name: str,
    restore_path: Path | None,
    device: str,
):
    from nemo.collections.asr.models import SortformerEncLabelModel

    if restore_path:
        model = SortformerEncLabelModel.restore_from(
            restore_path=str(restore_path),
            map_location=device,
            strict=False,
        )
    else:
        model = SortformerEncLabelModel.from_pretrained(model_name, map_location=device)
    model.eval()
    try:
        model.to(device)
    except Exception:
        pass
    return model


def _apply_streaming_config(
    model,
    *,
    chunk_len: int | None,
    right_context: int | None,
    fifo_len: int | None,
    spkcache_update_period: int | None,
    spkcache_len: int | None,
) -> dict:
    modules = getattr(model, "sortformer_modules", None)
    if modules is None:
        return {}
    applied = {}
    for attr, value in (
        ("chunk_len", chunk_len),
        ("chunk_right_context", right_context),
        ("fifo_len", fifo_len),
        ("spkcache_update_period", spkcache_update_period),
        ("spkcache_len", spkcache_len),
    ):
        if value is not None and hasattr(modules, attr):
            setattr(modules, attr, int(value))
            applied[attr] = int(value)
    checker = getattr(modules, "_check_streaming_parameters", None)
    if callable(checker):
        checker()
    return applied


def _run_sortformer(model, wave: np.ndarray, *, sample_rate: int, batch_size: int) -> list[dict]:
    predicted = model.diarize(
        audio=[wave],
        sample_rate=int(sample_rate),
        batch_size=int(batch_size),
        num_workers=0,
        verbose=False,
    )
    if not predicted:
        return []
    return _parse_sortformer_segments(predicted[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score NeMo Sortformer diarization with oracle cluster-to-speaker mapping."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-splits", default="test")
    parser.add_argument("--eval-sessions")
    parser.add_argument("--model-name", default=DEFAULT_SORTFORMER_MODEL)
    parser.add_argument("--restore-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-len", type=int, default=340)
    parser.add_argument("--right-context", type=int, default=40)
    parser.add_argument("--fifo-len", type=int, default=40)
    parser.add_argument("--spkcache-update-period", type=int, default=340)
    parser.add_argument("--spkcache-len", type=int, default=188)
    parser.add_argument("--save-word-records", action="store_true")
    args = parser.parse_args()

    manifest_dir = args.manifest.resolve().parent
    reference_groups = _load_reference_groups(args.reference_jsonl)
    grouped_rows = _group_manifest_rows(_read_jsonl(args.manifest))
    eval_splits = _split_values(args.eval_splits)
    eval_sessions = _split_values(args.eval_sessions or "")
    model = _load_sortformer_model(
        model_name=str(args.model_name),
        restore_path=args.restore_path,
        device=str(args.device),
    )
    streaming_config = _apply_streaming_config(
        model,
        chunk_len=args.chunk_len,
        right_context=args.right_context,
        fifo_len=args.fifo_len,
        spkcache_update_period=args.spkcache_update_period,
        spkcache_len=args.spkcache_len,
    )

    group_results = []
    word_records = []
    for key, all_rows in sorted(grouped_rows.items()):
        rows = [
            row
            for row in all_rows
            if str(row.get("split_id") or "") in eval_splits
            and (not eval_sessions or str(row.get("session") or "") in eval_sessions)
            and row.get("materialized")
        ]
        if not rows:
            continue
        if key not in reference_groups:
            group_results.append(
                {
                    "session": key[0],
                    "window_start": key[1],
                    "window_end": key[2],
                    "error": "missing_reference_group",
                }
            )
            continue
        mixture_path = _materialized_path(rows[0], "mixture_path", manifest_dir=manifest_dir)
        if mixture_path is None or not mixture_path.exists():
            group_results.append(
                {
                    "session": key[0],
                    "window_start": key[1],
                    "window_end": key[2],
                    "error": "missing_mixture_path",
                }
            )
            continue
        wave, sample_rate = _load_mono(mixture_path)
        turns = _run_sortformer(model, wave, sample_rate=sample_rate, batch_size=args.batch_size)
        result = _score_group(
            key=key,
            rows=rows,
            words=reference_groups[key],
            turns=turns,
        )
        result["mixture_path"] = str(mixture_path)
        result["model_turn_count"] = len(turns)
        result["model_cluster_count"] = len({turn["speaker"] for turn in turns})
        group_records = result.pop("word_records")
        group_results.append(result)
        if args.save_word_records:
            for record in group_records:
                record.update(
                    {
                        "session": key[0],
                        "window_start": key[1],
                        "window_end": key[2],
                    }
                )
                word_records.append(record)

    summary = _summarize(group_results)
    summary.update(
        {
            "manifest": str(args.manifest),
            "reference_jsonl": str(args.reference_jsonl),
            "eval_splits": sorted(eval_splits),
            "eval_sessions": sorted(eval_sessions),
            "model_name": str(args.model_name),
            "restore_path": str(args.restore_path) if args.restore_path else None,
            "device": str(args.device),
            "batch_size": int(args.batch_size),
            "streaming_config": streaming_config,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "sortformer_oracle_diarization_groups.jsonl", group_results)
    if word_records:
        _write_jsonl(args.output_dir / "sortformer_oracle_diarization_words.jsonl", word_records)
    (args.output_dir / "sortformer_oracle_diarization_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
