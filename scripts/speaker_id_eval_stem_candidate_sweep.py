from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (str(SCRIPT_ROOT), str(SRC_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from speaker_id_conditioned_tasnet_candidate_sweep import (  # noqa: E402
    _evaluate_candidate_embeddings,
    _load_candidate_embeddings,
)
from speaker_id_conditioned_tasnet_sweep import (  # noqa: E402
    ConditionedTasNetExtractor,
    _conditioning_vectors,
    _extract_batch,
    _slice_wave,
)
from speaker_id_eval_stem_target_extractor_sweep import _session_group  # noqa: E402
from speaker_id_learned_mask_sweep import CORE_SPEAKERS, _limit_rows_by_group  # noqa: E402
from speaker_id_oracle_mask_sweep import (  # noqa: E402
    MaskRow,
    _collect_training_items,
    _embed_waveforms,
    _load_audio,
    _load_clean_bank,
    _load_rows,
    _load_titanet,
    _load_titanet_word_npz,
)
from speaker_id_target_extractor_sweep import _mixed_same_rows  # noqa: E402
from speaker_id_word_window_sweep import _window_group  # noqa: E402


def _safe_group(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def _split_group(row: MaskRow, mode: str) -> str:
    if mode == "leave_group":
        return _window_group(row.window)
    if mode == "leave_session":
        return _session_group(row.window)
    raise ValueError(mode)


def _model_path_for_group(base_path: Path, group: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{_safe_group(group)}{base_path.suffix}")


def _load_fold_model(
    path: Path,
    *,
    embedding_dim: int,
    device: str,
    args: argparse.Namespace,
) -> ConditionedTasNetExtractor:
    payload = torch.load(path, map_location=device, weights_only=False)
    saved_args = dict(payload.get("args") or {})
    model = ConditionedTasNetExtractor(
        embedding_dim=embedding_dim,
        enc_feats=int(saved_args.get("enc_feats", args.enc_feats)),
        bottleneck=int(saved_args.get("bottleneck", args.bottleneck)),
        cond_dim=int(saved_args.get("cond_dim", args.cond_dim)),
        enc_kernel=int(saved_args.get("enc_kernel", args.enc_kernel)),
        layers=int(saved_args.get("layers", args.layers)),
        stacks=int(saved_args.get("stacks", args.stacks)),
        mask_activation=str(saved_args.get("mask_activation", "sigmoid")),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _ordered_rows(rows: Sequence[MaskRow]) -> Tuple[List[MaskRow], Dict[str, List[MaskRow]]]:
    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in rows:
        rows_by_window[row.window].append(row)
    ordered: List[MaskRow] = []
    for window in sorted(rows_by_window):
        ordered.extend(rows_by_window[window])
    return ordered, rows_by_window


def _write_fold_candidate_embeddings(
    rows: Sequence[MaskRow],
    *,
    model_base_path: Path,
    split_mode: str,
    centroids: Mapping[str, np.ndarray],
    prepared_root: Path,
    titanet_cache_root: Path,
    output_path: Path,
    window_seconds: float,
    sample_rate: int,
    row_batch_size: int,
    embed_batch_size: int,
    device: str,
    args: argparse.Namespace,
) -> None:
    ordered_rows, rows_by_window = _ordered_rows(rows)
    del ordered_rows
    candidates = tuple(CORE_SPEAKERS)
    titanet = _load_titanet(device)
    samples = int(round(window_seconds * sample_rate))
    embedding_dim = next(iter(centroids.values())).shape[0]
    model_cache: Dict[str, ConditionedTasNetExtractor] = {}

    all_embeddings: List[np.ndarray] = []
    all_windows: List[str] = []
    all_indices: List[int] = []
    all_truths: List[str] = []
    all_shares: List[float] = []
    all_active: List[int] = []

    for window_name, window_rows in sorted(rows_by_window.items()):
        word_payload = _load_titanet_word_npz(
            titanet_cache_root,
            "reference",
            window_name,
            window_seconds,
        )
        starts = np.asarray(word_payload["word_starts"], dtype=np.float32)
        ends = np.asarray(word_payload["word_ends"], dtype=np.float32)
        mixed = _load_audio(prepared_root / window_name / "mixed.wav", sample_rate)
        waves: List[np.ndarray] = []
        for row in window_rows:
            midpoint = (float(starts[row.index]) + float(ends[row.index])) / 2.0
            start_sample = int(math.floor((midpoint - window_seconds / 2.0) * sample_rate))
            waves.append(_slice_wave(mixed, start_sample, start_sample + samples))

        window_vectors: List[np.ndarray] = []
        for offset in range(0, len(waves), row_batch_size):
            batch_rows = window_rows[offset : offset + row_batch_size]
            batch_waves = waves[offset : offset + row_batch_size]
            extracted_chunks: List[np.ndarray] = []
            for group in sorted({_split_group(row, split_mode) for row in batch_rows}):
                group_indices = [
                    index
                    for index, row in enumerate(batch_rows)
                    if _split_group(row, split_mode) == group
                ]
                if group not in model_cache:
                    model_path = _model_path_for_group(model_base_path, group)
                    if not model_path.exists():
                        raise FileNotFoundError(model_path)
                    model_cache[group] = _load_fold_model(
                        model_path,
                        embedding_dim=embedding_dim,
                        device=device,
                        args=args,
                    )
                group_waves = np.stack([batch_waves[index] for index in group_indices])
                expanded_waves = np.repeat(group_waves, len(candidates), axis=0)
                expanded_labels = [
                    speaker for _wave_index in group_indices for speaker in candidates
                ]
                extracted = _extract_batch(
                    model_cache[group],
                    expanded_waves,
                    expanded_labels,
                    centroids=centroids,
                    device=device,
                )
                extracted_chunks.append(
                    extracted.reshape(len(group_indices), len(candidates), extracted.shape[-1])
                )
            extracted_by_row = np.concatenate(extracted_chunks, axis=0)
            embedded = _embed_waveforms(
                titanet,
                list(extracted_by_row.reshape(-1, extracted_by_row.shape[-1])),
                sample_rate=sample_rate,
                batch_size=embed_batch_size,
                device=device,
            )
            window_vectors.append(
                embedded.reshape(len(batch_waves), len(candidates), embedded.shape[-1])
            )
        all_embeddings.append(np.vstack(window_vectors).astype(np.float32))
        all_windows.extend([window_name] * len(window_rows))
        all_indices.extend([row.index for row in window_rows])
        all_truths.extend([row.truth for row in window_rows])
        all_shares.extend([row.target_share for row in window_rows])
        all_active.extend([row.active_5pct for row in window_rows])
        print(f"eval_stem_candidate_embedded {window_name}: {len(window_rows)} rows", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=np.concatenate(all_embeddings, axis=0).astype(np.float32),
        candidates=np.asarray(candidates),
        windows=np.asarray(all_windows),
        indices=np.asarray(all_indices, dtype=np.int32),
        truths=np.asarray(all_truths),
        target_shares=np.asarray(all_shares, dtype=np.float32),
        active_5pct=np.asarray(all_active, dtype=np.int16),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all-candidate selection for eval-stem trained target extractors."
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path(".outputs/speaker_id_baseline_smoke_graph/prepared_eval"),
    )
    parser.add_argument(
        "--model-base-path",
        type=Path,
        default=Path("/tmp/codex_eval_stem_tasnet_lgo_s300_s800.pt"),
    )
    parser.add_argument(
        "--embedding-output",
        type=Path,
        default=Path("/tmp/codex_eval_stem_candidate_embeddings.npz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/codex_eval_stem_candidate_results.json")
    )
    parser.add_argument(
        "--dominance-json",
        type=Path,
        default=Path("/tmp/codex_mixed_vs_clean_dominance_slices.json"),
    )
    parser.add_argument(
        "--clean-bank",
        type=Path,
        default=Path("/tmp/codex_titanet_clean_recent_s50_s60_train.npz"),
    )
    parser.add_argument(
        "--titanet-cache-root", type=Path, default=Path("/tmp/codex_titanet_word_window")
    )
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--max-target-share", type=float, default=0.90)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--row-batch-size", type=int, default=4)
    parser.add_argument("--embed-batch-size", type=int, default=24)
    parser.add_argument("--conditioning", choices=("one_hot",), default="one_hot")
    parser.add_argument(
        "--split-mode", choices=("leave_group", "leave_session"), default="leave_group"
    )
    parser.add_argument("--enc-feats", type=int, default=128)
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--cond-dim", type=int, default=64)
    parser.add_argument("--enc-kernel", type=int, default=16)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--stacks", type=int, default=2)
    parser.add_argument(
        "--mask-activation",
        choices=("sigmoid", "relu", "softplus"),
        default="sigmoid",
    )
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    centroids = _conditioning_vectors(
        clean_bank_path=args.clean_bank.expanduser(),
        mode=str(args.conditioning),
    )
    rows = _load_rows(args.dominance_json.expanduser(), float(args.max_target_share))
    rows = _limit_rows_by_group(rows, limit=int(args.eval_limit), seed=int(args.seed))
    if not args.embedding_output.expanduser().exists():
        _write_fold_candidate_embeddings(
            rows,
            model_base_path=args.model_base_path.expanduser(),
            split_mode=str(args.split_mode),
            centroids=centroids,
            prepared_root=args.prepared_root.expanduser(),
            titanet_cache_root=args.titanet_cache_root.expanduser(),
            output_path=args.embedding_output.expanduser(),
            window_seconds=float(args.window_seconds),
            sample_rate=int(args.sample_rate),
            row_batch_size=int(args.row_batch_size),
            embed_batch_size=int(args.embed_batch_size),
            device=device,
            args=args,
        )

    candidate_embeddings, candidates, embedding_rows = _load_candidate_embeddings(
        args.embedding_output.expanduser()
    )
    rows_by_window: Dict[str, List[MaskRow]] = defaultdict(list)
    for row in embedding_rows:
        rows_by_window[row.window].append(row)
    clean_bank = _load_clean_bank(args.clean_bank.expanduser())
    training_items = _collect_training_items(
        rows_by_window,
        prepared_root=args.prepared_root.expanduser(),
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        window_seconds=float(args.window_seconds),
    )
    mixed_embeddings = _mixed_same_rows(
        rows_by_window,
        titanet_cache_root=args.titanet_cache_root.expanduser(),
        window_seconds=float(args.window_seconds),
    )
    scored = _evaluate_candidate_embeddings(
        candidate_embeddings,
        candidates,
        embedding_rows,
        clean_bank=clean_bank,
        training_items=training_items,
        mixed_embeddings=mixed_embeddings,
    )
    payload = {
        "model": "eval_stem_all_candidate_target_extractor",
        "split_mode": str(args.split_mode),
        "selected_rows": len(embedding_rows),
        "selected_speakers": dict(Counter(row.truth for row in embedding_rows)),
        "model_base_path": str(args.model_base_path.expanduser()),
        "candidate_embedding_cache": str(args.embedding_output.expanduser()),
        "scores": {key: value for key, value in scored.items() if key != "diagnostic_rows"},
    }
    args.output.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("name,examples,accuracy", flush=True)
    for name, score in sorted(payload["scores"].items()):
        direct = score.get("direct", {})
        if direct:
            print(
                ",".join(
                    [
                        name,
                        str(direct["examples"]),
                        f"{float(direct['accuracy']):.4f}",
                    ]
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
