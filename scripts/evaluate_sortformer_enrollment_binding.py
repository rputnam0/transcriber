from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf

from speaker_id_oracle_mask_sweep import _embed_waveforms, _load_titanet


def _read_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _resolve_path(value: object, *, relative_to: Path) -> Path:
    path = Path(str(value))
    for candidate in (path, Path.cwd() / path, relative_to / path):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(path)


def _session_name(cut_id: object) -> str:
    match = re.match(r"session_(\d+)_", str(cut_id or ""))
    if not match:
        raise ValueError(f"Cannot infer session from cut id {cut_id!r}")
    return f"Session {int(match.group(1))}"


def _load_wave(
    path: Path,
    *,
    sample_rate: int = 16_000,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> np.ndarray:
    from scipy.signal import resample_poly

    info = sf.info(path)
    start = max(0, int(round(start_seconds * info.samplerate)))
    frames = -1
    if duration_seconds is not None:
        frames = max(1, int(round(duration_seconds * info.samplerate)))
    wave, source_rate = sf.read(
        path,
        start=start,
        frames=frames,
        dtype="float32",
        always_2d=True,
    )
    mono = np.asarray(wave.mean(axis=1), dtype=np.float32)
    if int(source_rate) != sample_rate:
        divisor = math.gcd(int(source_rate), sample_rate)
        mono = resample_poly(
            mono,
            sample_rate // divisor,
            int(source_rate) // divisor,
        ).astype(np.float32)
    return np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)


def trim_clean_enrollment(
    wave: np.ndarray,
    *,
    sample_rate: int,
    frame_seconds: float = 0.02,
    minimum_seconds: float = 0.5,
) -> np.ndarray:
    """Remove near-zero gaps from isolated enrollment audio, never from the mono mixture."""
    samples = np.asarray(wave, dtype=np.float32).reshape(-1)
    frame_samples = max(1, int(round(sample_rate * frame_seconds)))
    frame_count = len(samples) // frame_samples
    if not frame_count:
        return samples
    framed = samples[: frame_count * frame_samples].reshape(frame_count, frame_samples)
    rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
    positive = rms[rms > 1e-5]
    reference = float(np.quantile(positive, 0.9)) if positive.size else 0.0
    active = rms >= max(1e-5, reference * 0.04)
    selected = framed[active].reshape(-1)
    minimum_samples = int(round(minimum_seconds * sample_rate))
    if len(selected) < minimum_samples:
        return samples
    return selected.astype(np.float32)


def enrollment_paths(
    manifest_path: Path,
    *,
    session: str,
    max_clips_per_speaker: int,
) -> dict[str, list[Path]]:
    candidates: dict[str, list[tuple[float, Path]]] = collections.defaultdict(list)
    for row in _read_jsonl(manifest_path):
        if str(row.get("session") or "") != session:
            continue
        speaker = str(row.get("speaker_id") or "").strip()
        materialized = dict(row.get("materialized") or {})
        for value in list(materialized.get("positive_enrollment_paths") or []):
            try:
                path = _resolve_path(value, relative_to=manifest_path.resolve().parent)
            except FileNotFoundError:
                continue
            candidates[speaker].append((float(row.get("window_start") or 0.0), path))
    return {
        speaker: [path for _, path in sorted(values)[:max_clips_per_speaker]]
        for speaker, values in sorted(candidates.items())
        if values
    }


def enrollment_provenance(manifest_path: Path, *, session: str) -> dict:
    rows = [row for row in _read_jsonl(manifest_path) if str(row.get("session") or "") == session]
    uses_evaluation_audio = any(
        bool(row.get("uses_evaluation_session_audio", True)) for row in rows
    )
    source_splits = sorted(
        {str(row["enrollment_source_split"]) for row in rows if row.get("enrollment_source_split")}
    )
    source_sessions = sorted(
        {
            str(source_session)
            for row in rows
            for source_session in list(row.get("enrollment_source_sessions") or [])
        }
    )
    if rows and not uses_evaluation_audio and source_splits == ["train"]:
        source = "cross-session-training-profile"
    else:
        source = "same-session-or-unspecified-enrollment"
    return {
        "source": source,
        "source_splits": source_splits,
        "source_sessions": source_sessions,
        "uses_evaluation_session_audio": uses_evaluation_audio,
    }


def slot_waveforms(
    mixture: np.ndarray,
    probabilities: np.ndarray,
    *,
    sample_rate: int,
    activity_threshold: float,
    exclusivity_margin: float,
    min_slot_seconds: float,
    min_wave_seconds: float = 0.5,
) -> tuple[list[str], list[np.ndarray], list[dict]]:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"Expected [frames, speakers] probabilities, got {values.shape}")
    frame_samples = len(mixture) / values.shape[0]
    other_max = np.zeros_like(values)
    for slot in range(values.shape[1]):
        other = [index for index in range(values.shape[1]) if index != slot]
        if other:
            other_max[:, slot] = np.max(values[:, other], axis=1)

    names = []
    waves = []
    metadata = []
    for slot in range(values.shape[1]):
        active = values[:, slot] >= activity_threshold
        activity_seconds = float(active.sum() * frame_samples / sample_rate)
        if activity_seconds < min_slot_seconds:
            continue
        selected = active & (values[:, slot] - other_max[:, slot] >= exclusivity_margin)
        if float(selected.sum() * frame_samples / sample_rate) < min_wave_seconds:
            selected = active
        parts = []
        run_start = None
        for frame, enabled in enumerate(np.append(selected, False)):
            if enabled and run_start is None:
                run_start = frame
            elif not enabled and run_start is not None:
                first = int(round(run_start * frame_samples))
                last = min(len(mixture), int(round(frame * frame_samples)))
                if last > first:
                    parts.append(mixture[first:last])
                run_start = None
        if not parts:
            continue
        slot_wave = np.concatenate(parts).astype(np.float32)
        minimum_samples = int(round(min_wave_seconds * sample_rate))
        slot_wave = np.pad(slot_wave, (0, max(0, minimum_samples - len(slot_wave))))
        names.append(f"speaker_{slot}")
        waves.append(slot_wave)
        metadata.append(
            {
                "slot": f"speaker_{slot}",
                "activity_seconds": activity_seconds,
                "embedding_seconds": len(slot_wave) / sample_rate,
                "mean_activity_probability": float(values[active, slot].mean()),
                "mean_exclusivity_margin": float(
                    (values[active, slot] - other_max[active, slot]).mean()
                ),
            }
        )
    return names, waves, metadata


def assignment_maps(
    similarity: np.ndarray,
    slots: Sequence[str],
    speakers: Sequence[str],
) -> tuple[dict[str, str], dict[str, str], dict[str, dict]]:
    from scipy.optimize import linear_sum_assignment

    values = np.asarray(similarity, dtype=np.float32)
    if values.shape != (len(slots), len(speakers)):
        raise ValueError("Similarity matrix shape does not match slots and speakers")
    independent = {
        slot: speakers[int(np.argmax(values[index]))] for index, slot in enumerate(slots)
    }
    rows, columns = linear_sum_assignment(-values)
    one_to_one = {slots[int(row)]: speakers[int(column)] for row, column in zip(rows, columns)}
    evidence = {}
    for row, slot in enumerate(slots):
        order = np.argsort(values[row])[::-1]
        best = int(order[0])
        runner_up = int(order[1]) if len(order) > 1 else best
        evidence[slot] = {
            "best_speaker": speakers[best],
            "best_score": float(values[row, best]),
            "runner_up_speaker": speakers[runner_up],
            "runner_up_score": float(values[row, runner_up]),
            "margin": float(values[row, best] - values[row, runner_up]),
            "scores": {
                speaker: float(values[row, column]) for column, speaker in enumerate(speakers)
            },
        }
    return independent, one_to_one, evidence


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)


def enrollment_leave_one_out_score(
    embeddings: np.ndarray,
    labels: Sequence[str],
) -> dict:
    values = _normalize_rows(embeddings)
    predictions = []
    scores = []
    for index, truth in enumerate(labels):
        speakers = sorted(set(labels))
        centroids = []
        candidates = []
        for speaker in speakers:
            selected = [
                row for row, label in enumerate(labels) if label == speaker and row != index
            ]
            if not selected:
                continue
            candidates.append(speaker)
            centroids.append(_normalize_rows(values[selected].mean(axis=0, keepdims=True))[0])
        similarities = values[index] @ np.stack(centroids).T
        winner = int(np.argmax(similarities))
        predictions.append(candidates[winner])
        scores.append(float(similarities[winner]))
    correct = sum(prediction == truth for prediction, truth in zip(predictions, labels))
    return {
        "clips": len(labels),
        "correct": correct,
        "accuracy": correct / max(1, len(labels)),
        "mean_winner_score": float(np.mean(scores)) if scores else 0.0,
    }


def _score_mapping(predicted: Mapping[str, str], expected: Mapping[str, str]) -> dict:
    available = {slot: speaker for slot, speaker in expected.items() if slot in predicted}
    correct = sum(predicted[slot] == speaker for slot, speaker in available.items())
    return {
        "reference_slots": len(expected),
        "mapped_reference_slots": len(available),
        "correct_slots": correct,
        "accuracy": correct / len(expected) if expected else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bind mono Sortformer streams to known speakers using clean enrollment audio."
    )
    parser.add_argument("--activity-cutset", type=Path, required=True)
    parser.add_argument("--enrollment-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--activity-threshold", type=float, default=0.5)
    parser.add_argument("--exclusivity-margin", type=float, default=0.10)
    parser.add_argument("--min-slot-seconds", type=float, default=0.5)
    parser.add_argument("--max-enrollment-clips", type=int, default=4)
    parser.add_argument("--max-cuts", type=int, default=0)
    args = parser.parse_args()

    cuts = list(_read_jsonl(args.activity_cutset))
    if args.max_cuts > 0:
        cuts = cuts[: args.max_cuts]
    if not cuts:
        raise ValueError("No activity cuts were found")

    model = _load_titanet(args.device)
    enrollment_cache: dict[
        str,
        tuple[list[str], np.ndarray, dict[str, list[str]], dict, dict],
    ] = {}
    records = []
    for cut in cuts:
        cut_id = str(cut["id"]).removesuffix("-mask-sortformer")
        session = _session_name(cut_id)
        if session not in enrollment_cache:
            paths_by_speaker = enrollment_paths(
                args.enrollment_manifest,
                session=session,
                max_clips_per_speaker=args.max_enrollment_clips,
            )
            if not paths_by_speaker:
                raise ValueError(f"No enrollment audio found for {session}")
            speakers = sorted(paths_by_speaker)
            enrollment_waves = [
                trim_clean_enrollment(
                    _load_wave(path),
                    sample_rate=16_000,
                )
                for speaker in speakers
                for path in paths_by_speaker[speaker]
            ]
            enrollment_labels = [
                speaker for speaker in speakers for _path in paths_by_speaker[speaker]
            ]
            enrollment_vectors = _embed_waveforms(
                model,
                enrollment_waves,
                sample_rate=16_000,
                batch_size=args.batch_size,
                device=args.device,
            )
            enrollment_diagnostic = enrollment_leave_one_out_score(
                enrollment_vectors,
                enrollment_labels,
            )
            provenance = enrollment_provenance(args.enrollment_manifest, session=session)
            centroids = []
            for speaker in speakers:
                selected = enrollment_vectors[
                    [label == speaker for label in enrollment_labels]
                ].mean(axis=0, keepdims=True)
                centroids.append(_normalize_rows(selected)[0])
            enrollment_cache[session] = (
                speakers,
                np.stack(centroids),
                {
                    speaker: [str(path) for path in paths_by_speaker[speaker]]
                    for speaker in speakers
                },
                enrollment_diagnostic,
                provenance,
            )
        (
            speakers,
            centroids,
            enrollment_sources,
            enrollment_diagnostic,
            provenance,
        ) = enrollment_cache[session]

        source = list(dict(cut["recording"])["sources"])[0]
        audio_path = _resolve_path(
            source["source"],
            relative_to=args.activity_cutset.resolve().parent,
        )
        mixture = _load_wave(
            audio_path,
            start_seconds=float(cut.get("start") or 0.0),
            duration_seconds=float(cut.get("duration") or 0.0),
        )
        custom = dict(cut.get("custom") or {})
        probability_path = _resolve_path(
            custom.get("sortformer_probabilities_path"),
            relative_to=args.activity_cutset.resolve().parent,
        )
        slots, waves, slot_metadata = slot_waveforms(
            mixture,
            np.load(probability_path, allow_pickle=False),
            sample_rate=16_000,
            activity_threshold=args.activity_threshold,
            exclusivity_margin=args.exclusivity_margin,
            min_slot_seconds=args.min_slot_seconds,
        )
        slot_vectors = _embed_waveforms(
            model,
            waves,
            sample_rate=16_000,
            batch_size=args.batch_size,
            device=args.device,
        )
        similarity = _normalize_rows(slot_vectors) @ _normalize_rows(centroids).T
        independent, one_to_one, evidence = assignment_maps(
            similarity,
            slots,
            speakers,
        )
        expected = dict(custom.get("sortformer_slot_to_speaker") or {})
        records.append(
            {
                "cut_id": cut_id,
                "session": session,
                "mono_audio_path": str(audio_path),
                "mono_audio_offset_seconds": float(cut.get("start") or 0.0),
                "probability_path": str(probability_path),
                "inference_uses_isolated_target_audio": False,
                "enrollment_audio_source": provenance["source"],
                "enrollment_provenance": provenance,
                "enrollment_sources": enrollment_sources,
                "enrollment_leave_one_out": enrollment_diagnostic,
                "slots": slot_metadata,
                "independent_mapping": independent,
                "one_to_one_mapping": one_to_one,
                "binding_evidence": evidence,
                "oracle_mapping_diagnostic_only": expected,
                "independent_score": _score_mapping(independent, expected),
                "one_to_one_score": _score_mapping(one_to_one, expected),
            }
        )
        print(
            f"{cut_id}: independent={records[-1]['independent_score']['accuracy']:.3f} "
            f"one_to_one={records[-1]['one_to_one_score']['accuracy']:.3f}",
            flush=True,
        )

    summary = {}
    for mode in ("independent", "one_to_one"):
        scores = [record[f"{mode}_score"] for record in records]
        reference = sum(int(score["reference_slots"]) for score in scores)
        summary[mode] = {
            "reference_slots": reference,
            "mapped_reference_slots": sum(int(score["mapped_reference_slots"]) for score in scores),
            "correct_slots": sum(int(score["correct_slots"]) for score in scores),
        }
        summary[mode]["accuracy"] = summary[mode]["correct_slots"] / max(1, reference)
    result = {
        "activity_cutset": str(args.activity_cutset),
        "enrollment_manifest": str(args.enrollment_manifest),
        "activity_source": "mono-sortformer",
        "binding_model": "titanet_small",
        "cut_count": len(records),
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
