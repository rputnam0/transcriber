"""Read-only audit of saved references, current code behavior, and optional stem audio.

No inference, training, network access, or reference repair. ZIP contents are temporary.
Run with the repo's Python environment; see docs/analysis/diarization_eda_20260918.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

import numpy as np


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def provenance(path, root):
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def reference_profile(rows):
    counts = Counter()
    scores, durations, examples = [], [], []
    speakers = Counter()
    sessions = Counter()
    strata = defaultdict(Counter)
    for group in rows:
        words = group["words"]
        sessions[group["session"]] += len(words)
        for i, word in enumerate(words):
            start, end = float(word["start"]), float(word["end"])
            duration = end - start
            counts["words"] += 1
            durations.append(duration)
            speakers[word["speaker"]] += 1
            score = word.get("score")
            if score is not None:
                scores.append(score)
                counts["score_lt_050"] += score < 0.5
                counts["score_lt_080"] += score < 0.8
            counts["nonpositive_duration"] += duration <= 0
            counts["duration_gt_1s"] += duration > 1
            counts["duration_gt_2s"] += duration > 2
            counts["outside_window"] += (
                start < 0 or end > group["window_end"] - group["window_start"]
            )
            counts["outside_source_span_by_250ms"] += (
                start < word["source_span_start"] - 0.25 or end > word["source_span_end"] + 0.25
            )
            other_overlap = same_overlap = duplicate = False
            for j, other in enumerate(words):
                if i == j:
                    continue
                overlap = min(end, other["end"]) - max(start, other["start"])
                if overlap < 0.02:
                    continue
                if other["speaker"] != word["speaker"]:
                    other_overlap = True
                else:
                    same_overlap = True
                    union = max(end, other["end"]) - min(start, other["start"])
                    token = str(word.get("normalized", word["text"])).casefold()
                    other_token = str(other.get("normalized", other["text"])).casefold()
                    if token == other_token and overlap / union >= 0.5:
                        duplicate = True
                        if j > i and len(examples) < 12:
                            examples.append(
                                {
                                    "session": group["session"],
                                    "window_start": group["window_start"],
                                    "speaker": word["speaker"],
                                    "token": token,
                                    "intervals": [[start, end], [other["start"], other["end"]]],
                                }
                            )
            counts["cross_speaker_overlap_ge_20ms"] += other_overlap
            counts["same_speaker_overlap_ge_20ms"] += same_overlap
            counts["duplicate_candidate_words_iou_ge_050"] += duplicate
            bucket = (
                "score_ge_080" if score is not None and score >= 0.8 else "lower_or_missing_score"
            )
            strata[bucket]["words"] += 1
            strata[bucket]["cross_speaker_overlap_ge_20ms"] += other_overlap
            strata[bucket]["duplicate_candidate_words_iou_ge_050"] += duplicate
    return {
        "groups": len(rows),
        "sessions": dict(sessions),
        "counts": dict(counts),
        "speaker_word_counts": dict(speakers),
        "by_alignment_score": dict(strata),
        "duration_quantiles_seconds": np.quantile(durations, [0, 0.1, 0.5, 0.9, 0.99, 1]).tolist(),
        "score_quantiles": np.quantile(scores, [0, 0.1, 0.5, 0.9, 1]).tolist(),
        "duplicate_candidates_preview": examples,
        "caveat": "Alignment scores are not calibrated probabilities; overlap is label-derived, not verified physical speech.",
    }


def manifest_profile(rows):
    groups = {}
    splits = defaultdict(set)
    speakers = Counter()
    duplicate_spans = 0
    for row in rows:
        key = (row["session"], row["window_start"], row["window_end"])
        groups.setdefault(key, row)
        splits[row["split_id"]].add(row["session"])
    for row in groups.values():
        seen = set()
        for span in row["word_spans"]:
            key = (span["speaker"], span["start"], span["end"], span["text"])
            duplicate_spans += key in seen
            seen.add(key)
            speakers[span["speaker"]] += span["word_count"]
    return {
        "candidate_rows": len(rows),
        "unique_mixtures": len(groups),
        "unique_mixture_hours": sum(k[2] - k[1] for k in groups) / 3600,
        "summed_candidate_hours_NOT_unique_audio": sum(r["duration"] for r in rows) / 3600,
        "duplicate_row_ids": len(rows) - len({r["row_id"] for r in rows}),
        "exact_duplicate_spans_within_group": duplicate_spans,
        "sessions_by_split": {k: sorted(v) for k, v in splits.items()},
        "cross_split_session_intersections": {
            a + "/" + b: sorted(splits[a] & splits[b]) for a in splits for b in splits if a < b
        },
        "words_by_speaker_unique_mixtures": dict(speakers),
        "target_absent_candidate_rows": sum(r["target_word_count"] == 0 for r in rows),
        "active_speakers_per_mixture": dict(
            Counter(r["active_speaker_bucket"] for r in groups.values())
        ),
    }


def code_probes(root):
    sys.path.insert(0, str(root / "src"))
    from transcriber.diarization import DiarizationResult, DiarizationTurn as Turn
    from transcriber import transcript_pipeline as pipeline
    from transcriber.multitrack_eval import WordSpan, score_word_speaker_alignment

    turns = [Turn(0, 3, "A"), Turn(3, 3.5, "B")]
    from unittest.mock import patch

    with patch.object(pipeline, "extract_embeddings_for_segments", return_value=([], {})) as mocked:
        pipeline._aggregate_speaker_embeddings(
            "unused.wav",
            DiarizationResult(turns, turns, {}),
            hf_token=None,
            diarization_model_name=None,
            force_device="cpu",
            quiet=True,
        )
        submitted = mocked.call_args.args[1]
    metrics = score_word_speaker_alignment(
        [WordSpan("A", 0, 1, "hello")], [WordSpan("A", 0, 1, "banana")]
    )
    return {
        "short_speaker_embedding_submitted_labels": [x[2] for x in submitted],
        "fallback_label_despite_matching_B": pipeline._choose_turn_label(
            10, 10.4, [Turn(0, 1, "A")], [Turn(10, 11, "B")]
        ),
        "wrong_token_timed_accuracy": metrics["accuracy"],
        "wrong_token_lexical_accuracy": metrics["lexical_accuracy"],
    }


def waveform_summary(wave, sample_rate=16000):
    frames = wave[: len(wave) // 320 * 320].reshape(-1, 320)
    power = np.mean(frames**2, axis=1)
    db = 10 * np.log10(np.maximum(power, 1e-12))
    active = frames[db > -50]
    spectrum = np.sum(np.abs(np.fft.rfft(active * np.hanning(320), axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(320, 1 / sample_rate)
    total = max(float(spectrum.sum()), 1e-12)
    return {
        "rms_dbfs_p10_p50_p90": np.quantile(db, [0.1, 0.5, 0.9]).tolist(),
        "fraction_frames_below_minus70": float(np.mean(db < -70)),
        "fraction_samples_abs_ge_0999": float(np.mean(np.abs(wave) >= 0.999)),
        "peak": float(np.max(np.abs(wave))),
        "active_spectral_power_below_300hz_fraction": float(spectrum[freqs < 300].sum() / total),
        "active_spectral_power_above_4000hz_fraction": float(spectrum[freqs > 4000].sum() / total),
        "active_spectral_power_above_6000hz_fraction": float(spectrum[freqs > 6000].sum() / total),
    }


def mono_profile(path):
    info = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
        )
    )
    stream = info["streams"][0]
    duration = float(info["format"]["duration"])
    windows = []
    for fraction in [0, 0.5, 1]:
        start = round((duration - 300) * fraction, 3)
        data = subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                str(start),
                "-i",
                str(path),
                "-t",
                "300",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "f32le",
                "pipe:1",
            ]
        )
        wave = np.frombuffer(data, dtype="<f4")
        native_bytes = subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                str(start),
                "-i",
                str(path),
                "-t",
                "30",
                "-f",
                "f32le",
                "pipe:1",
            ]
        )
        native = np.frombuffer(native_bytes, dtype="<f4").reshape(-1, stream["channels"])
        native_summary = {
            "seconds": 30,
            "peak": float(np.max(np.abs(native))),
            "fraction_samples_at_pcm16_rail": float(np.mean(np.abs(native) >= 32767 / 32768)),
        }
        if stream["channels"] == 2:
            native_summary["left_right_correlation"] = float(np.corrcoef(native.T)[0, 1])
            averaged_bytes = subprocess.check_output(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-ss",
                    str(start),
                    "-i",
                    str(path),
                    "-t",
                    "30",
                    "-af",
                    "pan=mono|c0=0.5*c0+0.5*c1",
                    "-ar",
                    "16000",
                    "-f",
                    "f32le",
                    "pipe:1",
                ]
            )
            averaged = np.frombuffer(averaged_bytes, dtype="<f4")
            native_summary["ffmpeg_default_vs_channel_mean_gain_db"] = float(
                20 * np.log10(np.linalg.norm(wave[: len(averaged)]) / np.linalg.norm(averaged))
            )
        windows.append(
            {
                "start": start,
                "seconds": len(wave) / 16000,
                **waveform_summary(wave),
                "native_first_30s": native_summary,
            }
        )
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_rate": int(stream["sample_rate"]),
        "channels": stream["channels"],
        "codec": stream["codec_name"],
        "duration": duration,
        "windows": windows,
        "limitations": "Three windows from one early session; no clean source or measured speaker accuracy. Spectral metrics are descriptive, not causal evidence.",
    }


def audio_profile(zip_path):
    """Three deterministic 5-minute windows; energy gates are sensitivity checks, not VAD."""
    sample_rate = 16000
    seconds = 300
    with (
        tempfile.TemporaryDirectory(prefix="transcriber-eda-") as folder,
        ZipFile(zip_path) as archive,
    ):
        paths, metadata = [], []
        for index, member in enumerate(sorted(n for n in archive.namelist() if n.endswith(".ogg"))):
            path = Path(folder) / f"track_{index + 1}.ogg"
            path.write_bytes(archive.read(member))  # safe flat names; no archive extraction paths
            info = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_streams",
                        "-show_format",
                        "-of",
                        "json",
                        str(path),
                    ]
                )
            )
            stream = info["streams"][0]
            metadata.append(
                {
                    "track": index + 1,
                    "sample_rate": int(stream["sample_rate"]),
                    "channels": stream["channels"],
                    "codec": stream["codec_name"],
                    "duration": float(info["format"]["duration"]),
                }
            )
            paths.append(path)
        common_duration = min(m["duration"] for m in metadata)
        windows, levels = [], defaultdict(list)
        peaks, clip_counts, sample_counts = defaultdict(float), Counter(), Counter()
        for fraction in [0.2, 0.5, 0.8]:
            start = round((common_duration - seconds) * fraction, 3)
            waves = []
            for index, path in enumerate(paths):
                data = subprocess.check_output(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-ss",
                        str(start),
                        "-i",
                        str(path),
                        "-t",
                        str(seconds),
                        "-ac",
                        "1",
                        "-ar",
                        str(sample_rate),
                        "-f",
                        "f32le",
                        "pipe:1",
                    ]
                )
                wave = np.frombuffer(data, dtype="<f4")
                wave = np.pad(wave, (0, max(0, seconds * sample_rate - len(wave))))[
                    : seconds * sample_rate
                ]
                waves.append(wave)
                peaks[index] = max(peaks[index], float(np.max(np.abs(wave))))
                clip_counts[index] += int(np.sum(np.abs(wave) >= 0.999))
                sample_counts[index] += len(wave)
            matrix = np.stack(waves)
            power = np.mean(matrix.reshape(len(paths), -1, 320) ** 2, axis=2)
            db = 10 * np.log10(np.maximum(power, 1e-12))
            for index in range(len(paths)):
                levels[index].extend(db[index].tolist())
            threshold_results = {}
            for threshold in [-50, -40, -30]:
                active = db > threshold
                count = active.sum(axis=0)
                speech = count > 0
                threshold_results[str(threshold)] = {
                    "any_active_fraction": float(speech.mean()),
                    "multi_active_fraction_all_frames": float((count > 1).mean()),
                    "multi_active_fraction_active_frames": float(
                        np.sum(count > 1) / max(speech.sum(), 1)
                    ),
                }
            windows.append(
                {
                    "start": start,
                    "seconds": seconds,
                    "energy_threshold_dbfs": threshold_results,
                    "unscaled_sum_peak": float(np.max(np.abs(matrix.sum(axis=0)))),
                    "unscaled_sum_fraction_abs_ge_1": float(
                        np.mean(np.abs(matrix.sum(axis=0)) >= 1)
                    ),
                    "mixture_summary": waveform_summary(matrix.sum(axis=0)),
                }
            )
        for index, meta in enumerate(metadata):
            db = np.asarray(levels[index])
            active_db = db[db > -50]
            meta.update(
                {
                    "rms_dbfs_quantiles_p10_p50_p90": np.quantile(db, [0.1, 0.5, 0.9]).tolist(),
                    "rms_active_gt_minus50_median_dbfs": (
                        float(np.median(active_db)) if len(active_db) else None
                    ),
                    "fraction_frames_above_minus50": float(np.mean(db > -50)),
                    "sample_peak_16khz": peaks[index],
                    "fraction_samples_abs_ge_0999_16khz": clip_counts[index] / sample_counts[index],
                }
            )
        return {
            "selection": "Three fixed fractional positions, not chosen using model errors; one session only.",
            "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
            "frame_seconds": 0.02,
            "tracks": metadata,
            "windows": windows,
            "limitations": "Energy is not speech activity; 16kHz resampling is not a native-rate clipping audit. No lag, identity, ASR or perceptual quality validation.",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--audio-zip", type=Path)
    parser.add_argument("--mono-audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = (
        args.artifact_root
        or root
        / ".outputs/auditor_packages/speaker_diarization_forced_alignment_audit_20260601_105409/artifacts/wsl_tse_manifest_text"
    )
    sources = []
    result = {"artifact_snapshot": "2026-06-01 (not the later August runs)", "references": {}}
    for relative in [
        "codex_drive_v4_large_inventory/forced_word_reference_s51_s58_s65_mms/forced_word_reference_groups.jsonl",
        "codex_drive_v3_test_all_materialized/forced_word_reference_all_mms/forced_word_reference_groups.jsonl",
    ]:
        path = base / relative
        sources.append(provenance(path, base))
        result["references"][relative] = reference_profile(read_rows(path))
    path = base / "codex_drive_v4_large_inventory/speaker_tse_manifest.jsonl"
    sources.append(provenance(path, base))
    result["manifest"] = manifest_profile(read_rows(path))
    result["code_probes"] = code_probes(root)
    result["sources"] = sources
    if args.audio_zip:
        result["audio"] = audio_profile(args.audio_zip)
    if args.mono_audio:
        result["early_mono_audio"] = mono_profile(args.mono_audio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output.name}")


if __name__ == "__main__":
    main()
