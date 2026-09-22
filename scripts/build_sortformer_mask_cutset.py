from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf

from score_sortformer_oracle_diarization import (
    DEFAULT_SORTFORMER_MODEL,
    _apply_streaming_config,
    _load_sortformer_model,
    _run_sortformer,
)


def _span(item: object) -> dict:
    if isinstance(item, Mapping):
        start = float(item["start"])
        end = float(item.get("end", start + float(item.get("duration", 0.0))))
        speaker = item.get("speaker")
        text = item.get("text")
    else:
        start = float(item.start)
        end = float(item.end)
        speaker = item.speaker
        text = item.text
    return {
        "start": start,
        "end": end,
        "speaker": str(speaker),
        "text": str(text or "").strip(),
    }


def _speaker_frames(
    spans: Sequence[object],
    *,
    duration: float,
    frame_seconds: float,
) -> dict[str, set[int]]:
    frame_count = max(1, int(np.ceil(duration / frame_seconds)))
    frames: dict[str, set[int]] = defaultdict(set)
    for raw in spans:
        span = _span(raw)
        start = max(0, int(np.floor(span["start"] / frame_seconds)))
        end = min(frame_count, max(start + 1, int(np.ceil(span["end"] / frame_seconds))))
        frames[span["speaker"]].update(range(start, end))
    return dict(frames)


def match_predicted_speakers(
    reference: Sequence[object],
    predicted: Sequence[object],
    *,
    duration: float,
    frame_seconds: float = 0.08,
    min_f1: float = 0.10,
) -> tuple[dict[str, str], dict]:
    """Map mono-predicted slots to teacher identities for training-label construction only."""
    reference_frames = _speaker_frames(reference, duration=duration, frame_seconds=frame_seconds)
    predicted_frames = _speaker_frames(predicted, duration=duration, frame_seconds=frame_seconds)
    reference_speakers = sorted(reference_frames)
    predicted_speakers = sorted(predicted_frames)
    scores = {}
    for predicted_speaker in predicted_speakers:
        for reference_speaker in reference_speakers:
            predicted_set = predicted_frames[predicted_speaker]
            reference_set = reference_frames[reference_speaker]
            intersection = len(predicted_set & reference_set)
            scores[(predicted_speaker, reference_speaker)] = (
                2 * intersection / (len(predicted_set) + len(reference_set))
                if predicted_set or reference_set
                else 0.0
            )

    best_pairs: tuple[tuple[str, str], ...] = ()
    best_score = -1.0
    pair_count = min(len(predicted_speakers), len(reference_speakers))
    if pair_count:
        if len(predicted_speakers) <= len(reference_speakers):
            for reference_order in itertools.permutations(reference_speakers, pair_count):
                pairs = tuple(zip(predicted_speakers, reference_order))
                score = sum(scores[pair] for pair in pairs)
                if score > best_score:
                    best_pairs, best_score = pairs, score
        else:
            for predicted_order in itertools.permutations(predicted_speakers, pair_count):
                pairs = tuple(zip(predicted_order, reference_speakers))
                score = sum(scores[pair] for pair in pairs)
                if score > best_score:
                    best_pairs, best_score = pairs, score

    mapping = {
        predicted_speaker: reference_speaker
        for predicted_speaker, reference_speaker in best_pairs
        if scores[(predicted_speaker, reference_speaker)] >= min_f1
    }
    matched_reference = set(mapping.values())
    return mapping, {
        "reference_speakers": len(reference_speakers),
        "predicted_speakers": len(predicted_speakers),
        "matched_speakers": len(mapping),
        "reference_speaker_recall": (
            len(matched_reference) / len(reference_speakers) if reference_speakers else 1.0
        ),
        "mean_matched_f1": (
            sum(
                scores[(predicted_speaker, reference_speaker)]
                for predicted_speaker, reference_speaker in mapping.items()
            )
            / len(mapping)
            if mapping
            else 0.0
        ),
    }


def match_probability_speakers(
    reference: Sequence[object],
    probabilities: np.ndarray,
    *,
    duration: float,
) -> tuple[dict[str, str], dict]:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"Expected [frames, speakers] probabilities, got {values.shape}")
    frame_seconds = duration / max(1, values.shape[0])
    reference_frames = _speaker_frames(
        reference,
        duration=duration,
        frame_seconds=frame_seconds,
    )
    reference_speakers = sorted(reference_frames)
    slots = list(range(values.shape[1]))
    scores = {}
    for slot in slots:
        prediction = np.clip(values[:, slot], 0.0, 1.0)
        for speaker in reference_speakers:
            target = np.zeros(values.shape[0], dtype=np.float32)
            target[list(reference_frames[speaker])] = 1.0
            denominator = float(prediction.sum() + target.sum())
            scores[(slot, speaker)] = (
                2.0 * float((prediction * target).sum()) / denominator if denominator else 0.0
            )

    best_pairs: tuple[tuple[int, str], ...] = ()
    best_score = -1.0
    pair_count = min(len(slots), len(reference_speakers))
    for slot_order in itertools.permutations(slots, pair_count):
        pairs = tuple(zip(slot_order, reference_speakers))
        score = sum(scores[pair] for pair in pairs)
        if score > best_score:
            best_pairs, best_score = pairs, score
    mapping = {f"speaker_{slot}": speaker for slot, speaker in best_pairs}
    return mapping, {
        "reference_speakers": len(reference_speakers),
        "predicted_speakers": values.shape[1],
        "matched_speakers": len(mapping),
        "reference_speaker_recall": (
            len(mapping) / len(reference_speakers) if reference_speakers else 1.0
        ),
        "mean_matched_f1": (
            sum(scores[(slot, speaker)] for slot, speaker in best_pairs) / len(best_pairs)
            if best_pairs
            else 0.0
        ),
    }


def mapped_supervisions(
    reference: Sequence[object],
    predicted: Sequence[object],
    mapping: Mapping[str, str],
) -> list[dict]:
    transcript_by_speaker: dict[str, list[str]] = defaultdict(list)
    for raw in sorted(reference, key=lambda item: (_span(item)["start"], _span(item)["end"])):
        span = _span(raw)
        if span["text"]:
            transcript_by_speaker[span["speaker"]].append(span["text"])

    regions_by_speaker: dict[str, list[dict]] = defaultdict(list)
    for raw in predicted:
        span = _span(raw)
        speaker = mapping.get(span["speaker"])
        if speaker:
            regions_by_speaker[speaker].append(span)

    output = []
    for speaker, regions in sorted(regions_by_speaker.items()):
        transcript = " ".join(transcript_by_speaker[speaker])
        for index, region in enumerate(
            sorted(regions, key=lambda item: (item["start"], item["end"]))
        ):
            output.append(
                {
                    "speaker": speaker,
                    "start": region["start"],
                    "end": region["end"],
                    "text": transcript if index == 0 else "",
                }
            )
    return sorted(output, key=lambda item: (item["start"], item["end"], item["speaker"]))


def build_sortformer_cutset(
    *,
    input_cuts: Path,
    output_cuts: Path,
    model_name: str,
    restore_path: Path | None,
    device: str,
    batch_size: int,
    min_match_f1: float,
    max_cuts: int,
    chunk_len: int,
    right_context: int,
    fifo_len: int,
    spkcache_update_period: int,
    spkcache_len: int,
    save_probabilities: bool,
) -> dict:
    from lhotse import CutSet, SupervisionSegment
    from lhotse.utils import fastcopy

    model = _load_sortformer_model(
        model_name=model_name,
        restore_path=restore_path,
        device=device,
    )
    streaming_config = _apply_streaming_config(
        model,
        chunk_len=chunk_len,
        right_context=right_context,
        fifo_len=fifo_len,
        spkcache_update_period=spkcache_update_period,
        spkcache_len=spkcache_len,
    )
    output = []
    metrics = []
    probability_count = 0
    probability_dir = output_cuts.parent / (
        output_cuts.name.removesuffix(".jsonl.gz") + "_probabilities"
    )
    if save_probabilities:
        probability_dir.mkdir(parents=True, exist_ok=True)
    source_cuts = CutSet.from_file(input_cuts)
    for cut_index, cut in enumerate(source_cuts):
        if max_cuts > 0 and cut_index >= max_cuts:
            break
        samples, sample_rate = sf.read(
            cut.recording.sources[0].source,
            start=int(round(cut.start * cut.sampling_rate)),
            frames=cut.num_samples,
            dtype="float32",
            always_2d=True,
        )
        mono = np.asarray(samples.mean(axis=1), dtype=np.float32)
        probabilities = None
        if save_probabilities:
            raw_segments, raw_probabilities = model.diarize(
                audio=[mono],
                sample_rate=int(sample_rate),
                batch_size=batch_size,
                include_tensor_outputs=True,
                num_workers=0,
                verbose=False,
            )
            from score_sortformer_oracle_diarization import _parse_sortformer_segments

            predicted = _parse_sortformer_segments(raw_segments[0])
            probabilities = raw_probabilities[0].squeeze(0).detach().cpu().numpy()
        else:
            predicted = _run_sortformer(
                model,
                mono,
                sample_rate=int(sample_rate),
                batch_size=batch_size,
            )
        if probabilities is not None:
            mapping, cut_metrics = match_probability_speakers(
                cut.supervisions,
                probabilities,
                duration=cut.duration,
            )
            spans = [_span(supervision) for supervision in cut.supervisions]
        else:
            mapping, cut_metrics = match_predicted_speakers(
                cut.supervisions,
                predicted,
                duration=cut.duration,
                min_f1=min_match_f1,
            )
            spans = mapped_supervisions(cut.supervisions, predicted, mapping)
        if not spans:
            continue
        variant_id = f"{cut.id}-mask-sortformer"
        supervisions = [
            SupervisionSegment(
                id=f"{variant_id}-sup{index:04d}",
                recording_id=cut.recording_id,
                start=float(span["start"]),
                duration=max(0.01, float(span["end"]) - float(span["start"])),
                channel=cut.channel,
                text=str(span["text"]),
                speaker=str(span["speaker"]),
                language="en",
            )
            for index, span in enumerate(spans)
        ]
        custom = dict(cut.custom or {})
        custom.update(
            {
                "activity_mask_source": (
                    "mono-sortformer-soft" if probabilities is not None else "mono-sortformer"
                ),
                "sortformer_mapping_training_only": dict(mapping),
            }
        )
        if probabilities is not None:
            probability_path = probability_dir / f"{cut.id}.npy"
            np.save(probability_path, probabilities, allow_pickle=False)
            custom.update(
                {
                    "sortformer_probabilities_path": str(probability_path.resolve()),
                    "sortformer_slot_to_speaker": dict(mapping),
                    "supervision_times_are_training_metadata_only": True,
                }
            )
            probability_count += 1
        output.append(fastcopy(cut, id=variant_id, supervisions=supervisions, custom=custom))
        metrics.append(cut_metrics)
        if (cut_index + 1) % 25 == 0:
            print(f"processed {cut_index + 1} cuts", flush=True)

    if not output:
        raise RuntimeError("Sortformer produced no matched speaker masks")
    output_cuts.parent.mkdir(parents=True, exist_ok=True)
    CutSet.from_cuts(output).to_file(output_cuts)
    summary = {
        "input_cuts": str(input_cuts),
        "output_cuts": str(output_cuts),
        "output_cut_count": len(output),
        "model_name": model_name,
        "restore_path": str(restore_path) if restore_path else None,
        "min_match_f1": min_match_f1,
        "mean_reference_speaker_recall": sum(item["reference_speaker_recall"] for item in metrics)
        / len(metrics),
        "mean_matched_f1": sum(item["mean_matched_f1"] for item in metrics) / len(metrics),
        "streaming_config": streaming_config,
        "probability_files": probability_count,
        "probability_dir": str(probability_dir) if save_probabilities else None,
    }
    output_cuts.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace teacher activity with Sortformer masks predicted from mono training audio."
    )
    parser.add_argument("--input-cuts", type=Path, required=True)
    parser.add_argument("--output-cuts", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_SORTFORMER_MODEL)
    parser.add_argument("--restore-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-match-f1", type=float, default=0.10)
    parser.add_argument("--max-cuts", type=int, default=0)
    parser.add_argument("--chunk-len", type=int, default=340)
    parser.add_argument("--right-context", type=int, default=40)
    parser.add_argument("--fifo-len", type=int, default=40)
    parser.add_argument("--spkcache-update-period", type=int, default=340)
    parser.add_argument("--spkcache-len", type=int, default=188)
    parser.add_argument("--save-probabilities", action="store_true")
    args = parser.parse_args()
    build_sortformer_cutset(
        input_cuts=args.input_cuts,
        output_cuts=args.output_cuts,
        model_name=args.model_name,
        restore_path=args.restore_path,
        device=args.device,
        batch_size=args.batch_size,
        min_match_f1=args.min_match_f1,
        max_cuts=args.max_cuts,
        chunk_len=args.chunk_len,
        right_context=args.right_context,
        fifo_len=args.fifo_len,
        spkcache_update_period=args.spkcache_update_period,
        spkcache_len=args.spkcache_len,
        save_probabilities=args.save_probabilities,
    )


if __name__ == "__main__":
    main()
