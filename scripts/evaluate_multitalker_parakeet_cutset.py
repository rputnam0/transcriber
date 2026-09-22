from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from build_multitalker_parakeet_drive_dataset import activity_statistics
from run_multitalker_parakeet import (
    DEFAULT_ASR_MODEL,
    _build_config,
    _configure_diarization_model,
    _load_asr_model,
    _load_diarization_model,
    _set_oracle_rttm_lines_mask,
    _stream_audio,
)
from score_multitalker_parakeet import _tokens, score


def reference_words_from_supervisions(supervisions: Sequence[object]) -> list[dict]:
    words = []
    for supervision in supervisions:
        if isinstance(supervision, Mapping):
            speaker = str(supervision["speaker"])
            start = float(supervision["start"])
            duration = float(
                supervision.get(
                    "duration",
                    float(supervision.get("end", start)) - start,
                )
            )
            text = str(supervision.get("text") or "")
        else:
            speaker = str(supervision.speaker)
            start = float(supervision.start)
            duration = float(supervision.duration)
            text = str(supervision.text or "")
        tokens = _tokens(text)
        for index, token in enumerate(tokens):
            word_start = start + duration * index / max(1, len(tokens))
            word_end = start + duration * (index + 1) / max(1, len(tokens))
            words.append(
                {
                    "speaker": speaker,
                    "start": word_start,
                    "end": word_end,
                    "text": token,
                    "normalized": token,
                }
            )
    return sorted(words, key=lambda item: (item["start"], item["end"], item["speaker"]))


def oracle_rttm_lines_from_supervisions(
    supervisions: Sequence[object], *, recording_id: str
) -> list[str]:
    speakers = sorted({str(supervision.speaker) for supervision in supervisions})
    speaker_ids = {speaker: f"speaker_{index:02d}" for index, speaker in enumerate(speakers)}
    return [
        " ".join(
            [
                "SPEAKER",
                recording_id,
                "1",
                f"{float(supervision.start):.3f}",
                f"{float(supervision.duration):.3f}",
                "<NA>",
                "<NA>",
                speaker_ids[str(supervision.speaker)],
                "<NA>",
                "<NA>",
            ]
        )
        for supervision in supervisions
        if float(supervision.duration) > 0.0
    ]


def summarize_scores(rows: Iterable[Mapping[str, object]]) -> dict:
    rows = list(rows)
    reference_words = sum(int(row["reference_words"]) for row in rows)
    predicted_words = sum(int(row["predicted_words"]) for row in rows)
    matched_words = sum(int(row["oracle_one_to_one_matched_words"]) for row in rows)
    return {
        "cuts": len(rows),
        "reference_words": reference_words,
        "predicted_words": predicted_words,
        "oracle_one_to_one_matched_words": matched_words,
        "oracle_one_to_one_attributed_word_recall": (
            matched_words / reference_words if reference_words else 0.0
        ),
        "oracle_one_to_one_prediction_precision": (
            matched_words / predicted_words if predicted_words else 0.0
        ),
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Sortformer-conditioned multitalker Parakeet on a mono CutSet."
    )
    parser.add_argument("--cuts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diar-model", required=True)
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument(
        "--conditioning",
        choices=("kernel", "masked", "masked-preencode"),
        default="kernel",
    )
    parser.add_argument("--max-cuts", type=int, default=0)
    parser.add_argument("--max-speakers", type=int, default=4)
    parser.add_argument("--att-context-size", type=int, nargs=2, default=(70, 13))
    parser.add_argument("--diar-chunk-length", type=int, default=6)
    parser.add_argument("--diar-right-context", type=int, default=7)
    parser.add_argument("--speaker-cache-length", type=int, default=188)
    parser.add_argument("--fifo-length", type=int, default=188)
    parser.add_argument("--cache-gating-buffer-size", type=int, default=2)
    parser.add_argument("--disable-cache-gating", action="store_true")
    parser.add_argument("--binary-diarization", action="store_true")
    parser.add_argument(
        "--oracle-supervision-mask",
        action="store_true",
        help="Diagnostic only: replace mono Sortformer activity with held-out labels.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    import torch
    from lhotse import CutSet

    if not torch.cuda.is_available():
        raise RuntimeError("Multitalker Parakeet CutSet evaluation requires CUDA.")
    args.audio = Path("unused.wav")
    args.output = args.output_dir / "unused.json"
    cfg = _build_config(args)
    device = torch.device("cuda")
    diar_model = _configure_diarization_model(
        _load_diarization_model(args.diar_model, device).to(device), cfg
    )
    asr_model = _load_asr_model(args.asr_model, device).eval().to(device)
    asr_model.encoder.set_default_att_context_size(att_context_size=cfg.att_context_size)

    results = []
    for index, cut in enumerate(CutSet.from_file(args.cuts)):
        if args.max_cuts > 0 and index >= args.max_cuts:
            break
        cfg.audio_file = str(cut.recording.sources[0].source)
        if args.oracle_supervision_mask:
            _set_oracle_rttm_lines_mask(
                diar_model=diar_model,
                rttm_lines=oracle_rttm_lines_from_supervisions(
                    cut.supervisions, recording_id=cut.recording.id
                ),
                offset=0.0,
                duration=cut.duration,
                max_speakers=args.max_speakers,
                collar_seconds=0.0,
                device=device,
            )
        segments = _stream_audio(cfg=cfg, asr_model=asr_model, diar_model=diar_model)
        result = score(reference_words_from_supervisions(cut.supervisions), segments)
        spans = [
            {
                "speaker": supervision.speaker,
                "start": supervision.start,
                "end": supervision.end,
            }
            for supervision in cut.supervisions
        ]
        activity = activity_statistics(spans, duration=cut.duration)
        result.update(
            {
                "cut_id": cut.id,
                "audio": cfg.audio_file,
                "has_overlap": activity["overlap_seconds"] > 0,
                "overlap_speech_fraction": activity["overlap_speech_fraction"],
                "segments": segments,
            }
        )
        results.append(result)
        if (index + 1) % 10 == 0:
            print(f"evaluated {index + 1} cuts", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "cut_results.jsonl", results)
    summary = {
        "primary_metric": "oracle_one_to_one_attributed_word_recall",
        "name_binding": "oracle-per-cut; production name binding is not measured here",
        "cuts_path": str(args.cuts),
        "asr_model": str(args.asr_model),
        "diar_model": str(args.diar_model),
        "conditioning": args.conditioning,
        "speaker_activity": (
            "oracle-held-out-supervisions-diagnostic-only"
            if args.oracle_supervision_mask
            else "mono-sortformer"
        ),
        "all": summarize_scores(results),
        "overlap_clips": summarize_scores(row for row in results if row["has_overlap"]),
        "non_overlap_clips": summarize_scores(row for row in results if not row["has_overlap"]),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
