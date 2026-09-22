from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SPEAKER_TAG_RE = re.compile(r"\[(S\d+)\]", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9']+", "", str(value or "").lower()).strip("'")


def _parse_tagged_tokens(text: str) -> list[dict]:
    current_speaker = "<untagged>"
    tokens: list[dict] = []
    parts = SPEAKER_TAG_RE.split(str(text or ""))
    for index, part in enumerate(parts):
        if index % 2 == 1:
            current_speaker = part.upper()
            continue
        for match in TOKEN_RE.finditer(part):
            token = _normalize_token(match.group(0))
            if token:
                tokens.append(
                    {
                        "token": token,
                        "speaker": current_speaker,
                        "raw": match.group(0),
                    }
                )
    return tokens


def _reference_tokens(words: Sequence[Mapping[str, object]]) -> list[dict]:
    tokens: list[dict] = []
    for index, word in enumerate(words):
        token = _normalize_token(word.get("normalized") or word.get("text"))
        if not token:
            continue
        item = dict(word)
        item["token"] = token
        item["index"] = index
        item["overlap"] = False
        tokens.append(item)
    for left in tokens:
        start = float(left.get("start") or 0.0)
        end = float(left.get("end") or start)
        speaker = str(left.get("speaker") or "")
        left["overlap"] = any(
            right is not left
            and str(right.get("speaker") or "") != speaker
            and float(right.get("start") or 0.0) < end
            and float(right.get("end") or 0.0) > start
            for right in tokens
        )
    return tokens


def _lcs_token_pairs(
    reference: Sequence[Mapping[str, object]], predicted: Sequence[Mapping[str, object]]
) -> list[tuple[int, int]]:
    rows = len(reference)
    cols = len(predicted)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    for row in range(rows - 1, -1, -1):
        ref_token = str(reference[row].get("token") or "")
        for col in range(cols - 1, -1, -1):
            if ref_token and ref_token == str(predicted[col].get("token") or ""):
                dp[row][col] = dp[row + 1][col + 1] + 1
            else:
                dp[row][col] = max(dp[row + 1][col], dp[row][col + 1])

    pairs: list[tuple[int, int]] = []
    row = 0
    col = 0
    while row < rows and col < cols:
        ref_token = str(reference[row].get("token") or "")
        if ref_token and ref_token == str(predicted[col].get("token") or ""):
            pairs.append((row, col))
            row += 1
            col += 1
        elif dp[row + 1][col] >= dp[row][col + 1]:
            row += 1
        else:
            col += 1
    return pairs


def _many_to_one_mapping(confusion: Mapping[str, Mapping[str, int]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for tag, counts in confusion.items():
        if not counts:
            continue
        mapping[tag] = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    return mapping


def _one_to_one_mapping(confusion: Mapping[str, Mapping[str, int]]) -> dict[str, str]:
    tags = sorted(confusion)
    speakers = sorted({speaker for counts in confusion.values() for speaker in counts})
    if not tags or not speakers:
        return {}
    best_score = -1
    best_mapping: dict[str, str] = {}
    for speaker_subset in itertools.permutations(speakers, min(len(tags), len(speakers))):
        mapping = dict(zip(tags, speaker_subset, strict=False))
        score = sum(int(confusion[tag].get(speaker, 0)) for tag, speaker in mapping.items())
        if score > best_score:
            best_score = score
            best_mapping = mapping
    return best_mapping


def _score_clip(reference_words: Sequence[Mapping[str, object]], predicted_text: str) -> dict:
    reference = _reference_tokens(reference_words)
    predicted = _parse_tagged_tokens(predicted_text)
    pairs = _lcs_token_pairs(reference, predicted)

    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for ref_index, pred_index in pairs:
        tag = str(predicted[pred_index].get("speaker") or "<untagged>")
        speaker = str(reference[ref_index].get("speaker") or "")
        confusion[tag][speaker] += 1

    many_mapping = _many_to_one_mapping(confusion)
    one_mapping = _one_to_one_mapping(confusion)

    def mapped_counts(mapping: Mapping[str, str]) -> dict[str, int]:
        correct = 0
        overlap_correct = 0
        non_overlap_correct = 0
        for ref_index, pred_index in pairs:
            tag = str(predicted[pred_index].get("speaker") or "<untagged>")
            reference_speaker = str(reference[ref_index].get("speaker") or "")
            is_correct = mapping.get(tag) == reference_speaker
            if is_correct:
                correct += 1
                if reference[ref_index].get("overlap"):
                    overlap_correct += 1
                else:
                    non_overlap_correct += 1
        return {
            "correct": correct,
            "overlap_correct": overlap_correct,
            "non_overlap_correct": non_overlap_correct,
        }

    many_counts = mapped_counts(many_mapping)
    one_counts = mapped_counts(one_mapping)
    reference_count = len(reference)
    predicted_count = len(predicted)
    matched_count = len(pairs)
    overlap_count = sum(1 for item in reference if item.get("overlap"))
    non_overlap_count = reference_count - overlap_count

    return {
        "reference_words": reference_count,
        "predicted_words": predicted_count,
        "lexical_matched_words": matched_count,
        "lexical_coverage": matched_count / reference_count if reference_count else 0.0,
        "predicted_tags": sorted({str(item.get("speaker") or "") for item in predicted}),
        "reference_speakers": sorted({str(item.get("speaker") or "") for item in reference}),
        "overlap_words": overlap_count,
        "non_overlap_words": non_overlap_count,
        "confusion": {tag: dict(counts) for tag, counts in sorted(confusion.items())},
        "many_to_one_mapping": dict(sorted(many_mapping.items())),
        "one_to_one_mapping": dict(sorted(one_mapping.items())),
        "many_to_one_correct_words": many_counts["correct"],
        "many_to_one_accuracy": (
            many_counts["correct"] / reference_count if reference_count else 0.0
        ),
        "many_to_one_matched_accuracy": (
            many_counts["correct"] / matched_count if matched_count else 0.0
        ),
        "many_to_one_prediction_precision_proxy": (
            many_counts["correct"] / predicted_count if predicted_count else 0.0
        ),
        "many_to_one_overlap_correct_words": many_counts["overlap_correct"],
        "many_to_one_overlap_accuracy": (
            many_counts["overlap_correct"] / overlap_count if overlap_count else 0.0
        ),
        "many_to_one_non_overlap_correct_words": many_counts["non_overlap_correct"],
        "many_to_one_non_overlap_accuracy": (
            many_counts["non_overlap_correct"] / non_overlap_count if non_overlap_count else 0.0
        ),
        "one_to_one_correct_words": one_counts["correct"],
        "one_to_one_accuracy": one_counts["correct"] / reference_count if reference_count else 0.0,
        "one_to_one_matched_accuracy": (
            one_counts["correct"] / matched_count if matched_count else 0.0
        ),
        "one_to_one_prediction_precision_proxy": (
            one_counts["correct"] / predicted_count if predicted_count else 0.0
        ),
        "one_to_one_overlap_correct_words": one_counts["overlap_correct"],
        "one_to_one_overlap_accuracy": (
            one_counts["overlap_correct"] / overlap_count if overlap_count else 0.0
        ),
        "one_to_one_non_overlap_correct_words": one_counts["non_overlap_correct"],
        "one_to_one_non_overlap_accuracy": (
            one_counts["non_overlap_correct"] / non_overlap_count if non_overlap_count else 0.0
        ),
    }


def _aggregate(groups: Sequence[Mapping[str, object]]) -> dict:
    totals = Counter()
    for group in groups:
        for key in [
            "reference_words",
            "predicted_words",
            "lexical_matched_words",
            "overlap_words",
            "non_overlap_words",
            "many_to_one_correct_words",
            "many_to_one_overlap_correct_words",
            "many_to_one_non_overlap_correct_words",
            "one_to_one_correct_words",
            "one_to_one_overlap_correct_words",
            "one_to_one_non_overlap_correct_words",
        ]:
            totals[key] += int(group.get(key) or 0)
    reference_words = totals["reference_words"]
    predicted_words = totals["predicted_words"]
    lexical_matched = totals["lexical_matched_words"]
    overlap_words = totals["overlap_words"]
    non_overlap_words = totals["non_overlap_words"]
    return {
        "clip_count": len(groups),
        "primary_metric": "many_to_one_accuracy",
        **dict(totals),
        "lexical_coverage": lexical_matched / reference_words if reference_words else 0.0,
        "many_to_one_accuracy": (
            totals["many_to_one_correct_words"] / reference_words if reference_words else 0.0
        ),
        "many_to_one_matched_accuracy": (
            totals["many_to_one_correct_words"] / lexical_matched if lexical_matched else 0.0
        ),
        "many_to_one_prediction_precision_proxy": (
            totals["many_to_one_correct_words"] / predicted_words if predicted_words else 0.0
        ),
        "many_to_one_overlap_accuracy": (
            totals["many_to_one_overlap_correct_words"] / overlap_words if overlap_words else 0.0
        ),
        "many_to_one_non_overlap_accuracy": (
            totals["many_to_one_non_overlap_correct_words"] / non_overlap_words
            if non_overlap_words
            else 0.0
        ),
        "one_to_one_accuracy": (
            totals["one_to_one_correct_words"] / reference_words if reference_words else 0.0
        ),
        "one_to_one_matched_accuracy": (
            totals["one_to_one_correct_words"] / lexical_matched if lexical_matched else 0.0
        ),
        "one_to_one_prediction_precision_proxy": (
            totals["one_to_one_correct_words"] / predicted_words if predicted_words else 0.0
        ),
        "one_to_one_overlap_accuracy": (
            totals["one_to_one_overlap_correct_words"] / overlap_words if overlap_words else 0.0
        ),
        "one_to_one_non_overlap_accuracy": (
            totals["one_to_one_non_overlap_correct_words"] / non_overlap_words
            if non_overlap_words
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score speaker-tagged ASR text against forced-reference words."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows = {str(row["clip_id"]): row for row in _read_jsonl(args.manifest)}
    predictions = {str(row["clip_id"]): row for row in _read_jsonl(args.predictions_jsonl)}
    missing = sorted(set(manifest_rows) - set(predictions))
    if missing:
        raise ValueError(f"Missing predictions for clips: {missing}")

    group_rows = []
    for clip_id, manifest_row in sorted(manifest_rows.items()):
        prediction = predictions[clip_id]
        score = _score_clip(manifest_row.get("words") or [], str(prediction.get("text") or ""))
        score.update(
            {
                "clip_id": clip_id,
                "session": manifest_row.get("session"),
                "relative_start": manifest_row.get("relative_start"),
                "relative_end": manifest_row.get("relative_end"),
                "audio_path": manifest_row.get("audio_path"),
                "raw_text": prediction.get("text") or "",
            }
        )
        group_rows.append(score)

    summary = _aggregate(group_rows)
    summary.update(
        {
            "manifest": str(args.manifest),
            "predictions_jsonl": str(args.predictions_jsonl),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "speaker_tagged_asr_groups.jsonl", group_rows)
    (args.output_dir / "speaker_tagged_asr_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
