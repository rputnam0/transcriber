from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9']+", "", str(value or "").lower()).strip("'")


def _tokens(value: object) -> list[str]:
    return [
        token
        for match in TOKEN_RE.finditer(str(value or ""))
        if (token := _normalize_token(match.group(0)))
    ]


def _lcs_pairs(reference: Sequence[str], predicted: Sequence[str]) -> list[tuple[int, int]]:
    rows = len(reference)
    cols = len(predicted)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    for row in range(rows - 1, -1, -1):
        for col in range(cols - 1, -1, -1):
            if reference[row] == predicted[col]:
                dp[row][col] = dp[row + 1][col + 1] + 1
            else:
                dp[row][col] = max(dp[row + 1][col], dp[row][col + 1])

    pairs = []
    row = 0
    col = 0
    while row < rows and col < cols:
        if reference[row] == predicted[col]:
            pairs.append((row, col))
            row += 1
            col += 1
        elif dp[row + 1][col] >= dp[row][col + 1]:
            row += 1
        else:
            col += 1
    return pairs


def _load_segments(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return [dict(row) for row in json.loads(text)]
    return [dict(row) for row in _read_jsonl(path)]


def _prediction_streams(segments: Sequence[Mapping[str, object]]) -> dict[str, list[str]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for segment in segments:
        speaker = str(segment.get("speaker") or "<unknown>")
        grouped.setdefault(speaker, []).append(segment)
    return {
        speaker: [
            token
            for segment in sorted(
                rows,
                key=lambda item: (
                    float(item.get("start_time") or 0.0),
                    float(item.get("end_time") or 0.0),
                ),
            )
            for token in _tokens(segment.get("words") or segment.get("text"))
        ]
        for speaker, rows in sorted(grouped.items())
    }


def _overlap_flags(words: Sequence[Mapping[str, object]]) -> list[bool]:
    flags = []
    for index, word in enumerate(words):
        start = float(word.get("start") or 0.0)
        end = float(word.get("end") or start)
        speaker = str(word.get("speaker") or "")
        flags.append(
            any(
                other_index != index
                and str(other.get("speaker") or "") != speaker
                and float(other.get("start") or 0.0) < end
                and float(other.get("end") or 0.0) > start
                for other_index, other in enumerate(words)
            )
        )
    return flags


def _reference_streams(
    words: Sequence[Mapping[str, object]],
) -> tuple[dict[str, list[str]], dict[str, list[bool]]]:
    overlap = _overlap_flags(words)
    tokens: dict[str, list[str]] = {}
    overlap_by_speaker: dict[str, list[bool]] = {}
    for index, word in enumerate(words):
        token = _normalize_token(word.get("normalized") or word.get("text"))
        if not token:
            continue
        speaker = str(word.get("speaker") or "<unknown>")
        tokens.setdefault(speaker, []).append(token)
        overlap_by_speaker.setdefault(speaker, []).append(overlap[index])
    return tokens, overlap_by_speaker


def _best_one_to_one_mapping(
    pair_matches: Mapping[tuple[str, str], int],
    predicted_speakers: Sequence[str],
    reference_speakers: Sequence[str],
) -> dict[str, str]:
    best_score = -1
    best_mapping: dict[str, str] = {}
    max_pairs = min(len(predicted_speakers), len(reference_speakers))
    for pair_count in range(1, max_pairs + 1):
        for predicted_subset in itertools.combinations(predicted_speakers, pair_count):
            for reference_order in itertools.permutations(reference_speakers, pair_count):
                mapping = dict(zip(predicted_subset, reference_order, strict=True))
                score = sum(
                    pair_matches.get((predicted, reference), 0)
                    for predicted, reference in mapping.items()
                )
                if score > best_score or (score == best_score and len(mapping) > len(best_mapping)):
                    best_score = score
                    best_mapping = mapping
    return best_mapping


def score(
    reference_words: Sequence[Mapping[str, object]],
    predicted_segments: Sequence[Mapping[str, object]],
) -> dict:
    reference, overlap_flags = _reference_streams(reference_words)
    predicted = _prediction_streams(predicted_segments)
    pair_details = {}
    pair_matches = {}
    for predicted_speaker, predicted_tokens in predicted.items():
        for reference_speaker, reference_tokens in reference.items():
            pairs = _lcs_pairs(reference_tokens, predicted_tokens)
            matched_reference = [pair[0] for pair in pairs]
            overlap_matches = sum(
                overlap_flags[reference_speaker][index] for index in matched_reference
            )
            key = (predicted_speaker, reference_speaker)
            pair_matches[key] = len(pairs)
            pair_details[f"{predicted_speaker} -> {reference_speaker}"] = {
                "matched_words": len(pairs),
                "overlap_matched_words": overlap_matches,
                "predicted_words": len(predicted_tokens),
                "reference_words": len(reference_tokens),
            }

    mapping = _best_one_to_one_mapping(
        pair_matches,
        sorted(predicted),
        sorted(reference),
    )
    matched = sum(
        pair_matches[(predicted_speaker, reference_speaker)]
        for predicted_speaker, reference_speaker in mapping.items()
    )
    overlap_matched = sum(
        pair_details[f"{predicted_speaker} -> {reference_speaker}"]["overlap_matched_words"]
        for predicted_speaker, reference_speaker in mapping.items()
    )
    reference_count = sum(len(tokens) for tokens in reference.values())
    predicted_count = sum(len(tokens) for tokens in predicted.values())
    overlap_count = sum(sum(flags) for flags in overlap_flags.values())
    non_overlap_count = reference_count - overlap_count
    non_overlap_matched = matched - overlap_matched
    return {
        "primary_metric": "oracle_one_to_one_attributed_word_recall",
        "reference_words": reference_count,
        "predicted_words": predicted_count,
        "reference_speakers": sorted(reference),
        "predicted_speakers": sorted(predicted),
        "oracle_one_to_one_mapping": dict(sorted(mapping.items())),
        "oracle_one_to_one_matched_words": matched,
        "oracle_one_to_one_attributed_word_recall": (
            matched / reference_count if reference_count else 0.0
        ),
        "oracle_one_to_one_prediction_precision": (
            matched / predicted_count if predicted_count else 0.0
        ),
        "overlap_reference_words": overlap_count,
        "overlap_matched_words": overlap_matched,
        "overlap_attributed_word_recall": overlap_matched / overlap_count if overlap_count else 1.0,
        "non_overlap_reference_words": non_overlap_count,
        "non_overlap_matched_words": non_overlap_matched,
        "non_overlap_attributed_word_recall": (
            non_overlap_matched / non_overlap_count if non_overlap_count else 1.0
        ),
        "pair_details": pair_details,
    }


def _load_reference_words(
    path: Path,
    *,
    session: str,
    window_start: float,
    chunk_start: float,
    chunk_duration: float | None,
) -> list[dict]:
    group = next(
        (
            row
            for row in _read_jsonl(path)
            if str(row.get("session")) == session
            and abs(float(row.get("window_start") or 0.0) - window_start) < 1e-3
        ),
        None,
    )
    if group is None:
        raise ValueError(f"Reference group not found for {session} at {window_start:.3f}")
    chunk_end = None if chunk_duration is None else chunk_start + chunk_duration
    selected = []
    for word in group.get("words") or []:
        midpoint = (float(word.get("start") or 0.0) + float(word.get("end") or 0.0)) / 2.0
        if midpoint < chunk_start or (chunk_end is not None and midpoint >= chunk_end):
            continue
        item = dict(word)
        item["start"] = float(item.get("start") or 0.0) - chunk_start
        item["end"] = float(item.get("end") or 0.0) - chunk_start
        selected.append(item)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score multitalker Parakeet streams against named forced-word references."
    )
    parser.add_argument("--reference-groups", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--window-start", type=float, required=True)
    parser.add_argument("--chunk-start", type=float, default=0.0)
    parser.add_argument("--chunk-duration", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference_words = _load_reference_words(
        args.reference_groups,
        session=args.session,
        window_start=args.window_start,
        chunk_start=args.chunk_start,
        chunk_duration=args.chunk_duration,
    )
    summary = score(reference_words, _load_segments(args.predictions))
    summary.update(
        {
            "reference_groups": str(args.reference_groups),
            "predictions": str(args.predictions),
            "session": args.session,
            "window_start": args.window_start,
            "chunk_start": args.chunk_start,
            "chunk_duration": args.chunk_duration,
        }
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
