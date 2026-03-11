from __future__ import annotations

import copy
from typing import List, Sequence

from .cli import _set_segment_speaker_label


def smooth_short_speaker_flips(
    segments: Sequence[dict],
    *,
    max_total_duration: float = 2.0,
    max_gap: float = 0.75,
    max_score: float | None = None,
    max_run_segments: int = 1,
) -> List[dict]:
    relabeled = copy.deepcopy(list(segments))
    if len(relabeled) < 3:
        return relabeled

    def _duration(start_index: int, end_index: int) -> float:
        return float(relabeled[end_index]["end"]) - float(relabeled[start_index]["start"])

    def _speaker(index: int) -> str:
        return str(relabeled[index].get("speaker") or "unknown")

    def _score(index: int) -> float | None:
        value = relabeled[index].get("speaker_match_score")
        return float(value) if value is not None else None

    def _gaps_ok(start_index: int, end_index: int) -> bool:
        previous = relabeled[start_index - 1]
        following = relabeled[end_index + 1]
        if float(relabeled[start_index]["start"]) - float(previous["end"]) > max_gap:
            return False
        if float(following["start"]) - float(relabeled[end_index]["end"]) > max_gap:
            return False
        for index in range(start_index, end_index):
            if float(relabeled[index + 1]["start"]) - float(relabeled[index]["end"]) > max_gap:
                return False
        return True

    index = 1
    while index < len(relabeled) - 1:
        matched = False
        for run_length in range(min(max_run_segments, len(relabeled) - index - 1), 0, -1):
            start_index = index
            end_index = index + run_length - 1
            if end_index >= len(relabeled) - 1:
                continue
            left_speaker = _speaker(start_index - 1)
            right_speaker = _speaker(end_index + 1)
            if left_speaker != right_speaker or left_speaker == "unknown":
                continue
            run_speakers = {_speaker(item) for item in range(start_index, end_index + 1)}
            if left_speaker in run_speakers:
                continue
            if _duration(start_index, end_index) > max_total_duration:
                continue
            if not _gaps_ok(start_index, end_index):
                continue
            if max_score is not None:
                run_scores = [_score(item) for item in range(start_index, end_index + 1)]
                finite_scores = [score for score in run_scores if score is not None]
                if finite_scores and max(finite_scores) > max_score:
                    continue
            for item in range(start_index, end_index + 1):
                _set_segment_speaker_label(relabeled[item], left_speaker)
                relabeled[item]["speaker_postprocess_source"] = "smooth_short_speaker_flips"
            index = end_index + 1
            matched = True
            break
        if not matched:
            index += 1

    return relabeled
