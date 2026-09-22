from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence


def session_from_cut_id(cut_id: object) -> str:
    match = re.match(r"session_(\d+)_", str(cut_id or ""))
    if not match:
        raise ValueError(f"Cannot infer session from cut id {cut_id!r}")
    return f"Session {int(match.group(1))}"


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 0 or length <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def _window_and_offset(cut_id: object) -> tuple[str, int]:
    match = re.match(r"session_\d+_w(\d+)_c(\d+)", str(cut_id or ""))
    if not match:
        raise ValueError(f"Cannot infer window and offset from cut id {cut_id!r}")
    return match.group(1), int(match.group(2))


def select_balanced(
    cuts: Sequence[object],
    *,
    cuts_per_session: int,
    excluded_cut_ids: set[str] | None = None,
) -> list[object]:
    excluded_cut_ids = excluded_cut_ids or set()
    grouped: dict[str, list[object]] = defaultdict(list)
    for cut in cuts:
        if str(cut.id) in excluded_cut_ids:
            continue
        grouped[session_from_cut_id(cut.id)].append(cut)
    selected = []
    for session in sorted(grouped):
        session_cuts = sorted(grouped[session], key=lambda cut: cut.id)
        selected.extend(
            session_cuts[index]
            for index in evenly_spaced_indices(len(session_cuts), cuts_per_session)
        )
    return selected


def select_contiguous(
    cuts: Sequence[object],
    *,
    cuts_per_session: int,
    excluded_cut_ids: set[str] | None = None,
) -> list[object]:
    excluded_cut_ids = excluded_cut_ids or set()
    grouped: dict[str, dict[str, list[object]]] = defaultdict(lambda: defaultdict(list))
    for cut in cuts:
        if str(cut.id) in excluded_cut_ids:
            continue
        window, _offset = _window_and_offset(cut.id)
        grouped[session_from_cut_id(cut.id)][window].append(cut)
    selected = []
    for session in sorted(grouped):
        candidates = []
        for window, window_cuts in grouped[session].items():
            ordered = sorted(window_cuts, key=lambda cut: _window_and_offset(cut.id)[1])
            runs: list[list[object]] = []
            for cut in ordered:
                if (
                    runs
                    and _window_and_offset(cut.id)[1] - _window_and_offset(runs[-1][-1].id)[1]
                    == 30_000
                ):
                    runs[-1].append(cut)
                else:
                    runs.append([cut])
            candidates.extend((run, window) for run in runs)
        run, _window = min(
            candidates,
            key=lambda item: (-len(item[0]), item[1], item[0][0].id),
        )
        start = max(0, (len(run) - cuts_per_session) // 2)
        selected.extend(run[start : start + cuts_per_session])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze an evenly sampled mono evaluation slice from every held-out session."
    )
    parser.add_argument("--input-cuts", type=Path, required=True)
    parser.add_argument("--output-cuts", type=Path, required=True)
    parser.add_argument(
        "--exclude-cuts",
        type=Path,
        action="append",
        default=[],
        help="CutSet whose cut IDs must not appear in this evaluation slice; repeatable.",
    )
    parser.add_argument("--cuts-per-session", type=int, default=10)
    parser.add_argument(
        "--selection-mode",
        choices=("evenly-spaced", "contiguous-window"),
        default="evenly-spaced",
    )
    args = parser.parse_args()
    if args.cuts_per_session < 1:
        parser.error("--cuts-per-session must be positive")

    from lhotse import CutSet

    source = list(CutSet.from_file(args.input_cuts))
    excluded_cut_ids = {
        str(cut.id) for cutset_path in args.exclude_cuts for cut in CutSet.from_file(cutset_path)
    }
    selected = (
        select_contiguous(
            source,
            cuts_per_session=args.cuts_per_session,
            excluded_cut_ids=excluded_cut_ids,
        )
        if args.selection_mode == "contiguous-window"
        else select_balanced(
            source,
            cuts_per_session=args.cuts_per_session,
            excluded_cut_ids=excluded_cut_ids,
        )
    )
    args.output_cuts.parent.mkdir(parents=True, exist_ok=True)
    CutSet.from_cuts(selected).to_file(args.output_cuts)
    by_session = defaultdict(list)
    for cut in selected:
        by_session[session_from_cut_id(cut.id)].append(cut.id)
    summary = {
        "input_cuts": str(args.input_cuts),
        "output_cuts": str(args.output_cuts),
        "exclude_cuts": [str(path) for path in args.exclude_cuts],
        "excluded_cut_count": len(excluded_cut_ids),
        "selection_rule": (
            "longest-contiguous-window-run-before-model-inference"
            if args.selection_mode == "contiguous-window"
            else "evenly-spaced-by-sorted-cut-id-before-model-inference"
        ),
        "cuts_per_session": args.cuts_per_session,
        "cut_count": len(selected),
        "sessions": dict(sorted(by_session.items())),
    }
    args.output_cuts.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
