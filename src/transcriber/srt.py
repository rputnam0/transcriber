from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple


def _fmt_srt_ts(seconds: float) -> str:
    """Format seconds as SRT timestamp with correct rounding and carry.

    Avoids producing 1000 milliseconds by rounding the total milliseconds and carrying
    into seconds/minutes/hours as needed.
    """
    total_ms = int(round(max(seconds, 0.0) * 1000.0))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: Path, items: Iterable[Tuple[int, float, float, str]]) -> None:
    """Write a basic SRT file.
    items: iterable of (index, start_sec, end_sec, text)
    """
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        for idx, start, end, text in items:
            f.write(f"{idx}\n{_fmt_srt_ts(start)} --> {_fmt_srt_ts(end)}\n{text}\n\n")
