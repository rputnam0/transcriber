from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SHARE_BUCKETS = (
    ("lt_030", 0.0, 0.30),
    ("030_045", 0.30, 0.45),
    ("045_060", 0.45, 0.60),
    ("060_075", 0.60, 0.75),
    ("075_090", 0.75, 0.90),
    ("090_100", 0.90, 1.01),
)


@dataclass
class DurationCounter:
    count: int = 0
    seconds: float = 0.0

    def add(self, seconds: float) -> None:
        self.count += 1
        self.seconds += max(float(seconds), 0.0)

    def as_dict(self) -> Dict[str, float | int]:
        return {
            "count": self.count,
            "seconds": round(self.seconds, 3),
            "hours": round(self.seconds / 3600.0, 4),
        }


@dataclass
class EvalWindowSummary:
    window: str
    session: str
    duration_seconds: float
    speaker_count: int
    word_count: int
    words_by_speaker: Dict[str, int]
    words_by_active_speakers: Dict[str, int]
    words_by_target_share: Dict[str, int]


@dataclass
class CorpusInventory:
    prod_root: str
    prepared_roots: List[str]
    clean_bank: Dict[str, Any] = field(default_factory=dict)
    mixed_candidate_pool: Dict[str, Any] = field(default_factory=dict)
    hard_negatives: Dict[str, Any] = field(default_factory=dict)
    prepared_eval: Dict[str, Any] = field(default_factory=dict)
    split_risks: List[str] = field(default_factory=list)
    next_manifest_requirements: List[str] = field(default_factory=list)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _speaker_key(value: object) -> str:
    return str(value or "unknown")


def _share_bucket(value: float) -> str:
    for name, low, high in SHARE_BUCKETS:
        if low <= value < high:
            return name
    return "unknown"


def _duration(record: Mapping[str, object]) -> float:
    if record.get("duration") is not None:
        return float(record["duration"])
    return max(float(record.get("end") or 0.0) - float(record.get("start") or 0.0), 0.0)


def _latest_dataset_root(prod_root: Path, family: str) -> Path | None:
    family_root = prod_root / "artifacts" / "datasets" / family
    if not family_root.exists():
        return None
    candidates = [path for path in family_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _dataset_roots(prod_root: Path, family: str) -> List[Path]:
    family_root = prod_root / "artifacts" / "datasets" / family
    if not family_root.exists():
        return []
    return sorted(path for path in family_root.iterdir() if path.is_dir())


def _latest_bank_root(prod_root: Path) -> Path | None:
    family_root = prod_root / "artifacts" / "bank"
    if not family_root.exists():
        return None
    candidates = [path for path in family_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _counter_dict(counter: Mapping[str, int]) -> Dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def _duration_dict(counters: Mapping[str, DurationCounter]) -> Dict[str, Dict[str, float | int]]:
    return {key: value.as_dict() for key, value in sorted(counters.items())}


def _summarize_quality_records(path: Path) -> Dict[str, Any]:
    by_speaker: Dict[str, DurationCounter] = defaultdict(DurationCounter)
    by_session: Dict[str, DurationCounter] = defaultdict(DurationCounter)
    by_source: Dict[str, DurationCounter] = defaultdict(DurationCounter)
    rejected = Counter()
    total = DurationCounter()
    accepted = DurationCounter()
    for record in _read_jsonl(path):
        seconds = _duration(record)
        total.add(seconds)
        rejection = record.get("qa_rejection")
        if rejection:
            rejected[str(rejection)] += 1
            continue
        accepted.add(seconds)
        speaker = _speaker_key(record.get("speaker"))
        session = str(record.get("session") or "unknown")
        source = str(record.get("source") or "unknown")
        by_speaker[speaker].add(seconds)
        by_session[session].add(seconds)
        by_source[source].add(seconds)
    return {
        "path": str(path),
        "total": total.as_dict(),
        "accepted": accepted.as_dict(),
        "by_speaker": _duration_dict(by_speaker),
        "by_session": _duration_dict(by_session),
        "by_source": _duration_dict(by_source),
        "rejections": _counter_dict(rejected),
    }


def _summarize_candidate_pool(path: Path) -> Dict[str, Any]:
    total = DurationCounter()
    accepted = DurationCounter()
    rejected = DurationCounter()
    by_speaker: Dict[str, DurationCounter] = defaultdict(DurationCounter)
    by_session: Dict[str, DurationCounter] = defaultdict(DurationCounter)
    by_active: Dict[str, DurationCounter] = defaultdict(DurationCounter)
    by_share: Dict[str, DurationCounter] = defaultdict(DurationCounter)
    by_bucket: Dict[str, DurationCounter] = defaultdict(DurationCounter)
    rejections = Counter()
    accepted_records = 0
    for record in _read_jsonl(path):
        seconds = _duration(record)
        total.add(seconds)
        accepted_flag = bool(record.get("accepted"))
        if accepted_flag:
            accepted.add(seconds)
            accepted_records += 1
        else:
            rejected.add(seconds)
            rejections[
                str(record.get("rejection") or record.get("purity_rejection") or "unknown")
            ] += 1
        speaker = _speaker_key(record.get("speaker"))
        session = str(record.get("session") or "unknown")
        active = str(int(record.get("active_speakers") or 0))
        share = float(record.get("dominant_share") or 0.0)
        bucket = str(record.get("bucket") or _share_bucket(share))
        by_speaker[speaker].add(seconds)
        by_session[session].add(seconds)
        by_active[active].add(seconds)
        by_share[_share_bucket(share)].add(seconds)
        by_bucket[bucket].add(seconds)
    return {
        "path": str(path),
        "total": total.as_dict(),
        "accepted": accepted.as_dict(),
        "rejected": rejected.as_dict(),
        "accepted_records": accepted_records,
        "by_speaker": _duration_dict(by_speaker),
        "by_session": _duration_dict(by_session),
        "by_active_speakers": _duration_dict(by_active),
        "by_target_share": _duration_dict(by_share),
        "by_source_bucket": _duration_dict(by_bucket),
        "rejections": _counter_dict(rejections),
    }


def _summarize_hard_negatives(path: Path) -> Dict[str, Any]:
    by_speaker = Counter()
    by_partner = Counter()
    by_pair = Counter()
    by_session = Counter()
    by_active = Counter()
    by_share = Counter()
    total = DurationCounter()
    for record in _read_jsonl(path):
        speaker = _speaker_key(record.get("speaker"))
        partner = _speaker_key(record.get("confusion_partner"))
        pair = "::".join(str(item) for item in (record.get("tracked_pair") or [speaker, partner]))
        seconds = _duration(record)
        by_speaker[speaker] += 1
        by_partner[partner] += 1
        by_pair[pair] += 1
        by_session[str(record.get("source_session") or "unknown")] += 1
        by_active[str(int(record.get("active_speakers") or 0))] += 1
        by_share[_share_bucket(float(record.get("dominant_share") or 0.0))] += 1
        total.add(seconds)
    return {
        "path": str(path),
        "total": total.as_dict(),
        "by_speaker": _counter_dict(by_speaker),
        "by_confusion_partner": _counter_dict(by_partner),
        "by_pair": _counter_dict(by_pair),
        "by_session": _counter_dict(by_session),
        "by_active_speakers": _counter_dict(by_active),
        "by_target_share": _counter_dict(by_share),
    }


def _wav_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            return frames / float(rate) if rate else 0.0
    except (wave.Error, OSError):
        return 0.0


def _wav_samples(path: Path) -> tuple[List[float], int]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.getnframes()
            data = handle.readframes(frames)
    except (wave.Error, OSError):
        return [], 0
    if width != 2 or channels < 1:
        return [], rate
    import array

    values = array.array("h")
    values.frombytes(data)
    if channels > 1:
        values = array.array("h", values[::channels])
    return [float(value) / 32768.0 for value in values], rate


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def _segment_rms(samples: Sequence[float], rate: int, start: float, end: float) -> float:
    if not samples or rate <= 0:
        return 0.0
    left = max(0, int(math.floor(start * rate)))
    right = min(len(samples), int(math.ceil(end * rate)))
    if right <= left:
        return 0.0
    return _rms(samples[left:right])


def _load_reference_words(path: Path) -> List[dict]:
    words: List[dict] = []
    for record in _read_jsonl(path):
        for word in record.get("words") or []:
            if not isinstance(word, dict):
                continue
            if word.get("start") is None or word.get("end") is None:
                continue
            words.append(
                {
                    "speaker": str(word.get("speaker") or record.get("speaker") or "unknown"),
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                    "word": str(word.get("word") or word.get("text") or ""),
                }
            )
    return sorted(words, key=lambda item: (item["start"], item["end"], item["word"]))


def _summarize_prepared_eval(root: Path) -> Dict[str, Any]:
    windows: List[EvalWindowSummary] = []
    by_speaker = Counter()
    by_session = Counter()
    by_active = Counter()
    by_share = Counter()
    total_duration = 0.0
    window_dirs = sorted(path for path in root.glob("*/*") if (path / "mixed.wav").exists())
    for window_dir in window_dirs:
        reference_path = window_dir / "reference" / "clips" / "clips.jsonl"
        if not reference_path.exists():
            continue
        words = _load_reference_words(reference_path)
        clip_samples: Dict[str, tuple[List[float], int]] = {}
        for wav_path in sorted((window_dir / "clips").glob("*.wav")):
            clip_samples[wav_path.name] = _wav_samples(wav_path)
        file_by_speaker: Dict[str, str] = {}
        for record in _read_jsonl(reference_path):
            speaker = str(record.get("speaker") or "")
            file_name = str(record.get("file") or "")
            if speaker and file_name:
                file_by_speaker.setdefault(speaker, file_name)
        word_speakers = Counter()
        word_active = Counter()
        word_share = Counter()
        for word in words:
            speaker = str(word["speaker"])
            start = float(word["start"])
            end = float(word["end"])
            powers: Dict[str, float] = {}
            for candidate_speaker, file_name in file_by_speaker.items():
                samples, rate = clip_samples.get(file_name, ([], 0))
                value = _segment_rms(samples, rate, start, end)
                powers[candidate_speaker] = value * value
            total_power = sum(powers.values())
            target_power = powers.get(speaker, 0.0)
            target_share = target_power / total_power if total_power > 0.0 else 0.0
            max_power = max(powers.values()) if powers else 0.0
            active_count = sum(
                1 for value in powers.values() if max_power > 0.0 and value >= max_power * 0.05
            )
            share_bucket = _share_bucket(target_share)
            word_speakers[speaker] += 1
            word_active[str(active_count)] += 1
            word_share[share_bucket] += 1
            by_speaker[speaker] += 1
            by_active[str(active_count)] += 1
            by_share[share_bucket] += 1
        duration = _wav_duration_seconds(window_dir / "mixed.wav")
        total_duration += duration
        session = window_dir.parent.name
        by_session[session] += len(words)
        windows.append(
            EvalWindowSummary(
                window=str(window_dir.relative_to(root)),
                session=session,
                duration_seconds=round(duration, 3),
                speaker_count=len(file_by_speaker),
                word_count=len(words),
                words_by_speaker=_counter_dict(word_speakers),
                words_by_active_speakers=_counter_dict(word_active),
                words_by_target_share=_counter_dict(word_share),
            )
        )
    return {
        "root": str(root),
        "window_count": len(windows),
        "duration_seconds": round(total_duration, 3),
        "duration_hours": round(total_duration / 3600.0, 4),
        "word_count": int(sum(window.word_count for window in windows)),
        "words_by_speaker": _counter_dict(by_speaker),
        "words_by_session": _counter_dict(by_session),
        "words_by_active_speakers": _counter_dict(by_active),
        "words_by_target_share": _counter_dict(by_share),
        "windows": [asdict(window) for window in windows],
    }


def _manifest_requirements(inventory: CorpusInventory) -> List[str]:
    requirements = [
        "Create train/dev/test splits by held-out session, not by individual word/window.",
        "Balance hard-overlap examples by speaker and target-share bucket before domain-scale TSE.",
        "Include positive and negative enrollment sources outside each evaluation window.",
        "Record license and checkpoint provenance before external TSE model adaptation.",
    ]
    prepared = inventory.prepared_eval.get("combined") or {}
    if int(prepared.get("window_count") or 0) < 10:
        requirements.append(
            "Current prepared eval is too small for final claims; add more held-out sessions."
        )
    if int(prepared.get("words_by_session", {}).get("Session22", 0)) and int(
        prepared.get("words_by_session", {}).get("Session61", 0)
    ):
        requirements.append(
            "Current eval is dominated by Session22/Session61; add unrelated held-out calls."
        )
    return requirements


def _split_risks(inventory: CorpusInventory) -> List[str]:
    risks: List[str] = []
    prepared = inventory.prepared_eval.get("combined") or {}
    sessions = prepared.get("words_by_session") or {}
    if len(sessions) <= 3:
        risks.append(
            "Prepared eval has too few independent sessions for upper-90 generalization claims."
        )
    clean = inventory.clean_bank.get("quality_records") or {}
    clean_sessions = clean.get("by_session") or {}
    if sessions and clean_sessions:
        overlap = sorted(set(sessions) & set(clean_sessions))
        if overlap:
            risks.append(
                "Speaker-bank/session overlap exists with prepared eval; future manifests must audit "
                f"source-window leakage. Overlap examples: {', '.join(overlap[:5])}."
            )
    return risks


def build_inventory(args: argparse.Namespace) -> CorpusInventory:
    prod_root = args.prod_root.expanduser()
    prepared_roots = [path.expanduser() for path in args.prepared_root]
    inventory = CorpusInventory(
        prod_root=str(prod_root),
        prepared_roots=[str(path) for path in prepared_roots],
    )

    bank_root = _latest_bank_root(prod_root)
    if bank_root is not None:
        quality_path = bank_root / "dataset" / "quality_records.jsonl"
        summary_path = bank_root / "dataset" / "dataset_summary.json"
        inventory.clean_bank["root"] = str(bank_root)
        if summary_path.exists():
            inventory.clean_bank["dataset_summary"] = _read_json(summary_path)
        if quality_path.exists():
            inventory.clean_bank["quality_records"] = _summarize_quality_records(quality_path)

    mixed_root = _latest_dataset_root(prod_root, "mixed_base")
    if mixed_root is not None:
        candidate_pool = mixed_root / "candidate_pool.jsonl"
        summary_path = mixed_root / "dataset_summary.json"
        inventory.mixed_candidate_pool["root"] = str(mixed_root)
        if summary_path.exists():
            inventory.mixed_candidate_pool["dataset_summary"] = _read_json(summary_path)
        if candidate_pool.exists():
            inventory.mixed_candidate_pool["candidate_pool"] = _summarize_candidate_pool(
                candidate_pool
            )

    hard_candidates: List[Dict[str, Any]] = []
    for hard_root in _dataset_roots(prod_root, "hard_negative"):
        records = hard_root / "hard_negative_records.jsonl"
        if records.exists():
            summary = _summarize_hard_negatives(records)
            hard_candidates.append({"root": str(hard_root), "records": summary})
    if hard_candidates:
        primary = max(
            hard_candidates,
            key=lambda item: int(item["records"]["total"]["count"]),
        )
        hard_root = Path(primary["root"])
        summary_path = hard_root / "dataset_summary.json"
        inventory.hard_negatives["root"] = str(hard_root)
        inventory.hard_negatives["selection_strategy"] = "largest hard_negative_records.jsonl"
        inventory.hard_negatives["available_roots"] = [
            {
                "root": str(item["root"]),
                "records": int(item["records"]["total"]["count"]),
            }
            for item in hard_candidates
        ]
        if summary_path.exists():
            inventory.hard_negatives["dataset_summary"] = _read_json(summary_path)
        inventory.hard_negatives["records"] = primary["records"]

    prepared_summaries = []
    unique_windows: Dict[str, Mapping[str, Any]] = {}
    for root in prepared_roots:
        if not root.exists():
            continue
        summary = _summarize_prepared_eval(root)
        prepared_summaries.append(summary)
        for window in summary.get("windows") or []:
            key = str(window.get("window") or "")
            if key and key not in unique_windows:
                unique_windows[key] = window
    combined_words_by_speaker = Counter()
    combined_words_by_session = Counter()
    combined_words_by_active = Counter()
    combined_words_by_share = Counter()
    total_duration = 0.0
    total_words = 0
    for window in unique_windows.values():
        total_duration += float(window.get("duration_seconds") or 0.0)
        total_words += int(window.get("word_count") or 0)
        combined_words_by_speaker.update(window.get("words_by_speaker") or {})
        combined_words_by_active.update(window.get("words_by_active_speakers") or {})
        combined_words_by_share.update(window.get("words_by_target_share") or {})
        session = str(window.get("session") or "unknown")
        combined_words_by_session[session] += int(window.get("word_count") or 0)
    inventory.prepared_eval = {
        "roots": prepared_summaries,
        "dedupe_key": "relative session/window name",
        "combined": {
            "window_count": len(unique_windows),
            "duration_seconds": round(total_duration, 3),
            "duration_hours": round(total_duration / 3600.0, 4),
            "word_count": total_words,
            "words_by_speaker": _counter_dict(combined_words_by_speaker),
            "words_by_session": _counter_dict(combined_words_by_session),
            "words_by_active_speakers": _counter_dict(combined_words_by_active),
            "words_by_target_share": _counter_dict(combined_words_by_share),
        },
    }
    inventory.split_risks = _split_risks(inventory)
    inventory.next_manifest_requirements = _manifest_requirements(inventory)
    return inventory


def _fmt_hours(seconds: float) -> str:
    return f"{seconds / 3600.0:.2f}h"


def _markdown_table_count(title: str, values: Mapping[str, Any], *, limit: int = 12) -> List[str]:
    lines = [f"## {title}", "", "| key | value |", "| --- | ---: |"]
    for key, value in list(values.items())[:limit]:
        if isinstance(value, Mapping) and "hours" in value:
            rendered = f"{value.get('count', 0)} / {value.get('hours', 0):.2f}h"
        else:
            rendered = str(value)
        lines.append(f"| {key} | {rendered} |")
    lines.append("")
    return lines


def write_markdown(inventory: CorpusInventory, path: Path) -> None:
    clean = inventory.clean_bank.get("quality_records") or {}
    mixed = inventory.mixed_candidate_pool.get("candidate_pool") or {}
    hard = inventory.hard_negatives.get("records") or {}
    prepared = inventory.prepared_eval.get("combined") or {}
    lines = [
        "# Speaker Big-Bet Corpus Inventory",
        "",
        "This inventory is a planning artifact for domain-scale target-speaker extraction and joint",
        "speaker-attributed ASR work. It intentionally avoids recommending more shallow selectors or",
        "small architecture tweaks.",
        "",
        "## Summary",
        "",
        f"- Production root: `{inventory.prod_root}`",
        f"- Clean accepted bank: {_fmt_hours(float((clean.get('accepted') or {}).get('seconds') or 0.0))}",
        f"- Mixed candidate pool total: {_fmt_hours(float((mixed.get('total') or {}).get('seconds') or 0.0))}",
        f"- Mixed candidate pool accepted: {_fmt_hours(float((mixed.get('accepted') or {}).get('seconds') or 0.0))}",
        f"- Hard-negative records: {(hard.get('total') or {}).get('count', 0)}",
        f"- Prepared eval windows: {prepared.get('window_count', 0)}",
        f"- Prepared eval duration: {_fmt_hours(float(prepared.get('duration_seconds') or 0.0))}",
        f"- Prepared eval words: {prepared.get('word_count', 0)}",
        "",
    ]
    if clean.get("by_speaker"):
        lines.extend(_markdown_table_count("Clean Bank By Speaker", clean["by_speaker"]))
    if mixed.get("by_speaker"):
        lines.extend(_markdown_table_count("Mixed Candidate Pool By Speaker", mixed["by_speaker"]))
    if mixed.get("by_active_speakers"):
        lines.extend(
            _markdown_table_count(
                "Mixed Candidate Pool By Active Speakers", mixed["by_active_speakers"]
            )
        )
    if mixed.get("by_target_share"):
        lines.extend(
            _markdown_table_count(
                "Mixed Candidate Pool By Dominant Share", mixed["by_target_share"]
            )
        )
    if hard.get("by_pair"):
        lines.extend(_markdown_table_count("Hard Negatives By Pair", hard["by_pair"], limit=20))
    if prepared.get("words_by_speaker"):
        lines.extend(
            _markdown_table_count("Prepared Eval Words By Speaker", prepared["words_by_speaker"])
        )
    if prepared.get("words_by_target_share"):
        lines.extend(
            _markdown_table_count(
                "Prepared Eval Words By Target Share", prepared["words_by_target_share"]
            )
        )
    lines.extend(
        [
            "## Split Risks",
            "",
            *(f"- {item}" for item in inventory.split_risks),
            "",
            "## Next Manifest Requirements",
            "",
            *(f"- {item}" for item in inventory.next_manifest_requirements),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory corpus coverage for speaker big-bet work."
    )
    parser.add_argument(
        "--prod-root",
        type=Path,
        default=Path(".outputs/speaker_id_baseline_prod_graph"),
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        action="append",
        default=[],
        help="Prepared eval root containing session/window/mixed.wav directories.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/tmp/codex_speaker_big_bet_corpus_inventory.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/specs/speaker_big_bet_corpus_inventory.md"),
    )
    args = parser.parse_args()
    if not args.prepared_root:
        args.prepared_root = [args.prod_root.expanduser() / "prepared_eval"]
    inventory = build_inventory(args)
    payload = asdict(inventory)
    args.output_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output_json.expanduser().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(inventory, args.output_md.expanduser())
    print(
        json.dumps(
            {
                "output_json": str(args.output_json.expanduser()),
                "output_md": str(args.output_md.expanduser()),
                "clean_hours": (
                    inventory.clean_bank.get("quality_records", {}).get("accepted", {}) or {}
                ).get("hours", 0),
                "mixed_candidate_hours": (
                    inventory.mixed_candidate_pool.get("candidate_pool", {}).get("total", {}) or {}
                ).get("hours", 0),
                "prepared_eval_windows": inventory.prepared_eval.get("combined", {}).get(
                    "window_count", 0
                ),
                "prepared_eval_words": inventory.prepared_eval.get("combined", {}).get(
                    "word_count", 0
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
