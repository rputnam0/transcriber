from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from render_moss_named_manifest import cut_offset_seconds


TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: object) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def lexical_coverage(candidate: str, reference: str) -> tuple[int, float]:
    candidate_tokens = _tokens(candidate)
    matches = sum((Counter(candidate_tokens) & Counter(_tokens(reference))).values())
    return matches, matches / max(1, len(candidate_tokens))


def reconcile_bindings(
    window_records: Sequence[Mapping[str, object]],
    binding_records: Sequence[Mapping[str, object]],
    primary_segments: Sequence[Mapping[str, object]],
    primary_mapping: Mapping[str, str],
    *,
    minimum_matches: int,
    minimum_coverage: float,
    minimum_margin: float,
) -> list[dict]:
    records_by_cut = {str(record.get("cut_id") or ""): record for record in window_records}
    reconciled = []
    for source_binding in binding_records:
        binding = dict(source_binding)
        cut_id = str(binding.get("cut_id") or "")
        record = records_by_cut.get(cut_id)
        if record is None:
            raise ValueError(f"No MOSS window record for {cut_id}")
        offset = cut_offset_seconds(cut_id)
        window_end = offset + float(record.get("duration") or 30.0)
        primary_text: dict[str, list[str]] = defaultdict(list)
        for segment in primary_segments:
            midpoint = (float(segment.get("start") or 0.0) + float(segment.get("end") or 0.0)) / 2
            if offset <= midpoint < window_end:
                speaker = str(primary_mapping.get(str(segment.get("speaker") or "")) or "")
                if speaker:
                    primary_text[speaker].append(str(segment.get("text") or ""))
        local_text: dict[str, list[str]] = defaultdict(list)
        for segment in list(record.get("segments") or []):
            local_text[str(segment.get("speaker") or "")].append(str(segment.get("text") or ""))

        identity_evidence = {}
        independent = dict(binding.get("independent_mapping") or {})
        one_to_one = dict(binding.get("one_to_one_mapping") or {})
        for stream, parts in local_text.items():
            text = " ".join(parts)
            scores = []
            for speaker, primary_parts in primary_text.items():
                matches, coverage = lexical_coverage(text, " ".join(primary_parts))
                scores.append((coverage, matches, speaker))
            scores.sort(reverse=True)
            best = scores[0] if scores else (0.0, 0, None)
            runner_up = scores[1] if len(scores) > 1 else (0.0, 0, None)
            margin = best[0] - runner_up[0]
            anchored = (
                best[2] is not None
                and best[1] >= minimum_matches
                and best[0] >= minimum_coverage
                and margin >= minimum_margin
            )
            if anchored:
                independent[stream] = best[2]
                one_to_one[stream] = best[2]
            identity_evidence[stream] = {
                "source": (
                    "long-transcript-lexical-anchor" if anchored else "historical-enrollment"
                ),
                "speaker": best[2] if anchored else one_to_one.get(stream),
                "lexical_matches": best[1],
                "lexical_coverage": best[0],
                "lexical_margin": margin,
                "lexical_runner_up": runner_up[2],
            }
        binding["independent_mapping"] = independent
        binding["one_to_one_mapping"] = one_to_one
        binding["identity_evidence"] = identity_evidence
        reconciled.append(binding)
    return reconciled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Anchor window-local MOSS identities to a confidently named long decode."
    )
    parser.add_argument("--window-output", type=Path, required=True)
    parser.add_argument("--window-binding", type=Path, required=True)
    parser.add_argument("--primary-output", type=Path, required=True)
    parser.add_argument("--primary-binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-matches", type=int, default=3)
    parser.add_argument("--minimum-coverage", type=float, default=0.6)
    parser.add_argument("--minimum-margin", type=float, default=0.25)
    args = parser.parse_args()

    windows = json.loads(args.window_output.read_text(encoding="utf-8"))
    binding = json.loads(args.window_binding.read_text(encoding="utf-8"))
    primary = json.loads(args.primary_output.read_text(encoding="utf-8"))
    primary_binding = json.loads(args.primary_binding.read_text(encoding="utf-8"))
    provenance = dict(binding.get("enrollment_provenance") or {})
    if bool(provenance.get("uses_evaluation_session_audio", True)):
        raise ValueError("Window binding used evaluation-session enrollment")
    if bool(primary_binding.get("inference_uses_isolated_target_audio")):
        raise ValueError("Primary binding used isolated evaluation audio")
    reconciled = reconcile_bindings(
        list(windows.get("records") or []),
        list(binding.get("record_bindings") or []),
        list(primary.get("segments") or []),
        dict(primary_binding.get("one_to_one_mapping") or {}),
        minimum_matches=args.minimum_matches,
        minimum_coverage=args.minimum_coverage,
        minimum_margin=args.minimum_margin,
    )
    payload = {
        **binding,
        "primary_moss_output": str(args.primary_output),
        "primary_binding": str(args.primary_binding),
        "reconciliation": {
            "minimum_matches": args.minimum_matches,
            "minimum_coverage": args.minimum_coverage,
            "minimum_margin": args.minimum_margin,
            "anchored_streams": sum(
                evidence.get("source") == "long-transcript-lexical-anchor"
                for record in reconciled
                for evidence in dict(record.get("identity_evidence") or {}).values()
            ),
        },
        "record_bindings": reconciled,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["reconciliation"], indent=2))


if __name__ == "__main__":
    main()
