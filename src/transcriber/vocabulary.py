"""Read canonical campaign names without promoting transcription-error aliases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_vocabulary(path: str | Path | None) -> dict:
    """Accept a word list, an explicit term bank, or a D&D campaign.json.

    Campaign aliases are lookup keys and can include known misspellings. Only
    aliases that are exact word spans of the canonical name become ASR hints.
    Explicit ``spoken_forms`` in a custom bank can add confirmed nicknames.
    """
    if path is None:
        return {"terms": [], "sha256": None, "source_sha256": None}
    source = Path(path).expanduser()
    raw = source.read_bytes()
    payload = json.loads(raw)
    rows = []
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for key in ("pcs", "characters", "items", "terms", "table_people", "locations"):
            rows.extend(payload.get(key) or [])
        groups = [payload, *(payload.get("parties") or [])]
        for group in groups:
            if group is not payload:
                rows.append({"name": group.get("name")})
                rows.extend(group.get("pcs") or [])
            for person in group.get("participants") or []:
                if person.get("role") == "dm":
                    continue
                rows.append(person.get("character") or {"name": person.get("display_name")})
    else:
        raise ValueError("Vocabulary must be a JSON list or campaign/term-bank object")

    terms: dict[str, str] = {}

    def add(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Vocabulary names must be nonempty strings")
        text = " ".join(value.split())
        if len(text) > 200:
            raise ValueError("Vocabulary names must be at most 200 characters")
        terms.setdefault(text.casefold(), text)

    for row in rows:
        if isinstance(row, str):
            add(row)
            continue
        if not isinstance(row, dict):
            raise ValueError("Vocabulary entries must be strings or name objects")
        name = row.get("canonical") or row.get("name")
        add(name)
        padded = f" {name} "
        for alias in row.get("aliases") or []:
            if isinstance(alias, str) and f" {alias} " in padded:
                add(alias)
        for form in row.get("spoken_forms") or []:
            add(form)
    if not terms:
        raise ValueError("Vocabulary contains no canonical names")
    ordered = list(terms.values())
    return {
        "terms": ordered,
        "sha256": hashlib.sha256(json.dumps(ordered, ensure_ascii=False).encode()).hexdigest(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    }
