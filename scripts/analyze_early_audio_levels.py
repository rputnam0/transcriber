#!/usr/bin/env python3
"""Measure level imbalance and overlap on the fixed 20:00–30:00 evaluation regions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_early_speaker_identity import session_tracks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    result = {"region_seconds": [1200, 1800], "audible_floor_dbfs": -55, "sessions": {}}
    for session in ["62", "63"]:
        waves, masks = session_tracks(manifest["sessions"][session])
        rms, activity = {}, {}
        for name, wave in waves.items():
            frames = np.asarray(wave[1200 * 16000 : 1800 * 16000]).reshape(-1, 320)
            rms[name] = np.sqrt(np.mean(frames**2, axis=1))
            times = 1200 + (np.arange(len(frames)) + 0.5) / 50
            activity[name] = masks[name][(times / 0.032).astype(int)]
        count = sum(activity.values())
        per = {}
        for name, values in rms.items():
            db = 20 * np.log10(np.maximum(values, 1e-9))
            active = activity[name]
            audible = active & (db > -55)
            overlap = audible & (count > 1)
            others = np.sqrt(sum(v**2 for n, v in rms.items() if n != name))
            sir = 20 * np.log10(np.maximum(values, 1e-9) / np.maximum(others, 1e-9))
            per[name] = dict(
                reference_seconds=float(active.sum() / 50),
                source_vad_overlap_fraction=float(
                    (active & (count > 1)).sum() / max(active.sum(), 1)
                ),
                active_below_floor_fraction=float((active & ~audible).sum() / max(active.sum(), 1)),
                median_active_dbfs=float(np.median(db[active])),
                median_above_floor_dbfs=float(np.median(db[audible])) if audible.any() else None,
                median_overlap_sir_db=float(np.median(sir[overlap])) if overlap.any() else None,
            )
        result["sessions"][session] = per
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
