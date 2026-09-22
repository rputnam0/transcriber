#!/usr/bin/env python3
"""Compare waveform levels/channels and enrollment similarity across recording domains."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, action="append", required=True)
    p.add_argument("--development-report", type=Path, required=True)
    p.add_argument("--deployment-attribution", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    waveforms = []
    for path in args.audio:
        rows = []
        with sf.SoundFile(path) as f:
            duration = len(f) / f.samplerate
            for start in np.linspace(0, max(0, duration - 30), 20):
                f.seek(round(start * f.samplerate))
                wave = f.read(30 * f.samplerate, dtype="float32", always_2d=True)
                rows.append(
                    dict(
                        start=float(start),
                        rms_db=float(20 * np.log10(max(np.sqrt(np.mean(wave**2)), 1e-9))),
                        peak=float(np.abs(wave).max()),
                        clipped_fraction=float(np.mean(np.abs(wave) >= 0.999)),
                        stereo_correlation=(
                            float(np.corrcoef(wave.T)[0, 1]) if wave.shape[1] == 2 else None
                        ),
                    )
                )
            waveforms.append(
                dict(
                    file=path.name,
                    sample_rate=f.samplerate,
                    channels=f.channels,
                    duration=duration,
                    windows=rows,
                )
            )
    (args.output / "early_audio_quality.json").write_text(json.dumps(waveforms, indent=2) + "\n")
    dev = json.loads(args.development_report.read_text())
    domains = {
        "later_development": [b for row in dev["records"] for b in row["bindings"].values()],
        "early_recording": [
            b
            for path in args.deployment_attribution.glob("*.json")
            for b in json.loads(path.read_text())["bindings"].values()
        ],
    }
    rows = []
    for domain, bindings in domains.items():
        names = sorted({b["proposed_speaker"] for b in bindings if b.get("proposed_speaker")})
        for name in ["all", *names]:
            selected = [
                b
                for b in bindings
                if b.get("clean_seconds", 0) >= 1
                and b.get("cosine") is not None
                and (name == "all" or b.get("proposed_speaker") == name)
            ]
            if selected:
                rows.append(
                    dict(
                        domain=domain,
                        proposed_speaker=name,
                        clusters=len(selected),
                        median_centroid_cosine=float(np.median([b["cosine"] for b in selected])),
                        median_posterior=float(np.median([b["mean_posterior"] for b in selected])),
                    )
                )
    with (args.output / "embedding_domain_diagnostic.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved waveform and embedding diagnostics; similarity does not verify speaker names.")


if __name__ == "__main__":
    main()
