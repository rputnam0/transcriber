#!/usr/bin/env python3
"""Select on development, freeze the decision, then evaluate once on held-out sessions.

This driver only considers deployable naming methods. It never selects using test scores.
Full-recording deployment is intentionally a separate command using the frozen selection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from test_moss_mac import MODEL, REVISION, save

METHODS = ("moss_named", "moss_local_names", "moss_direct_names", "moss_hybrid_names")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def wait_for(path, timeout=14400):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(path)
        time.sleep(10)


def select_candidate(reports):
    """Only common development clips may enter the selector; ignore oracle diagnostics."""
    candidates, reference = [], None
    for path, report in reports:
        if report["split"] != "dev":
            raise ValueError("Checkpoint selection may only use development scores")
        current = {r["cut_id"]: r["reference"] for r in report["records"]}
        if reference is not None and current != reference:
            raise ValueError("Selector references or clip sets differ")
        reference = current
        for method in METHODS:
            if method in report["metrics"]:
                metric = report["metrics"][method]
                candidates.append(
                    dict(
                        selector=str(path),
                        method=method,
                        named_word_error_rate=metric["named_word_error_rate"],
                        macro_f1=metric["macro_f1"],
                    )
                )
    if not candidates:
        raise ValueError("No deployable candidates")
    # Keep deterministic caller/method order on exact ties.
    return min(candidates, key=lambda c: c["named_word_error_rate"]), candidates


def promotion(candidate, public):
    return (
        candidate["named_word_error_rate"] < public["named_word_error_rate"]
        and candidate["macro_f1"] >= public["macro_f1"] - 0.01
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--moss-python", type=Path, required=True)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--selector", type=Path, action="append", required=True)
    p.add_argument("--also-finalist", type=Path, action="append", default=[])
    args = p.parse_args()
    run = args.run_root
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTORCH_ENABLE_MPS_FALLBACK="1")

    def command(parts, log):
        print("RUN", " ".join(map(str, parts)), flush=True)
        with log.open("w") as output:
            subprocess.run(
                list(map(str, parts)), env=env, stdout=output, stderr=subprocess.STDOUT, check=True
            )

    def evaluate(output, model, split, identity=None):
        identity = identity or args.identity
        output.mkdir(parents=True, exist_ok=True)
        manifest = run / "evaluation/manifest.json"
        if (output / "manifest.json").exists():
            if digest(output / "manifest.json") != digest(manifest):
                raise ValueError("Full evaluation manifest changed")
        else:
            shutil.copy2(manifest, output / "manifest.json")
        command(
            [
                args.moss_python,
                "scripts/test_moss_mac.py",
                "infer",
                "--output",
                output,
                "--model",
                model,
                "--split",
                split,
                "--batch-size",
                "4",
            ],
            output / f"{split}_infer.log",
        )
        command(
            [
                sys.executable,
                "scripts/test_moss_mac.py",
                "score",
                "--output",
                output,
                "--split",
                split,
                "--corpus",
                args.corpus,
                "--baseline",
                args.corpus,
                "--reference-root",
                args.corpus / "references_v3",
                "--identity",
                identity,
            ],
            output / f"{split}_score.log",
        )
        return json.loads((output / f"{split}_scores.json").read_text())

    reports = []
    for path in args.selector:
        wait_for(path)
        reports.append((path, json.loads(path.read_text())))
    winner, candidates = select_candidate(reports)
    finalists = [winner]
    for path in args.also_finalist:
        matching = [(p, r) for p, r in reports if p == path]
        if not matching:
            raise ValueError("Additional finalist must already belong to the selector")
        extra, _ = select_candidate(matching)
        if extra not in finalists:
            finalists.append(extra)
    public_path = run / "full_public/dev_scores.json"
    wait_for(public_path)
    public = json.loads(public_path.read_text())
    pm = public["metrics"]["moss_named"]
    validated = []
    for finalist in finalists:
        source = Path(finalist["selector"]).parent
        selector_report = json.loads(Path(finalist["selector"]).read_text())
        identity = Path(selector_report.get("identity_path", str(args.identity)))
        expected_identity = selector_report.get("identity_sha256")
        if expected_identity and digest(identity) != expected_identity:
            raise ValueError("Enrollment changed after selector scoring")
        prediction = next((source / "predictions").glob("*.json"))
        model = json.loads(prediction.read_text())["model"]
        output = run / source.name.replace("selector_", "full_", 1)
        if not (output / "dev_scores.json").exists():
            (output / "predictions").mkdir(parents=True, exist_ok=True)
            for prediction in (source / "predictions").glob("*.json"):
                shutil.copy2(prediction, output / "predictions" / prediction.name)
            candidate = evaluate(output, model, "dev", identity)
        else:
            candidate = json.loads((output / "dev_scores.json").read_text())
        if {r["cut_id"]: r["reference"] for r in public["records"]} != {
            r["cut_id"]: r["reference"] for r in candidate["records"]
        }:
            raise ValueError("Full-development references differ")
        cm = candidate["metrics"][finalist["method"]]
        validated.append(
            dict(
                selector=finalist,
                identity=str(identity),
                model=model,
                output=str(output),
                metrics=cm,
                passes=promotion(cm, pm),
            )
        )
    accepted = [v for v in validated if v["passes"]]
    chosen = min(accepted or validated, key=lambda v: v["metrics"]["named_word_error_rate"])
    winner, model, output, cm = (
        chosen["selector"],
        chosen["model"],
        Path(chosen["output"]),
        chosen["metrics"],
    )
    promoted = chosen["passes"]
    identity = Path(chosen["identity"])
    selection = dict(
        rule="Selector finalists compared on full development; minimum named word error among candidates improving public error and passing macro F1 guard (at most 0.01 decline)",
        validated_finalists=validated,
        candidate=winner,
        candidates=candidates,
        candidate_model=model,
        candidate_full_output=str(output),
        candidate_full_development=cm,
        public_full_development=pm,
        promoted=promoted,
        deployment_model=model if promoted else MODEL,
        deployment_identity=str(identity if promoted else args.identity),
        candidate_identity=str(identity),
        deployment_method=winner["method"] if promoted else "moss_named",
        evaluation_manifest_sha256=digest(run / "evaluation/manifest.json"),
        identity_sha256=digest(identity),
        checkpoint_sha256=(
            digest(Path(model) / "model.safetensors") if Path(model).is_dir() else None
        ),
        public_model_revision=REVISION,
        test_sessions=[37, 56],
        test_used_for_selection=False,
        frozen_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    frozen_path = run / "selection.json"
    if frozen_path.exists():
        previous = json.loads(frozen_path.read_text())
        comparable = {k: v for k, v in selection.items() if k != "frozen_at_utc"}
        if comparable != {k: v for k, v in previous.items() if k != "frozen_at_utc"}:
            raise ValueError("Selection is already frozen and differs")
    else:
        save(frozen_path, selection)
    print("FROZEN", selection["deployment_model"], selection["deployment_method"], flush=True)
    # Also evaluate the preselected candidate when rejected: a declared comparison,
    # never an opportunity to reselect or tune from test results.
    evaluate(run / "full_public", MODEL, "test")
    evaluate(output, model, "test", identity)
    for split in ("dev", "test"):
        command(
            [
                sys.executable,
                "scripts/summarize_moss_finetune_results.py",
                "--public",
                run / f"full_public/{split}_scores.json",
                "--candidate",
                output / f"{split}_scores.json",
                "--method",
                winner["method"],
                "--output",
                run / f"final_{split}_comparison",
            ],
            run / f"final_{split}_comparison.log",
        )
    print("COMPLETE: frozen selection and held-out comparison saved", flush=True)


if __name__ == "__main__":
    main()
