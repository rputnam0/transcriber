#!/usr/bin/env python3
"""Reproducible mono-only MOSS comparison using the frozen early-session benchmark.

Prepare/score in the app environment; infer in the separate official MOSS environment.
References are isolated-track ASR and VAD, not human gold. No reference enters inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
REVISION = "e8681d68e7042738ffca8ac8212bc8fcb1131ab8"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def prepare(args):
    import numpy as np
    import soundfile as sf

    from train_early_speaker_identity import session_tracks

    manifest = json.loads((args.corpus / "manifest.json").read_text())
    records = []
    for session, meta in manifest["sessions"].items():
        if meta["split"] not in {"dev", "test"}:
            continue
        waves, _ = session_tracks(meta)
        for index, window in enumerate(meta["evaluation_windows"]):
            for offset in (0, 30):
                start = window["start"] + offset
                wave = sum(
                    np.array(w[int(start * 16000) : int((start + 30) * 16000)])
                    for w in waves.values()
                )
                cut = f"{session}_{index:02d}_{offset:02d}"
                audio = args.output / "audio" / f"{cut}.wav"
                audio.parent.mkdir(parents=True, exist_ok=True)
                sf.write(audio, wave, 16000, subtype="FLOAT")
                records.append(
                    dict(
                        cut_id=cut,
                        session=session,
                        split=meta["split"],
                        window=index,
                        offset=offset,
                        start=start,
                        duration=30,
                        audio=str(audio.resolve()),
                        sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
                    )
                )
    save(args.output / "manifest.json", records)
    print(f"Prepared {len(records)} fixed 30-second mono windows", flush=True)


def infer(args):
    if args.device != "mps":
        return _infer(args)
    import fcntl

    lock_path = Path.home() / ".cache/transcriber/moss-inference.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _infer(args)


def _infer(args):
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        generate_transcription,
    )
    from run_moss_transcribe_diarize import normalized_segments

    torch.set_num_threads(4)
    records = json.loads((args.output / "manifest.json").read_text())
    records = [r for r in records if r["split"] == args.split]
    if args.limit:
        records = records[: args.limit]
    dtype = getattr(torch, args.dtype)
    model_kwargs = {} if Path(args.model).is_dir() else {"revision": REVISION}
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            **model_kwargs,
            trust_remote_code=True,
            dtype=dtype,
            attn_implementation=getattr(args, "attention", "eager"),
        )
        .to(args.device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    identity_names = getattr(model.config, "stable_speaker_names", None)
    prompt_kwargs = {}
    if identity_names:
        from moss_identity_prompt import identity_prompt

        prompt_kwargs["prompt"] = identity_prompt(identity_names)
    batch_results = {}
    for index, record in enumerate(records, 1):
        target = args.output / "predictions" / f"{record['cut_id']}.json"
        cached_raw = None
        if target.exists():
            cached = json.loads(target.read_text())
            if (
                cached["sha256"] != record["sha256"]
                or cached["revision"] != REVISION
                or cached["model"] != args.model
            ):
                raise ValueError("Stale inference cache; use a fresh output directory")
            from moss_batch_inference import decode_issues

            if getattr(args, "repair_repetition", False) and decode_issues(cached["raw_text"]):
                cached_raw = cached["raw_text"]
            else:
                continue
        wave, _ = sf.read(record["audio"], dtype="float32")
        assert hashlib.sha256(np.asarray(wave).tobytes()).hexdigest() == record["sha256"]
        started = time.monotonic()
        if cached_raw is not None:
            raw, elapsed = cached_raw, cached["elapsed_seconds"]
        elif getattr(args, "batch_size", 1) > 1:
            from moss_batch_inference import generate_batch

            if record["cut_id"] not in batch_results:
                batch = [
                    r
                    for r in records[index - 1 : index - 1 + args.batch_size]
                    if not (args.output / "predictions" / f"{r['cut_id']}.json").exists()
                ]
                texts = generate_batch(
                    model,
                    processor,
                    batch,
                    prompt_kwargs=prompt_kwargs,
                    device=torch.device(args.device),
                )
                elapsed = (time.monotonic() - started) / len(batch)
                batch_results.update(
                    {r["cut_id"]: (text, elapsed) for r, text in zip(batch, texts, strict=True)}
                )
            raw, elapsed = batch_results.pop(record["cut_id"])
        else:
            generated = generate_transcription(
                model,
                processor,
                build_transcription_messages(Path(record["audio"]), **prompt_kwargs),
                max_new_tokens=2048,
                do_sample=False,
                device=torch.device(args.device),
                dtype=dtype,
            )
            raw = str(generated["text"])
            elapsed = time.monotonic() - started
        repair = None
        if getattr(args, "repair_repetition", False):
            from moss_batch_inference import decode_issues, generate_batch

            issues = decode_issues(raw)
            if issues:
                repair = dict(
                    issues=issues,
                    original_raw_text=raw,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=8,
                )
                print(f"RETRY DECODE {record['cut_id']}: {issues}", flush=True)
                raw = generate_batch(
                    model,
                    processor,
                    [record],
                    prompt_kwargs=prompt_kwargs,
                    device=torch.device(args.device),
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=8,
                )[0]
                if decode_issues(raw):
                    save(
                        args.output / "decode_failures" / f"{record['cut_id']}.json",
                        dict(record=record, repair=repair, retry_raw_text=raw),
                    )
                    raise RuntimeError(f"Unresolved incomplete decode for {record['cut_id']}")
        parser_input = raw
        if getattr(args, "repair_repetition", False):
            from moss_batch_inference import repair_speaker_brackets

            parser_input = repair_speaker_brackets(raw)
        parsed = parse_transcript(parser_input)
        recovered = []
        if getattr(args, "repair_repetition", False):
            from moss_batch_inference import recover_compact_segments

            parsed, recovered = recover_compact_segments(parser_input, parsed)
        segments = normalized_segments(parsed, duration=record["duration"])
        result = dict(
            record,
            model=args.model,
            revision=REVISION,
            dtype=args.dtype,
            device=args.device,
            raw_text=raw,
            identity_names=identity_names,
            segments=segments,
            elapsed_seconds=elapsed,
            batch_size=getattr(args, "batch_size", 1),
            attention=getattr(args, "attention", "eager"),
        )
        if repair:
            result["decode_repair"] = repair
        if parser_input != raw or recovered:
            result["parser_repair"] = {
                "rule": "recover bounded turns and speaker closing braces",
                "parser_input": parser_input,
                "recovered_turns": recovered,
            }
        save(target, result)
        print(
            f"{index}/{len(records)} {record['cut_id']}: {len(segments)} segments "
            f"in {result['elapsed_seconds']:.1f}s",
            flush=True,
        )


def reference_words(transcripts, masks, start, offset, duration):
    """Keep source-VAD supported words; derive brief turns from word gaps, not sentence spans."""
    from evaluate_named_words import normalized

    all_words = []
    for name, segments in transcripts.items():
        if name == "mix":
            continue
        words = [w for s in segments for w in s.get("words", []) if normalized(w["word"])]
        words.sort(key=lambda w: w["start"])
        groups = []
        for word in words:
            if not groups or word["start"] - groups[-1][-1]["end"] > 0.3:
                groups.append([])
            groups[-1].append(word)
        for group in groups:
            brief = group[-1]["end"] - group[0]["start"] <= 2
            for word in group:
                center = (word["start"] + word["end"]) / 2
                frame = min(int((start + center) / 0.032), len(masks[name]) - 1)
                if not masks[name][frame] or not offset <= center < offset + duration:
                    continue
                lo, hi = max(0, word["start"] - offset), min(duration, word["end"] - offset)
                if hi > lo:
                    all_words.append(
                        dict(start=lo, end=hi, speaker=name, text=word["word"], brief=brief)
                    )
    for word in all_words:
        word["overlap"] = any(
            other["speaker"] != word["speaker"]
            and min(word["end"], other["end"]) > max(word["start"], other["start"])
            for other in all_words
        )
    return all_words


def combine(scores):
    totals = {
        key: sum(s[key] for s in scores)
        for key in ("reference_words", "predicted_words", "matched_words")
    }
    totals["recall"] = totals["matched_words"] / max(totals["reference_words"], 1)
    totals["precision"] = totals["matched_words"] / max(totals["predicted_words"], 1)
    totals["f1"] = (
        2 * totals["matched_words"] / max(totals["reference_words"] + totals["predicted_words"], 1)
    )
    totals["categories"] = {}
    if "named_edit_errors" in scores[0]:
        totals["named_edit_errors"] = sum(s["named_edit_errors"] for s in scores)
        totals["named_word_error_rate"] = totals["named_edit_errors"] / max(
            totals["reference_words"], 1
        )
    for category in scores[0]["categories"]:
        ref = sum(s["categories"][category]["reference_words"] for s in scores)
        matched = sum(s["categories"][category]["matched_words"] for s in scores)
        totals["categories"][category] = dict(
            reference_words=ref, matched_words=matched, recall=matched / max(ref, 1)
        )
    if "per_speaker" in scores[0]:
        totals["per_speaker"] = {
            name: combine([s["per_speaker"][name] for s in scores])
            for name in scores[0]["per_speaker"]
        }
        supported = [s for s in totals["per_speaker"].values() if s["reference_words"]]
        totals["macro_f1"] = sum(s["f1"] for s in supported) / max(len(supported), 1)
    return totals


def named_edit_errors(reference, prediction, mapping):
    """Speaker-concatenated edit distance within each fixed cut, independent of timestamps."""
    from evaluate_se_dicow_oracle_cut import normalized_words
    from score_moss_speaker_attributed import prediction_texts

    refs = prediction_texts(reference)
    named = [dict(s, speaker=mapping.get(s["speaker"]) or "Unknown") for s in prediction]
    preds = prediction_texts(named)
    total = 0
    for name in refs.keys() | preds.keys():
        left, right = normalized_words(refs.get(name, "")), normalized_words(preds.get(name, ""))
        row = list(range(len(right) + 1))
        for i, token in enumerate(left, 1):
            previous, row = row, [i]
            for j, other in enumerate(right, 1):
                row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (token != other)))
        total += row[-1]
    return total


def score(args):
    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import get_token

    from attribute_cached_diarization import cluster_features, bind_clusters
    from evaluate_named_words import predict_labels
    from prepare_early_domain_corpus import NAMES
    from refine_named_turns import corrected_names, turn_features
    from score_moss_manifest import temporal_attributed_score, reference_texts
    from score_moss_speaker_attributed import _optimal_unique_mapping, prediction_texts
    from train_early_speaker_identity import session_tracks
    from transcriber.diarization import (
        DEFAULT_DIARIZATION_MODEL,
        DiarizationTurn,
        _resolve_embedder,
    )

    torch.set_num_threads(4)
    embedder = _resolve_embedder(
        model_name=DEFAULT_DIARIZATION_MODEL, hf_token=get_token(), device="mps"
    )
    identity = dict(np.load(args.identity))
    policy = json.loads((args.baseline / "diarization_dev/turn_refinement.json").read_text())[
        "policy"
    ]
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    records = [
        r
        for r in json.loads((args.output / "manifest.json").read_text())
        if r["split"] == args.split
    ]
    if args.limit:
        records = records[: args.limit]
    scores = {k: [] for k in ("baseline", "moss_named", "moss_local_names", "moss_oracle_names")}
    details = []
    for record in records:
        prediction = json.loads(
            (args.output / "predictions" / f"{record['cut_id']}.json").read_text()
        )
        folder = (
            args.baseline / f"words_{args.split}" / f"{record['session']}_{record['window']:02d}"
        )
        meta = manifest["sessions"][record["session"]]
        # Missing roster members are genuinely absent, not missing reference files.
        # A present owner's missing transcript must still fail instead of hiding data loss.
        transcripts = {
            n: (
                json.loads(
                    (
                        (args.reference_root / f"words_{args.split}" / folder.name / f"{n}.json")
                        if args.reference_root and n != "mix"
                        else (folder / f"{n}.json")
                    ).read_text()
                )
                if n == "mix" or n in meta["tracks"]
                else []
            )
            for n in NAMES + ["mix"]
        }
        _, masks = session_tracks(meta)
        offset = record["offset"]
        refs = reference_words(transcripts, masks, record["start"] - offset, offset, 30)
        diar = json.loads((folder / "diarization.json").read_text())
        base = []
        for word, name in predict_labels(
            transcripts["mix"],
            [DiarizationTurn(**t) for t in diar["regular"]],
            [DiarizationTurn(**t) for t in diar["exclusive"]],
            False,
        ):
            center = (word["start"] + word["end"]) / 2
            if offset <= center < offset + 30:
                base.append(
                    dict(
                        start=max(0, word["start"] - offset),
                        end=min(30, word["end"] - offset),
                        text=word["word"],
                        speaker=name or "Unknown",
                    )
                )
        wave, _ = sf.read(record["audio"], dtype="float32")
        turns = prediction["segments"]
        features = cluster_features(wave, turns, embedder)
        bindings = bind_clusters(features, identity)
        mapping = {label: b.get("proposed_speaker") or "Unknown" for label, b in bindings.items()}
        local_features = turn_features(wave, turns, embedder)
        local_names = corrected_names(turns, bindings, local_features, identity, policy)
        local_turns = [dict(t, speaker=n or "Unknown") for t, n in zip(turns, local_names)]
        oracle = _optimal_unique_mapping(prediction_texts(turns), reference_texts(refs))
        sources = dict(
            baseline=(base, {name: name for name in NAMES}),
            moss_named=(turns, mapping),
            moss_local_names=(local_turns, {name: name for name in NAMES}),
            moss_oracle_names=(turns, oracle),
        )
        if prediction.get("identity_names"):
            direct_mapping = {
                f"S{i+1:02d}": name for i, name in enumerate(prediction["identity_names"])
            }
            sources["moss_direct_names"] = (
                turns,
                direct_mapping,
            )
            sources["moss_hybrid_names"] = (
                turns,
                {
                    label: name if name != "Unknown" else direct_mapping.get(label, "Unknown")
                    for label, name in mapping.items()
                },
            )
        if "enrollment_mapping" in prediction:
            sources["moss_enrollment_names"] = (
                turns,
                {
                    label: prediction["enrollment_mapping"].get(label, name)
                    for label, name in mapping.items()
                },
            )
        cut_scores = {}
        for method, (segments, names) in sources.items():
            result = temporal_attributed_score(
                refs, segments, names, brief_turn_seconds=2, tolerance_seconds=0.5
            )
            result["named_edit_errors"] = named_edit_errors(refs, segments, names)
            result["per_speaker"] = {}
            for name in NAMES:
                ref_person = [s for s in refs if s["speaker"] == name]
                pred_person = [s for s in segments if names.get(s["speaker"]) == name]
                person = temporal_attributed_score(
                    ref_person, pred_person, names, brief_turn_seconds=2, tolerance_seconds=0.5
                )
                result["per_speaker"][name] = person
            scores.setdefault(method, []).append(result)
            cut_scores[method] = result
        details.append(
            dict(
                cut_id=record["cut_id"],
                bindings=bindings,
                oracle_mapping=oracle,
                scores=cut_scores,
                reference=refs,
                named_segments=[
                    dict(t, speaker=mapping.get(t["speaker"], "Unknown")) for t in turns
                ],
                local_named_segments=local_turns,
            )
        )
        print("SCORED", record["cut_id"], flush=True)
    report = dict(
        split=args.split,
        windows=len(records),
        seconds=30 * len(records),
        reference_type="isolated-track automatic ASR word timestamps + source VAD; not human gold",
        reference_root=str(args.reference_root or args.baseline),
        identity_path=str(args.identity.resolve()),
        identity_sha256=hashlib.sha256(args.identity.read_bytes()).hexdigest(),
        caveat="MOSS provides segment timing; baseline provides finer word timing. Span matching can favor MOSS.",
        metrics={k: combine(v) for k, v in scores.items()},
        records=details,
    )
    save(args.output / f"{args.split}_scores.json", report)
    print(json.dumps(report["metrics"], indent=2))


def select(args):
    report = json.loads((args.output / "dev_scores.json").read_text())
    if report["split"] != "dev" or report["windows"] != 24:
        raise ValueError("Selection requires all 24 frozen development clips")
    selected = min(
        ["moss_named", "moss_local_names"],
        key=lambda name: report["metrics"][name]["named_word_error_rate"],
    )
    save(
        args.output / "selection.json",
        dict(
            method=selected,
            criterion="minimum development named_word_error_rate; tie prefers cluster naming",
            split="dev",
        ),
    )
    print(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "infer", "score", "select"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--model", default=MODEL, help="Public model ID or local checkpoint")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    args = parser.parse_args()
    {"prepare": prepare, "infer": infer, "score": score, "select": select}[args.command](args)


if __name__ == "__main__":
    main()
