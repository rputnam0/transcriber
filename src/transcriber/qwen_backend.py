"""Isolated Qwen ASR worker with vocabulary and aligned chunk ownership.

Run through the ordinary CLI with --backend qwen. Apple Silicon uses MLX; --device cuda
uses the official Qwen checkpoints through PyTorch. Model dependencies are imported
only in the separate worker so the existing application environment stays intact.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .audio import load_audio_mono

MODEL = "mlx-community/Qwen3-ASR-1.7B-bf16"
MODEL_REVISION = "e1f6c266914abc5a46e8756e02580f834a6cf8a7"
ALIGNER = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
ALIGNER_REVISION = "0e1a68e91d815300c7c9754b2a7639378b23db15"
CUDA_MODEL = "Qwen/Qwen3-ASR-1.7B"
CUDA_MODEL_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
CUDA_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"
CUDA_ALIGNER_REVISION = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
RUNTIMES = {
    "mlx": dict(
        model=MODEL,
        model_revision=MODEL_REVISION,
        aligner=ALIGNER,
        aligner_revision=ALIGNER_REVISION,
        python=".venv-qwen/bin/python",
        batch_size=1,
    ),
    "cuda": dict(
        model=CUDA_MODEL,
        model_revision=CUDA_MODEL_REVISION,
        aligner=CUDA_ALIGNER,
        aligner_revision=CUDA_ALIGNER_REVISION,
        python=".venv-qwen-cuda/bin/python",
        # Batched first-pass decoding; 8 chunks peak near 7 GB of GPU memory.
        batch_size=8,
    ),
}


def resolve_model(model: str, runtime: str) -> tuple[str, str | None]:
    """Map the generic CLI model names to the runtime's pinned checkpoint."""
    pinned = RUNTIMES[runtime]
    if model in {"qwen", "large-v3", "medium.en"}:
        model = pinned["model"]
    return model, pinned["model_revision"] if model == pinned["model"] else None


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def transcribe_files(
    files: list[str],
    *,
    vocabulary: dict,
    python: str | None,
    model: str,
    cache_root: str | None,
    work_dir: Path,
    local_files_only: bool,
    runtime: str = "mlx",
) -> list[list[dict]]:
    executable = (
        Path(python).expanduser()
        if python
        else (Path(__file__).resolve().parents[2] / RUNTIMES[runtime]["python"])
    )
    if not executable.is_file():
        raise ValueError("Qwen runtime missing; see docs/mac-isolated-track-transcription.md")
    request = dict(
        files=[str(Path(f).resolve()) for f in files],
        vocabulary=vocabulary,
        runtime=runtime,
        model=resolve_model(model, runtime)[0],
        cache_root=cache_root,
        work_dir=str(work_dir.resolve()),
        local_files_only=local_files_only,
    )
    with tempfile.TemporaryDirectory(prefix="transcriber_qwen_") as tmp:
        request_path, response_path = Path(tmp) / "request.json", Path(tmp) / "response.json"
        _save(request_path, request)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        command = [
            str(executable),
            "-m",
            "transcriber.qwen_backend",
            str(request_path),
            str(response_path),
        ]
        subprocess.run(command, env=env, check=True)
        return json.loads(response_path.read_text())["files"]


def aligned_words(text: str, aligned: list[dict], duration: float) -> list[dict]:
    """Retain recognizer spelling/punctuation; reject lexical changes by alignment."""

    def norm(s):
        return "".join(c.casefold() for c in s if c.isalnum())

    tokens, prefix = [], []
    for token in text.split():
        if norm(token):
            tokens.append(" ".join(prefix + [token]))
            prefix = []
        elif tokens:
            tokens[-1] += " " + token
        else:
            prefix.append(token)
    if len(tokens) != len(aligned) or any(
        norm(t) != norm(a["text"]) for t, a in zip(tokens, aligned)
    ):
        raise ValueError("Qwen aligner changed lexical tokens; raw decode retained for review")
    words = []
    previous = -1.0
    for token, item in zip(tokens, aligned):
        start, end = float(item["start"]), float(item["end"])
        if not (
            math.isfinite(start)
            and math.isfinite(end)
            # The pinned aligner quantizes timestamps in 80 ms steps.
            and 0 <= start <= end <= duration + 0.08
            and start >= previous
        ):
            raise ValueError("Invalid Qwen word timestamps; raw decode retained for review")
        words.append(dict(word=token, start=min(start, duration), end=min(end, duration)))
        previous = start
    return words


def transcribe_wave(
    wave, model, aligner, *, vocabulary, cache, provenance, regions=None, batch_size=1
):
    """Decode speech regions without removing their offsets on the source timeline.

    With ``batch_size`` > 1 the first hinted decode of up to that many uncached chunks runs as one
    ``model.generate_batch`` call; retries, alignment and the chunk records stay per chunk.
    """
    rate, core_samples, padding = 16000, 30 * 16000, 3 * 16000
    segments = []
    regions = [[0, len(wave)]] if regions is None else regions
    windows = []
    for region_start, region_end in regions:
        for start in range(region_start, region_end, core_samples):
            end = min(start + core_samples, region_end)
            windows.append(
                (
                    start,
                    end,
                    max(region_start, start - padding),
                    min(region_end, end + padding),
                    end == region_end,
                )
            )
    first_pass = {}

    def decode_ahead(first):
        pending = []
        for j in range(first, len(windows)):
            if len(pending) == batch_size:
                break
            if not (cache / f"{j:06d}.json").exists() and wave[windows[j][2] : windows[j][3]].any():
                pending.append(j)
        results = model.generate_batch(
            [wave[windows[j][2] : windows[j][3]] for j in pending],
            language="English",
            hotwords=vocabulary["terms"] or None,
            max_tokens=1024,
            temperature=0.0,
        )
        first_pass.update(zip(pending, results))

    for index, (start, end, lo, hi, final_window) in enumerate(windows):
        target = cache / f"{index:06d}.json"
        if target.exists():
            record = json.loads(target.read_text())
            if record["provenance"] != provenance:
                raise ValueError("Qwen cache provenance mismatch")
        else:
            chunk_started = time.perf_counter()
            hint_echo = None
            retry_reason = None
            clip = wave[lo:hi]
            if not clip.any():
                text, aligned, generated = "", [], 0
            else:
                if batch_size > 1 and index not in first_pass:
                    decode_ahead(index)
                result = first_pass.pop(index, None) or model.generate(
                    clip,
                    language="English",
                    hotwords=vocabulary["terms"] or None,
                    max_tokens=1024,
                    temperature=0.0,
                    verbose=False,
                )
                text, generated = result.text.strip(), result.generation_tokens
                raw_path = cache / f"{index:06d}.raw.json"

                # Quiet input can cause literal copying of the complete vocabulary.
                # Preserve that evidence and retry the audio without the hints.
                def normalize(value):
                    return "".join(c.casefold() for c in value if c.isalnum())

                terms = vocabulary["terms"]
                echo = len(terms) >= 4 and normalize(" ".join(terms)) in normalize(text)
                if echo or (generated >= 1024 and terms):
                    retry_reason = "hint_echo" if echo else "token_budget"
                    original = dict(text=text, generation_tokens=generated, reason=retry_reason)
                    hint_echo = original if echo else None
                    _save(cache / f"{index:06d}.{retry_reason}.json", original)
                    result = model.generate(
                        clip,
                        language="English",
                        hotwords=None,
                        max_tokens=1024,
                        temperature=0.0,
                        verbose=False,
                    )
                    text, generated = result.text.strip(), result.generation_tokens
                _save(
                    raw_path,
                    dict(
                        text=text,
                        generation_tokens=generated,
                        provenance=provenance,
                        start=lo / rate,
                        end=hi / rate,
                    ),
                )
                aligned = (
                    aligner.generate(audio=clip, text=text, language="English").segments
                    if text and generated < 1024
                    else []
                )
                _save(
                    raw_path,
                    dict(
                        text=text,
                        generation_tokens=generated,
                        aligned=aligned,
                        provenance=provenance,
                        start=lo / rate,
                        end=hi / rate,
                    ),
                )
            review_reason = None
            if generated >= 1024:
                words = []
                review_reason = "Recognition reached its output limit; listen and transcribe."
            else:
                try:
                    words = aligned_words(text, aligned, len(clip) / rate)
                except ValueError as error:
                    words = []
                    review_reason = str(error)
            owned = []
            for w in words:
                absolute = dict(w, start=w["start"] + lo / rate, end=w["end"] + lo / rate)
                midpoint = (absolute["start"] + absolute["end"]) / 2
                if start / rate <= midpoint < end / rate or (
                    final_window and midpoint == end / rate
                ):
                    owned.append(absolute)
            record = dict(
                provenance=provenance,
                text=text,
                aligned=aligned,
                words=owned,
                core_start=start / rate,
                core_end=end / rate,
                zero_duration_words=sum(w["start"] == w["end"] for w in owned),
                clipped_timestamp_count=sum(a["end"] > len(clip) / rate for a in aligned),
                hint_echo_retry=hint_echo is not None,
                retry_reason=retry_reason,
                review_reason=review_reason,
                processing_seconds=time.perf_counter() - chunk_started,
            )
            _save(target, record)
        words = record["words"]
        if record.get("review_reason"):
            segments.append(
                dict(
                    start=start / rate,
                    end=end / rate,
                    text="[Transcription needs review]",
                    words=[],
                    speaker=None,
                    review_required=True,
                    review_reason=record["review_reason"],
                )
            )
        # Split at pauses for useful subtitles and readable speaker turns.
        groups = []
        for word in words:
            if not groups or word["start"] - groups[-1][-1]["end"] > 0.8:
                groups.append([])
            groups[-1].append(word)
        for group in groups:
            segments.append(
                dict(
                    start=group[0]["start"],
                    end=group[-1]["end"],
                    text=" ".join(w["word"] for w in group),
                    words=group,
                    speaker=None,
                )
            )
        print(f"Qwen speech chunk {index + 1}/{len(windows)}", flush=True)
    return segments


def _load_runtime(runtime: str, model_path: str, aligner_path: str):
    """Load the recognizer and aligner; return them with the runtime's package versions."""
    if runtime == "cuda":
        from .qwen_cuda import load

        return load(model_path, aligner_path)
    from mlx_audio.stt import load

    versions = dict(mlx_audio=version("mlx-audio"), mlx=version("mlx"))
    return load(model_path), load(aligner_path), versions


def run_worker(request: dict) -> dict:
    started = time.perf_counter()
    from huggingface_hub import snapshot_download
    from .speech_activity import SETTINGS, VAD_SHA256, probabilities, resolve_vad, speech_regions

    runtime = request.get("runtime", "mlx")
    pinned = RUNTIMES[runtime]

    def resolve(name, revision=None):
        if Path(name).is_dir():
            return str(Path(name).resolve())
        root = request["cache_root"]
        hub = str(Path(root) / "hub") if root and Path(root).name != "hub" else root
        return snapshot_download(
            name, revision=revision, cache_dir=hub, local_files_only=request["local_files_only"]
        )

    model_path = resolve(*resolve_model(request["model"], runtime))
    aligner_path = resolve(pinned["aligner"], pinned["aligner_revision"])
    vad_path = resolve_vad(request["cache_root"], request["local_files_only"])
    model, aligner, runtime_versions = _load_runtime(runtime, model_path, aligner_path)
    setup_seconds = time.perf_counter() - started
    vocabulary = request["vocabulary"]
    files = []
    timings = []
    activity = []
    for path in request["files"]:
        track_started = time.perf_counter()
        wave = load_audio_mono(path)
        regions = speech_regions(probabilities(wave, vad_path), len(wave), **SETTINGS)
        speech = dict(
            regions=[dict(start=a / 16000, end=b / 16000) for a, b in regions],
            retained_seconds=sum(b - a for a, b in regions) / 16000,
            audio_seconds=len(wave) / 16000,
        )
        # MLX keeps its original key set so existing chunk caches still resume.
        provenance = dict(
            schema=6,
            **({} if runtime == "mlx" else dict(runtime=runtime, batch_size=pinned["batch_size"])),
            model_path=model_path,
            aligner_path=aligner_path,
            **runtime_versions,
            transformers=version("transformers"),
            tokenizers=version("tokenizers"),
            numpy=version("numpy"),
            vocabulary=vocabulary,
            audio_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
            sample_rate=16000,
            core_seconds=30,
            context_seconds=3,
            language="English",
            temperature=0.0,
            max_tokens=1024,
            speech_detection=dict(
                model_sha256=VAD_SHA256, settings=SETTINGS, onnxruntime=version("onnxruntime")
            ),
        )
        ident = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        cache = Path(request["work_dir"]) / ident
        print(
            f"Qwen track {len(files) + 1}/{len(request['files'])}; "
            f"{len(vocabulary['terms'])} vocabulary terms",
            flush=True,
        )
        segments = transcribe_wave(
            wave,
            model,
            aligner,
            vocabulary=vocabulary,
            cache=cache,
            provenance=provenance,
            regions=regions,
            batch_size=pinned["batch_size"],
        )
        _save(
            cache / "result.json", dict(provenance=provenance, segments=segments, activity=speech)
        )
        files.append(segments)
        activity.append(speech)
        timings.append(
            dict(
                audio_seconds=len(wave) / 16000,
                processing_seconds=time.perf_counter() - track_started,
            )
        )
        if runtime == "mlx":
            import mlx.core as mx

            mx.clear_cache()
    return {
        "files": files,
        "activity": activity,
        "timing": dict(
            setup_seconds=setup_seconds, tracks=timings, total_seconds=time.perf_counter() - started
        ),
    }


if __name__ == "__main__":
    _save(Path(sys.argv[2]), run_worker(json.loads(Path(sys.argv[1]).read_text())))
