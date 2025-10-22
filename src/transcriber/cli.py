from __future__ import annotations

import argparse
import json
import sys
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import time
from tqdm import tqdm

from .audio import gather_inputs, cleanup_tmp, is_audio_file
from .consolidate import consolidate, save_outputs, choose_speaker

def _setup_logging_and_warnings(log_level: str, quiet: bool) -> None:
    """Configure logging and warnings to reduce noise by default.

    - When quiet: show only errors, suppress Python warnings, and mute common
      third‑party loggers; keep progress bars.
    - When not quiet: honor the requested log level.
    """
    import warnings

    # Default to WARNING in quiet mode so our own progress logs are visible
    requested = getattr(logging, (log_level or "ERROR").upper(), logging.ERROR)
    level = logging.WARNING if quiet and requested > logging.WARNING else requested
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")

    # Always clamp a few very chatty libraries
    noisy_loggers = [
        "whisperx",
        "pyannote",
        "pyannote.audio",
        "speechbrain",
        "pytorch_lightning",
        "lightning",
        "torch",
        "faster_whisper",
        "transformers",
        "huggingface_hub",
        "urllib3",
        "onnxruntime",
        "numba",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.ERROR)

    if quiet:
        # Suppress all logs at INFO and below (keep WARNING+ visible)
        logging.disable(logging.INFO)
        # Hide Python warnings (deprecations, reproducibility notes, etc.)
        warnings.filterwarnings("ignore")
        # Silence ONNX Runtime C++ warnings
        os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")  # 3=ERROR
        try:
            import onnxruntime as _ort  # type: ignore

            try:
                _ort.set_default_logger_severity(3)
            except Exception:
                pass
        except Exception:
            pass
    else:
        logging.disable(logging.NOTSET)
        # Show warnings according to defaults
        warnings.simplefilter("default")


def _tqdm_enabled() -> bool:
    """True if interactive progress bars should be shown.

    Disabled when not attached to a TTY, when running under systemd (INVOCATION_ID
    present), or when TRANSCRIBER_NO_TQDM=1.
    """
    if os.getenv("TRANSCRIBER_NO_TQDM") == "1":
        return False
    if os.getenv("INVOCATION_ID"):
        return False
    return sys.stderr.isatty()


def _find_config_path(explicit: str | None) -> str | None:
    """Return a config file path if one can be found.

    Priority:
      1) Explicit path via --config
      2) TRANSCRIBER_CONFIG environment variable
      3) ./transcriber.yaml or ./transcriber.json
      4) ./config/transcriber.yaml or ./config/transcriber.json
      5) ~/.config/transcriber/config.yaml or config.json
    """
    if explicit:
        p = Path(explicit)
        return str(p) if p.exists() else None
    env = os.getenv("TRANSCRIBER_CONFIG")
    if env and Path(env).exists():
        return env
    cwd = Path.cwd()
    for candidate in (
        cwd / "transcriber.yaml",
        cwd / "transcriber.json",
        cwd / "config" / "transcriber.yaml",
        cwd / "config" / "transcriber.json",
        Path.home() / ".config" / "transcriber" / "config.yaml",
        Path.home() / ".config" / "transcriber" / "config.json",
    ):
        if candidate.exists():
            return str(candidate)
    return None


def _apply_config_defaults(ap: argparse.ArgumentParser, cfg: Dict) -> None:
    """Translate config keys to argparse defaults.

    CLI remains authoritative; this only sets defaults.
    """
    if not cfg:
        return
    defaults: Dict[str, object] = {}
    key_map = {
        "backend": "backend",
        "model": "model",
        "compute_type": "compute_type",
        "batch_size": "batch_size",
        "output_dir": "output_dir",
        "speaker_mapping": "speaker_mapping",
        "min_speakers": "min_speakers",
        "max_speakers": "max_speakers",
        "cache_root": "cache_root",
        "cache_mode": "cache_mode",
        "local_files_only": "local_files_only",
        "single_file_speaker": "single_file_speaker",
        "vad_on_cpu": "vad_on_cpu",
        "pyannote_on_cpu": "pyannote_on_cpu",
        "log_level": "log_level",
        "quiet": "quiet",
        "auto_batch": "auto_batch",
        "watch": "watch",
        "watch_interval": "watch_interval",
        "watch_stability": "watch_stability",
        "input": "input",
    }
    for ck, ak in key_map.items():
        if ck in cfg and cfg[ck] is not None:
            defaults[ak] = cfg[ck]
    # write_srt/write_jsonl are inverted flags in CLI
    if "write_srt" in cfg:
        defaults["no_srt"] = not bool(cfg["write_srt"])
    if "write_jsonl" in cfg:
        defaults["no_jsonl"] = not bool(cfg["write_jsonl"])
    ap.set_defaults(**defaults)

def _load_yaml_or_json(path: str | None) -> Dict:
    """Load a mapping from YAML or JSON, returning an empty mapping when absent.

    Provides friendlier errors for malformed inputs and treats empty YAML files as empty maps.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    try:
        if p.suffix.lower() == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return data or {}
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to parse mapping file {p}: {exc}") from exc

def _resolve_cache_root(explicit: str | None, cache_mode: str | None = None) -> str | None:
    """Return a base cache directory depending on config/environment.

    Priority:
      1) Explicit path (CLI/config `cache_root`)
      2) cache_mode:
         - 'env': do not set; return None and rely on env/defaults of libraries
         - 'repo': use project-local path (`$CWD/.hf_cache`)
         - 'home' (default): use `~/hf_cache`
      3) Environment overrides when `cache_root` is unset and mode is not 'env':
         - `TRANSCRIBER_CACHE_ROOT`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE`
    """
    if explicit:
        return explicit
    mode = (cache_mode or "home").lower()
    if mode == "env":
        return None
    # Respect env if provided
    for env_var in ("TRANSCRIBER_CACHE_ROOT", "HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        value = os.getenv(env_var)
        if value:
            return value
    if mode == "repo":
        return str((Path.cwd() / ".hf_cache"))
    # Default: home
    return str(Path.home() / "hf_cache")


def run_transcribe(
    input_path: str,
    backend: str = "whisperx",
    model_name: str = "medium.en",
    compute_type: str = "float16",
    batch_size: int = 32,
    output_dir: str = "outputs",
    speaker_mapping_path: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    write_srt: bool = True,
    write_jsonl: bool = True,
    cache_root: str | None = None,
    local_files_only: bool = False,
    single_file_speaker: str | None = None,
    vad_on_cpu: bool = False,
    pyannote_on_cpu: bool = False,
    quiet: bool = True,
    auto_batch: bool = True,
    cache_mode: str | None = None,
    device: str | None = None,
) -> None:
    logger = logging.getLogger("transcriber")
    # Track whether the user explicitly provided a cache root
    user_provided_cache = bool(cache_root)
    cache_root = _resolve_cache_root(cache_root, cache_mode)
    # Avoid requiring hf_transfer across all environments by default. Users can override.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    # Configure HF cache environment to ensure reuse and offline behavior for hub downloads
    if cache_root:
        # Expand and normalize (~, symlinks)
        hub_dir = Path(os.path.expanduser(cache_root)).resolve()
        hub_dir.mkdir(parents=True, exist_ok=True)
        # If a hub/ subfolder is provided, treat its parent as HF_HOME
        if hub_dir.name == "hub" and hub_dir.exists():
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_dir)
            os.environ["HF_HOME"] = str(hub_dir.parent)
        else:
            os.environ["HF_HOME"] = str(hub_dir)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_dir / "hub")
        # Also align other HF caches to reduce surprises
        os.environ.setdefault("HF_DATASETS_CACHE", str(hub_dir / "datasets"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hub_dir / "transformers"))
        # Avoid requiring hf_transfer for downloads in many environments
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        logger.info("Running in local-files-only mode (offline).")
    if cache_root:
        logger.info(
            "Using cache: HF_HOME=%s HUGGINGFACE_HUB_CACHE=%s",
            os.getenv("HF_HOME"),
            os.getenv("HUGGINGFACE_HUB_CACHE"),
        )

    files, tmp_root = gather_inputs(input_path)
    if not files:
        raise SystemExit(f"No audio files found in: {input_path}")

    logger.info("Found %d audio file(s) under %s", len(files), input_path)

    per_file_segments: List[Tuple[str, List[dict]]] = []
    diar_by_file: Dict[str, List[dict]] = {}

    # Compute sub-dirs for non-HF-hub model caches (faster-whisper/ctranslate2 + align models)
    if cache_root:
        root_path = Path(os.path.expanduser(cache_root)).resolve()
        model_cache_dir = str(root_path / "models")
        align_cache_dir = str(root_path / "align")
    else:
        model_cache_dir = None
        align_cache_dir = None

    # Preload mapping early so we can show friendly labels during progress
    try:
        mapping_pre = _load_yaml_or_json(speaker_mapping_path)
    except BaseException as exc:  # noqa: PIE786
        logging.getLogger("transcriber").error(
            "Failed to load speaker mapping %s: %s (continuing without mapping)",
            speaker_mapping_path,
            exc,
        )
        mapping_pre = {}
    file_labels: Dict[str, str] = {
        f: (choose_speaker(f, mapping_pre) if mapping_pre else Path(f).stem) for f in files
    }

    if backend == "whisperx":
        # Import after environment setup so Hugging Face picks up HF_* vars
        from .whisperx_backend import transcribe_with_whisperx, _detect_device as _wx_detect_device
        # Accept common token env vars for gated models
        hf_token = (
            os.getenv("HUGGING_FACE_HUB_TOKEN")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
        )
        if not hf_token:
            logger.warning("HF_TOKEN not set — pyannote diarization models may fail if gated.")
        # Determine effective batch size (auto picks a conservative value by device/model)
        device_guess = device if device in {"cpu", "cuda"} else _wx_detect_device()
        eff_bs = batch_size
        if auto_batch:
            eff_bs = _recommend_batch_size(device_guess, model_name, compute_type, user_hint=batch_size)
        iter_files = files
        use_tqdm = _tqdm_enabled()
        if use_tqdm:
            iter_files = tqdm(
                files,
                desc="Transcribing (whisperx)",
                unit="file",
                bar_format="{l_bar}{bar} | ETA: {remaining} | {n_fmt}/{total_fmt}",
                dynamic_ncols=True,
            )
        for f in iter_files:
            label = file_labels.get(f, Path(f).stem)
            logger.warning("Start: %s", Path(f).name)
            start_ts = time.time()
            if use_tqdm:
                from tqdm import tqdm as _tqdm  # local alias to avoid confusion
                pbar = _tqdm(total=3, desc=f"{label}", leave=False, unit="phase")
            else:
                pbar = None

            def _progress_cb(phase: str, step: int, total: int) -> None:  # noqa: ANN001
                if pbar is not None:
                    current = pbar.n
                    to = max(min(step, total), current)
                    if to > current:
                        pbar.update(to - current)
                    pbar.set_postfix_str(phase)
                else:
                    if step == 1 and phase == "asr":
                        logger.warning("Phase: asr -> %s", Path(f).name)
                    elif step == 2 and phase == "align":
                        logger.warning("Phase: align -> %s", Path(f).name)
                    elif step == 3 and phase == "diar":
                        logger.warning("Phase: diar -> %s", Path(f).name)

            logger.warning(
                "Using device=%s compute=%s for %s",
                device_guess,
                compute_type,
                Path(f).name,
            )
            segs, diar = transcribe_with_whisperx(
                f,
                model_name=model_name,
                compute_type=compute_type,
                batch_size=eff_bs,
                hf_token=hf_token,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                model_cache_dir=model_cache_dir,
                align_cache_dir=align_cache_dir,
                local_files_only=local_files_only,
                vad_on_cpu=(vad_on_cpu or pyannote_on_cpu),
                pyannote_on_cpu=pyannote_on_cpu,
                progress_cb=_progress_cb,
                    quiet=quiet,
                    force_device=device_guess,
                )
            if pbar is not None:
                pbar.close()
            dur = int(time.time() - start_ts)
            logger.warning("Finished: %s in %ds (segments=%d)", Path(f).name, dur, len(segs))
            per_file_segments.append((f, segs))
            if diar:
                diar_by_file[f] = diar
    else:
        # faster-whisper only (no diarization)
        logger.info("Using faster-whisper backend.")
        # Import after environment setup so Hugging Face picks up HF_* vars
        from .backend import load_model as fw_load, transcribe_file as fw_transcribe
        from .whisperx_backend import _detect_device as _wx_detect_device

        fw_device = device if device in {"cpu", "cuda"} else _wx_detect_device()
        fw_compute = compute_type
        cpu_supported = {"int8", "int8_float32", "int8_np"}
        if fw_device == "cpu" and fw_compute not in cpu_supported:
            logger.info(
                "Compute type %s is not supported on CPU; falling back to int8 for faster-whisper.",
                fw_compute,
            )
            fw_compute = "int8"
        logger.warning("ASR Device (faster-whisper): %s compute=%s", fw_device, fw_compute)
        # Determine effective batch size (auto picks conservative by device/model)
        eff_bs = batch_size
        if auto_batch:
            eff_bs = _recommend_batch_size(fw_device, model_name, fw_compute, user_hint=batch_size)
        model = fw_load(
            model_name,
            compute_type=fw_compute,
            device=fw_device,
            download_root=model_cache_dir,
            local_files_only=local_files_only,
        )
        use_tqdm = _tqdm_enabled()
        iter_files = tqdm(
            files,
            desc="Transcribing (faster-whisper)",
            unit="file",
            bar_format="{l_bar}{bar} | ETA: {remaining} | {n_fmt}/{total_fmt}",
            dynamic_ncols=True,
        ) if use_tqdm else files
        for f in iter_files:
            label = file_labels.get(f, Path(f).stem)
            logger.warning("Start: %s", Path(f).name)
            start_ts = time.time()
            if use_tqdm:
                from tqdm import tqdm as _tqdm
                with _tqdm(total=1, desc=f"{label}", leave=False, unit="phase") as pbar:
                    segs = fw_transcribe(f, model, batch_size=eff_bs)
                    pbar.update(1)
                    pbar.set_postfix_str("asr")
            else:
                segs = fw_transcribe(f, model, batch_size=eff_bs)
            dur = int(time.time() - start_ts)
            logger.warning("Finished: %s in %ds (segments=%d)", Path(f).name, dur, len(segs))
            per_file_segments.append((f, segs))

    base = Path(input_path).stem

    # Optional speaker renaming and filename-based labeling (prototype parity)
    # Reload mapping after transcription for application (kept separate from progress labels)
    mapping = mapping_pre
    if mapping:
        # If we have multiple input files (e.g., multi-track ZIP), enforce per-file labels
        # regardless of diarization cluster names to match prototype behavior.
        multi_file = len(files) > 1
        for i, (fname, segs) in enumerate(per_file_segments):
            file_label = choose_speaker(fname, mapping)
            for s in segs:
                if multi_file:
                    # Always use the file-derived label for multi-file inputs
                    s["speaker"] = file_label
                else:
                    # Single-file: only fill missing and allow cluster -> name override
                    if not s.get("speaker"):
                        s["speaker"] = file_label
                    if s.get("speaker") in mapping:
                        s["speaker"] = mapping[s["speaker"]]
            per_file_segments[i] = (fname, segs)
        # Apply mapping to diarization output if present (best-effort, non-critical)
        for fname, diar in list(diar_by_file.items()):
            for d in diar:
                if d.get("speaker") in mapping:
                    d["speaker"] = mapping[d["speaker"]]
            diar_by_file[fname] = diar

    # Prototype single-file behavior: allow forcing a generic label
    if single_file_speaker and len(files) == 1:
        target = files[0]
        for i, (fname, segs) in enumerate(per_file_segments):
            if fname == target:
                for s in segs:
                    if not s.get("speaker"):
                        s["speaker"] = single_file_speaker
                per_file_segments[i] = (fname, segs)

    consolidated_pairs = consolidate(per_file_segments)

    # Default behavior: nest outputs under a folder named after the input base
    # e.g., input "Session 32.zip" -> outputs/Session 32/
    effective_output_dir = Path(output_dir)
    base = Path(input_path).stem
    if effective_output_dir.name != base:
        effective_output_dir = effective_output_dir / base

    final_out_dir = save_outputs(
        base_stem=base,
        output_dir=str(effective_output_dir),
        per_file_segments=per_file_segments,
        consolidated_pairs=consolidated_pairs,
        diar_by_file=diar_by_file or None,
        write_srt_file=write_srt,
        write_jsonl_file=write_jsonl,
    )
    # Clean up extracted ZIP contents (prototype parity)
    cleanup_tmp(tmp_root)
    logging.getLogger("transcriber").warning(
        "Done. Outputs in %s", str(Path(final_out_dir).resolve())
    )

def _recommend_batch_size(device: str, model_name: str, compute_type: str, user_hint: int | None = None) -> int:
    """Heuristic batch-size recommendation per device/model/precision.

    - Uses total VRAM when CUDA is available (via torch) to scale conservatively.
    - If `user_hint` is provided, treat it as an upper bound.
    """
    # Base defaults
    recommended = 32
    mem_gb = None
    if device == "cuda":
        try:
            import torch

            props = torch.cuda.get_device_properties(0)  # type: ignore[attr-defined]
            mem_gb = int(props.total_memory / (1024**3))
        except Exception:
            mem_gb = None
    name = (model_name or "").lower()
    prec = (compute_type or "").lower()

    # Model size buckets
    is_large = any(k in name for k in ("large", "large-v3"))
    is_medium = "medium" in name
    # conservative scaling by VRAM and precision
    if device == "cuda":
        if is_large:
            if mem_gb and mem_gb >= 20:
                recommended = 48
            elif mem_gb and mem_gb >= 16:
                recommended = 32
            else:
                recommended = 24
        elif is_medium:
            if mem_gb and mem_gb >= 16:
                recommended = 64
            elif mem_gb and mem_gb >= 12:
                recommended = 48
            else:
                recommended = 32
        else:
            if mem_gb and mem_gb >= 16:
                recommended = 96
            else:
                recommended = 64
        # Float16/bfloat16 tend to fit larger batches than int8 on CPU
        if "int8" in prec:
            recommended = max(16, recommended // 2)
    else:
        # CPU: smaller batches tend to be faster
        recommended = 16 if is_large else 24

    if user_hint is not None:
        return max(1, min(int(user_hint), int(recommended)))
    return int(recommended)


def _expected_txt_path_for_input(input_path: str, output_dir: str) -> Path:
    base = Path(input_path).stem
    return Path(output_dir) / base / f"{base}.txt"


def _iter_candidate_media(root_dir: Path) -> list[str]:
    exts = {".zip"} | {e for e in getattr(__import__(__name__), 'AUDIO_EXTS', set())}  # fallback
    ignore_dirs = {"quarantine", ".cache", "outputs"}
    files: list[str] = []
    for f in root_dir.rglob("*"):
        if not f.is_file():
            continue
        # Skip ignored directories
        parts = set(p.name for p in f.parents)
        if parts & ignore_dirs:
            continue
        if f.suffix.lower() == ".zip" or is_audio_file(f):
            files.append(str(f))
    files.sort()
    return files


def _file_is_stable(path: Path, stability_seconds: int) -> bool:
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return False
    return (time.time() - mtime) >= max(0, stability_seconds)


def watch_and_transcribe(args: argparse.Namespace, cfg: Dict) -> None:
    """Continuously scan an input directory for new audio/ZIP files and transcribe them.

    A file is processed when its corresponding output TXT is missing and the file appears
    stable (no recent modification for `watch_stability` seconds).
    """
    logger = logging.getLogger("transcriber")
    interval = int(getattr(args, "watch_interval", 10))
    stability = int(getattr(args, "watch_stability", 5))

    try:
        while True:
            input_root = args.input or (cfg.get("input") if isinstance(cfg, dict) else None)
            if not input_root:
                logger.error("Watch: missing INPUT and no 'input' in config; sleeping %ss", interval)
                time.sleep(interval)
                continue
            root = Path(input_root)
            if not root.exists():
                logger.error("Watch: path does not exist: %s; sleeping %ss", root, interval)
                time.sleep(interval)
                continue
            if not root.is_dir():
                if root.is_file():
                    parent = root.parent
                    logger.warning("Watch: INPUT is a file; watching parent directory: %s", parent)
                    root = parent
                else:
                    logger.error("Watch: %s is neither file nor directory; sleeping %ss", root, interval)
                    time.sleep(interval)
                    continue

            # Use WARNING so it appears even in quiet mode
            logger.warning("Watch: monitoring %s (every %ss, stability %ss)", root, interval, stability)

            candidates = _iter_candidate_media(root)
            pending: list[str] = []
            for f in candidates:
                expected_txt = _expected_txt_path_for_input(f, args.output_dir)
                if expected_txt.exists():
                    continue
                if not _file_is_stable(Path(f), stability):
                    continue
                pending.append(f)

            if pending:
                logger.warning(
                    "Watch: found %d new file(s); processing (e.g., %s)",
                    len(pending),
                    Path(pending[0]).name,
                )
                outer = tqdm(
                    pending,
                    desc="Watch: processing new files",
                    unit="file",
                    bar_format="{l_bar}{bar} | ETA: {remaining} | {n_fmt}/{total_fmt}",
                    dynamic_ncols=True,
                    leave=False,
                )
                quarantine_dir = root / "quarantine"
                quarantine_dir.mkdir(parents=True, exist_ok=True)
                for f in outer:
                    try:
                        run_transcribe(
                            input_path=f,
                            backend=args.backend,
                            model_name=args.model,
                            compute_type=args.compute_type,
                            batch_size=args.batch_size,
                            output_dir=args.output_dir,
                            speaker_mapping_path=args.speaker_mapping,
                            min_speakers=args.min_speakers,
                            max_speakers=args.max_speakers,
                            write_srt=not args.no_srt,
                            write_jsonl=not args.no_jsonl,
                            cache_root=args.cache_root,
                            local_files_only=args.local_files_only,
                            single_file_speaker=args.single_file_speaker,
                            vad_on_cpu=args.vad_on_cpu,
                            pyannote_on_cpu=args.pyannote_on_cpu,
                            quiet=args.quiet,
                            auto_batch=args.auto_batch,
                        )
                    except BaseException as exc:  # noqa: PIE786
                        msg = str(exc)
                        logger.error("Watch: failed to process %s: %s", f, msg)
                        # Quarantine known-bad inputs to avoid hammering the same file forever
                        if any(token in msg for token in ("Bad CRC-32", "No audio files found")):
                            try:
                                dest = quarantine_dir / Path(f).name
                                Path(f).replace(dest)
                                logger.warning("Watch: moved %s to %s due to input error", f, dest)
                            except Exception as move_exc:
                                logger.error("Watch: failed to quarantine %s: %s", f, move_exc)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n👋 Watch mode stopped by user.")


def main():
    # Phase 1: parse just config + logging to set up environment early
    ap0 = argparse.ArgumentParser(add_help=False)
    ap0.add_argument("--config", help="Path to YAML/JSON config file")
    ap0.add_argument(
        "--log-level",
        default="ERROR",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: ERROR)",
    )
    ap0.add_argument("--quiet", dest="quiet", action="store_true", default=True)
    ap0.add_argument("--no-quiet", dest="quiet", action="store_false")
    prelim, _ = ap0.parse_known_args()

    # Early logging noise clamp
    _setup_logging_and_warnings(prelim.log_level, prelim.quiet)

    # Find and load config (if any)
    cfg_path = _find_config_path(prelim.config)
    cfg = _load_yaml_or_json(cfg_path) if cfg_path else {}

    # Phase 2: full parser with defaults from config
    ap = argparse.ArgumentParser(description="Transcription with alignment + diarization (WhisperX)")
    ap.add_argument("input", nargs="?", help="Path to audio file, directory, or .zip of audios")
    ap.add_argument("--backend", choices=["whisperx", "faster"], default="whisperx")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default=cfg.get("device", "auto"), help="Force device selection (default: auto)")
    ap.add_argument("--model", default="large-v3", help="Model name (e.g., large-v3)")
    ap.add_argument("--compute-type", default="float16", help="Compute type (e.g., float16, int8_float16)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--auto-batch", dest="auto_batch", action="store_true", default=True, help="Choose batch size automatically for your GPU (default)")
    ap.add_argument("--no-auto-batch", dest="auto_batch", action="store_false", help="Disable auto batch-size selection")
    ap.add_argument("--watch", action="store_true", help="Continuously watch INPUT directory for new audio/ZIP files")
    ap.add_argument("--watch-interval", type=int, default=10, help="Polling interval (seconds) in watch mode")
    ap.add_argument("--watch-stability", type=int, default=5, help="Minimum age (seconds) before a new file is considered stable for processing")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--speaker-mapping", dest="speaker_mapping", help="YAML/JSON mapping of speaker IDs -> names")
    ap.add_argument("--min-speakers", type=int)
    ap.add_argument("--max-speakers", type=int)
    ap.add_argument("--no-srt", action="store_true", help="Don't write combined SRT")
    ap.add_argument("--no-jsonl", action="store_true", help="Don't write JSONL")
    ap.add_argument("--cache-root", help="Directory to reuse for all model caches")
    ap.add_argument(
        "--cache-mode",
        choices=["home", "repo", "env"],
        default=cfg.get("cache_mode") if isinstance(cfg, dict) else None,
        help="Cache strategy: home=~/hf_cache (default), repo=.hf_cache in project, env=respect existing env vars",
    )
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="Fail fast when required models are not already cached (no network).",
    )
    ap.add_argument(
        "--log-level",
        default=prelim.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    ap.add_argument("--config", default=cfg_path, help="Path to YAML/JSON config file")
    # Quiet mode
    ap.add_argument("--quiet", dest="quiet", action="store_true", default=prelim.quiet, help="Reduce log noise and hide warnings (default)")
    ap.add_argument("--no-quiet", dest="quiet", action="store_false", help="Show library warnings and info logs")
    ap.add_argument(
        "--single-file-speaker",
        help="If input resolves to one audio file, label segments with this speaker name (no diarization)",
    )
    ap.add_argument(
        "--vad-on-cpu",
        action="store_true",
        help="Force WhisperX VAD stage to run on CPU (workaround for some GPU/cuDNN issues)",
    )
    ap.add_argument(
        "--pyannote-on-cpu",
        action="store_true",
        help="Force all pyannote stages (VAD + diarization) to run on CPU",
    )

    _apply_config_defaults(ap, cfg)
    args = ap.parse_args()

    # Re-apply logging config in case config/CLI changed it
    _setup_logging_and_warnings(args.log_level, args.quiet)

    # Input may come from config; ensure it exists
    effective_input = args.input or (cfg.get("input") if isinstance(cfg, dict) else None)
    if not effective_input:
        ap.error("Missing INPUT and no 'input' provided in config")

    if args.watch:
        # Keep the service resilient: never exit on uncaught exceptions
        while True:
            try:
                watch_and_transcribe(args, cfg)
            except BaseException as exc:  # noqa: PIE786
                logging.getLogger("transcriber").error("Watch loop crashed: %s", exc)
                time.sleep(int(getattr(args, "watch_interval", 10)))
                continue
        # Unreachable
        # return

    run_transcribe(
        input_path=effective_input,
        backend=args.backend,
        model_name=args.model,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        auto_batch=args.auto_batch,
        output_dir=args.output_dir,
        speaker_mapping_path=args.speaker_mapping,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        write_srt=not args.no_srt,
        write_jsonl=not args.no_jsonl,
        cache_root=args.cache_root,
        cache_mode=args.cache_mode,
        local_files_only=args.local_files_only,
        single_file_speaker=args.single_file_speaker,
        vad_on_cpu=args.vad_on_cpu,
        pyannote_on_cpu=args.pyannote_on_cpu,
        quiet=args.quiet,
        device=None if args.device == "auto" else args.device,
    )

if __name__ == "__main__":
    main()
