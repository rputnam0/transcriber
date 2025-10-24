from __future__ import annotations

import argparse
import json
import sys
import logging
import os
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import time
from tqdm import tqdm

from .audio import gather_inputs, cleanup_tmp, is_audio_file
from .consolidate import consolidate, save_outputs, choose_speaker
from .speaker_bank import SpeakerBank, SpeakerBankConfig

# Keep global references to preloaded CUDA libraries so they aren't dlclosed
_CUDA_PRELOAD_HANDLES: list = []

def _ensure_cuda_libs_on_path() -> None:
    """Expose bundled CUDA libraries (cudnn/cublas) to the dynamic loader.

    Many Python wheels (torch, onnxruntime-gpu) rely on split NVIDIA libraries
    provided by pip packages like `nvidia-cudnn-cu12` and `nvidia-cublas-cu12`.
    When these aren't discoverable, users may see errors such as:
      Unable to load any of {libcudnn_cnn.so.9.1.0, libcudnn_cnn.so.9.1, ...}

    This helper prepends their lib directories to LD_LIBRARY_PATH at runtime.
    """
    try:
        import sys as _sys
        from pathlib import Path as _Path

        lib_dirs = []
        # Probe sys.path for namespace packages installed by pip
        for root in map(_Path, _sys.path):
            for sub in (
                root / "nvidia" / "cudnn" / "lib",
                root / "nvidia" / "cublas" / "lib",
            ):
                if sub.exists() and sub.is_dir():
                    lib_dirs.append(str(sub.resolve()))
        if not lib_dirs:
            return
        current = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in current.split(":") if p]
        for d in lib_dirs:
            if d not in parts:
                parts.insert(0, d)
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
    except Exception:
        # Best-effort; don't crash CLI on path adjustments
        pass

def _preload_cudnn_libs() -> None:
    """Attempt to preload cuDNN split libraries to avoid lazy loader issues.

    On some systems, libraries like `libcudnn_cnn.so.9` are present but not
    discovered consistently by deep dependencies (e.g., pyannote/onnxruntime).
    Explicitly dlopen the common cuDNN components.
    """
    try:
        import ctypes
        import glob
        import sys as _sys
        from pathlib import Path as _Path

        # Find candidate directories first
        lib_dirs = []
        for root in map(_Path, _sys.path):
            cand = root / "nvidia" / "cudnn" / "lib"
            if cand.exists():
                lib_dirs.append(str(cand))
        names = [
            "libcudnn.so.9",
            "libcudnn_cnn.so.9",
            "libcudnn_ops.so.9",
            "libcudnn_adv.so.9",
            "libcudnn_graph.so.9",
            "libcudnn_heuristic.so.9",
        ]
        for d in lib_dirs:
            for n in names:
                p = _Path(d) / n
                if p.exists():
                    try:
                        _CUDA_PRELOAD_HANDLES.append(ctypes.CDLL(str(p)))
                    except Exception:
                        # Best-effort; continue preloading remaining libraries
                        pass
    except Exception:
        # Ignore preload failures
        pass

def _setup_logging_and_warnings(log_level: str, quiet: bool) -> None:
    """Configure logging and warnings to reduce noise by default.

    - When quiet: show only errors, suppress Python warnings, and mute common
      third‑party loggers; keep progress bars.
    - When not quiet: honor the requested log level.
    """
    import warnings
    from matplotlib import MatplotlibDeprecationWarning  # type: ignore

    warnings.filterwarnings(
        "ignore",
        message=".*get_cmap function was deprecated.*",
        category=MatplotlibDeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Using `TRANSFORMERS_CACHE` is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="torchaudio._backend.list_audio_backends has been deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="builtin type SwigPy.* has no __module__ attribute",
        category=DeprecationWarning,
    )

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
        cwd / "transcriber.cli.yaml",
        cwd / "transcriber.cli.json",
        cwd / "config" / "transcriber.cli.yaml",
        cwd / "config" / "transcriber.cli.json",
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
        "watch_input": "watch_input",
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
    if "hf_cache_root" in cfg and cfg["hf_cache_root"] is not None:
        defaults["hf_cache_root"] = cfg["hf_cache_root"]
    if "speaker_bank_root" in cfg and cfg["speaker_bank_root"] is not None:
        defaults["speaker_bank_root"] = cfg["speaker_bank_root"]
    if "cache_root" in cfg and cfg["cache_root"] is not None:
        defaults.setdefault("hf_cache_root", cfg["cache_root"])
    sb_cfg = {}
    if isinstance(cfg, dict):
        sb_cfg = cfg.get("speaker_bank") or {}
    if isinstance(sb_cfg, dict):
        if "enabled" in sb_cfg:
            defaults["speaker_bank_enabled"] = bool(sb_cfg.get("enabled"))
        if "path" in sb_cfg and sb_cfg.get("path") is not None:
            defaults["speaker_bank_path"] = sb_cfg.get("path")
        if "threshold" in sb_cfg and sb_cfg.get("threshold") is not None:
            defaults["speaker_bank_threshold"] = float(sb_cfg.get("threshold"))
        if "radius_factor" in sb_cfg and sb_cfg.get("radius_factor") is not None:
            defaults["speaker_bank_radius_factor"] = float(sb_cfg.get("radius_factor"))
        if "use_existing" in sb_cfg:
            defaults["speaker_bank_use_existing"] = bool(sb_cfg.get("use_existing"))
        if "train_from_stems" in sb_cfg:
            defaults["speaker_bank_train_stems"] = bool(sb_cfg.get("train_from_stems"))
        if "emit_pca" in sb_cfg:
            defaults["speaker_bank_emit_pca"] = bool(sb_cfg.get("emit_pca"))
        cluster_cfg = sb_cfg.get("cluster") or {}
        if "method" in cluster_cfg and cluster_cfg.get("method") is not None:
            defaults["speaker_bank_cluster_method"] = cluster_cfg.get("method")
        if "eps" in cluster_cfg and cluster_cfg.get("eps") is not None:
            defaults["speaker_bank_cluster_eps"] = float(cluster_cfg.get("eps"))
        if "min_samples" in cluster_cfg and cluster_cfg.get("min_samples") is not None:
            defaults["speaker_bank_cluster_min_samples"] = int(cluster_cfg.get("min_samples"))
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


def _resolve_speaker_bank_paths(
    cfg: SpeakerBankConfig,
    root_override: str | None,
    hf_cache_root: str | None,
) -> Tuple[Path, str, Path]:
    base_root: Path
    if root_override:
        base_root = Path(root_override).expanduser().resolve()
    elif hf_cache_root:
        base_root = Path(hf_cache_root).expanduser().resolve()
    else:
        base_root = (Path.home() / "hf_cache").resolve()
    bank_root = base_root / "speaker_bank"
    raw_path = Path(cfg.path or "default").expanduser()
    if raw_path.is_absolute():
        profile_dir = raw_path
        profile = raw_path.name or "default"
        root = raw_path.parent
        return root, profile, profile_dir
    rel = raw_path
    if str(rel).strip() in ("", ".", "./"):
        rel = Path("default")
    if rel.parent != Path("."):
        root = (bank_root / rel.parent).resolve()
    else:
        root = bank_root.resolve()
    profile = rel.name or "default"
    profile_dir = root / profile
    return root, profile, profile_dir


def _resolve_speaker_bank_settings(
    cfg: Dict,
    args: argparse.Namespace,
) -> Tuple[Optional[SpeakerBankConfig], Optional[str]]:
    config = SpeakerBankConfig()
    if isinstance(cfg, dict):
        sb_cfg = cfg.get("speaker_bank") or {}
    else:
        sb_cfg = {}
    if isinstance(sb_cfg, dict) and sb_cfg:
        if sb_cfg.get("enabled") is not None:
            config.enabled = bool(sb_cfg.get("enabled"))
        if sb_cfg.get("path"):
            config.path = str(sb_cfg.get("path"))
        if sb_cfg.get("threshold") is not None:
            config.threshold = float(sb_cfg.get("threshold"))
        if sb_cfg.get("radius_factor") is not None:
            config.radius_factor = float(sb_cfg.get("radius_factor"))
        if sb_cfg.get("use_existing") is not None:
            config.use_existing = bool(sb_cfg.get("use_existing"))
        if sb_cfg.get("train_from_stems") is not None:
            config.train_from_stems = bool(sb_cfg.get("train_from_stems"))
        if sb_cfg.get("emit_pca") is not None:
            config.emit_pca = bool(sb_cfg.get("emit_pca"))
        cluster_cfg = sb_cfg.get("cluster") or {}
        if cluster_cfg.get("method"):
            config.cluster_method = str(cluster_cfg.get("method"))
        if cluster_cfg.get("eps") is not None:
            config.cluster_eps = float(cluster_cfg.get("eps"))
        if cluster_cfg.get("min_samples") is not None:
            config.cluster_min_samples = int(cluster_cfg.get("min_samples"))

    if getattr(args, "speaker_bank_enabled", None) is not None:
        config.enabled = bool(args.speaker_bank_enabled)
    if getattr(args, "speaker_bank_path", None):
        config.path = str(args.speaker_bank_path)
    if getattr(args, "speaker_bank_threshold", None) is not None:
        config.threshold = float(args.speaker_bank_threshold)
    if getattr(args, "speaker_bank_radius_factor", None) is not None:
        config.radius_factor = float(args.speaker_bank_radius_factor)
    if getattr(args, "speaker_bank_use_existing", None) is not None:
        config.use_existing = bool(args.speaker_bank_use_existing)
    if getattr(args, "speaker_bank_train_stems", None) is not None:
        config.train_from_stems = bool(args.speaker_bank_train_stems)
    if getattr(args, "speaker_bank_emit_pca", None) is not None:
        config.emit_pca = bool(args.speaker_bank_emit_pca)
    if getattr(args, "speaker_bank_cluster_method", None):
        config.cluster_method = str(args.speaker_bank_cluster_method)
    if getattr(args, "speaker_bank_cluster_eps", None) is not None:
        config.cluster_eps = float(args.speaker_bank_cluster_eps)
    if getattr(args, "speaker_bank_cluster_min_samples", None) is not None:
        config.cluster_min_samples = int(args.speaker_bank_cluster_min_samples)

    train_only = getattr(args, "speaker_bank_train_only", None)
    if train_only:
        train_only = str(train_only)

    if not config.enabled and not train_only:
        return None, train_only
    return config, train_only


def run_speaker_bank_training(
    input_path: str,
    hf_cache_root: str | None,
    cache_mode: str | None,
    local_files_only: bool,
    backend: str,
    model_name: str,
    compute_type: str,
    batch_size: int,
    auto_batch: bool,
    vad_on_cpu: bool,
    pyannote_on_cpu: bool,
    quiet: bool,
    device: str | None,
    speaker_bank_config: SpeakerBankConfig,
    speaker_mapping_path: str | None = None,
    speaker_bank_root_override: str | None = None,
) -> None:
    if backend != "whisperx":
        raise SystemExit("Speaker bank training requires the whisperx backend")

    from .whisperx_backend import extract_speaker_embeddings, _detect_device as _wx_detect_device

    _ensure_cuda_libs_on_path()
    _preload_cudnn_libs()
    logger = logging.getLogger("transcriber")

    cache_root_resolved = _resolve_cache_root(hf_cache_root, cache_mode)
    if cache_root_resolved:
        hub_dir = Path(os.path.expanduser(cache_root_resolved)).resolve()
        hub_dir.mkdir(parents=True, exist_ok=True)
        if hub_dir.name == "hub" and hub_dir.exists():
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_dir)
            os.environ["HF_HOME"] = str(hub_dir.parent)
        else:
            os.environ["HF_HOME"] = str(hub_dir)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_dir / "hub")
        os.environ.setdefault("HF_DATASETS_CACHE", str(hub_dir / "datasets"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hub_dir / "transformers"))
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    files, tmp_root = gather_inputs(input_path)
    if not files:
        raise SystemExit(f"No audio files found in: {input_path}")

    logger.warning(
        "Speaker bank training start: %d file(s) from %s (cache=%s)",
        len(files),
        input_path,
        cache_root_resolved or "default",
    )

    try:
        mapping_pre = _load_yaml_or_json(speaker_mapping_path)
    except BaseException as exc:  # noqa: PIE786
        logger.error("Speaker bank training: failed to load mapping %s: %s", speaker_mapping_path, exc)
        mapping_pre = {}

    def _resolve_bank_profile_paths(cfg: SpeakerBankConfig) -> Tuple[Path, str, Path]:
        base_cache = Path(cache_root_resolved).expanduser() if cache_root_resolved else Path.home() / "hf_cache"
        bank_root = base_cache / "speaker_bank"
        raw_path = Path(cfg.path or "default").expanduser()
        if raw_path.is_absolute():
            profile_dir = raw_path
            profile = raw_path.name or "default"
            root = raw_path.parent
            return root, profile, profile_dir
        rel = raw_path
        if str(rel).strip() in ("", ".", "./"):
            rel = Path("default")
        if rel.parent != Path("."):
            root = (bank_root / rel.parent).resolve()
        else:
            root = bank_root.resolve()
        profile = rel.name or "default"
        profile_dir = root / profile
        return root, profile, profile_dir

    bank_root, bank_profile, bank_profile_dir = _resolve_speaker_bank_paths(
        speaker_bank_config,
        speaker_bank_root_override,
        cache_root_resolved,
    )
    logger.info(
        "Speaker bank storage root: %s (profile=%s)",
        bank_root,
        bank_profile,
    )
    speaker_bank = SpeakerBank(
        bank_root,
        profile=bank_profile,
        cluster_method=speaker_bank_config.cluster_method,
        dbscan_eps=speaker_bank_config.cluster_eps,
        dbscan_min_samples=speaker_bank_config.cluster_min_samples,
    )
    summary = {
        "profile": str(bank_profile_dir),
        "initial": speaker_bank.summary(),
        "files": {},
    }

    hf_token = (
        os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )
    device_guess = device if device in {"cpu", "cuda"} else _wx_detect_device()
    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        cuda_available = False
        logger.debug("Speaker bank training: CUDA availability check failed: %s", exc)
    if device == "cuda" and not cuda_available:
        logger.warning(
            "Speaker bank training requested CUDA but it is unavailable; falling back to CPU."
        )
        device_guess = "cpu"
    logger.warning(
        "Speaker bank training using device=%s (vad_on_cpu=%s, pyannote_on_cpu=%s)",
        device_guess,
        vad_on_cpu,
        pyannote_on_cpu,
    )

    use_tqdm = _tqdm_enabled() and not quiet
    iter_files = files
    progress_bar = None
    if use_tqdm:
        from tqdm import tqdm as _tqdm

        progress_bar = _tqdm(
            files,
            desc="Speaker bank training",
            unit="file",
            bar_format="{l_bar}{bar} | ETA: {remaining} | {n_fmt}/{total_fmt}",
            dynamic_ncols=True,
        )
        iter_files = progress_bar

    total_added = 0
    for path in iter_files:
        label, _matched = choose_speaker(path, mapping_pre, return_match=True)
        logger.info("Training: extracting embeddings for %s (label=%s)", Path(path).name, label)
        try:
            embeddings, _ = extract_speaker_embeddings(
                path,
                hf_token=hf_token,
                min_speakers=1,
                max_speakers=1,
                pyannote_on_cpu=pyannote_on_cpu,
                force_device=device_guess,
                quiet=quiet,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Speaker bank training: failed to extract embeddings for %s: %s", Path(path).name, exc)
            continue
        added = 0
        for diar_label, vec in embeddings.items():
            speaker_bank.extend(
                [
                    (
                        label,
                        np.asarray(vec, dtype=np.float32),
                        Path(path).name,
                        {"diar_label": diar_label, "mode": "train_command"},
                    )
                ]
            )
            added += 1
            total_added += 1
        summary["files"][Path(path).name] = {
            "speaker": label,
            "embeddings_added": added,
        }
        logger.info(
            "Training: %s -> %d embedding(s) appended (total=%d)",
            Path(path).name,
            added,
            total_added,
        )
        if progress_bar:
            progress_bar.set_postfix_str(f"added={total_added}")

    if progress_bar:
        progress_bar.close()

    if total_added:
        speaker_bank.save()
        summary["final"] = speaker_bank.summary()
        if speaker_bank_config.emit_pca:
            try:
                pca_path = speaker_bank.render_pca(bank_profile_dir / "pca.png")
                if pca_path:
                    summary.setdefault("artifacts", {})
                    summary["artifacts"]["pca"] = str(pca_path)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Speaker bank training: PCA render failed: %s", exc)
        summary_path = bank_profile_dir / f"{bank_profile}.training_summary.json"
        try:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            logger.warning(
                "Speaker bank training complete: %d embeddings added (summary at %s)",
                total_added,
                summary_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Speaker bank training: failed to persist summary %s: %s", summary_path, exc)
    else:
        logger.info("Speaker bank training: no embeddings extracted from %s", input_path)

    cleanup_tmp(tmp_root)


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
    hf_cache_root: str | None = None,
    speaker_bank_root: str | None = None,
    local_files_only: bool = False,
    single_file_speaker: str | None = None,
    vad_on_cpu: bool = False,
    pyannote_on_cpu: bool = False,
    quiet: bool = True,
    auto_batch: bool = True,
    cache_mode: str | None = None,
    device: str | None = None,
    speaker_bank_config: SpeakerBankConfig | None = None,
) -> None:
    # Ensure cuDNN/cuBLAS split libraries are visible to the loader
    _ensure_cuda_libs_on_path()
    _preload_cudnn_libs()
    logger = logging.getLogger("transcriber")
    # Track whether the user explicitly provided a cache root
    user_provided_cache = bool(hf_cache_root)
    hf_cache_root = _resolve_cache_root(hf_cache_root, cache_mode)
    # Avoid requiring hf_transfer across all environments by default. Users can override.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    # Configure HF cache environment to ensure reuse and offline behavior for hub downloads
    if hf_cache_root:
        # Expand and normalize (~, symlinks)
        hub_dir = Path(os.path.expanduser(hf_cache_root)).resolve()
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
    if hf_cache_root:
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
    if hf_cache_root:
        root_path = Path(os.path.expanduser(hf_cache_root)).resolve()
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
    file_labels: Dict[str, str] = {}
    mapping_hits: Dict[str, bool] = {}
    if mapping_pre:
        for f in files:
            label, matched = choose_speaker(f, mapping_pre, return_match=True)
            file_labels[f] = label
            mapping_hits[f] = matched
    else:
        for f in files:
            file_labels[f] = Path(f).stem
            mapping_hits[f] = False

    speaker_bank: Optional[SpeakerBank] = None
    speaker_bank_summary: Dict[str, object] = {}
    speaker_bank_profile_dir: Optional[Path] = None
    speaker_bank_training_entries: List[Tuple[str, object, Dict[str, object]]] = []
    speaker_bank_modified = False
    speaker_bank_debug_by_file: Dict[str, Dict[str, object]] = {}
    rendered_pca_path: Optional[Path] = None

    def _apply_speaker_bank(
        segments: List[dict],
        embeddings: Dict[str, np.ndarray],
        file_key: str,
    ) -> Dict[str, object]:
        summary = {
            "attempted": len(embeddings),
            "matched": 0,
            "matches": {},
            "segment_counts": {"matched": 0, "unknown": 0},
        }
        if not speaker_bank or not speaker_bank_config or not embeddings:
            return summary
        if not speaker_bank_config.use_existing:
            return summary

        label_matches: Dict[str, Optional[Dict[str, object]]] = {}
        for label, vector in embeddings.items():
            try:
                match = speaker_bank.match(
                    vector,
                    threshold=speaker_bank_config.threshold,
                    radius_factor=speaker_bank_config.radius_factor,
                )
            except Exception as exc:  # noqa: BLE001
                logging.getLogger("transcriber").warning(
                    "Speaker bank match failed for %s (%s): %s",
                    Path(file_key).name,
                    label,
                    exc,
                )
                match = None
            if match:
                summary["matched"] += 1
                label_matches[label] = match
            else:
                label_matches[label] = None
            summary["matches"][label] = match

        matched_segments = 0
        unknown_segments = 0
        for seg in segments:
            raw_label = seg.get("speaker")
            if raw_label is None:
                continue
            seg["speaker_raw"] = raw_label
            if raw_label not in label_matches:
                continue
            match = label_matches[raw_label]
            if match:
                seg["speaker"] = match["speaker"]
                seg["speaker_match"] = {
                    "speaker": match["speaker"],
                    "score": float(match["score"]),
                    "cluster_id": match["cluster_id"],
                    "distance": float(match["distance"]),
                    "source": "speaker_bank",
                    "label": raw_label,
                }
                seg["speaker_match_score"] = float(match["score"])
                seg["speaker_match_distance"] = float(match["distance"])
                seg["speaker_match_cluster"] = match["cluster_id"]
                seg["speaker_match_source"] = "speaker_bank"
                matched_segments += 1
            else:
                seg["speaker"] = "unknown"
                seg["speaker_match"] = {
                    "speaker": None,
                    "score": None,
                    "cluster_id": None,
                    "distance": None,
                    "source": "speaker_bank",
                    "label": raw_label,
                }
                seg["speaker_match_score"] = None
                seg["speaker_match_distance"] = None
                seg["speaker_match_cluster"] = None
                seg["speaker_match_source"] = "speaker_bank"
                unknown_segments += 1
        summary["segment_counts"]["matched"] = matched_segments
        summary["segment_counts"]["unknown"] = unknown_segments
        if summary["attempted"]:
            logging.getLogger("transcriber").info(
                "Speaker bank matched %d/%d diar speakers for %s",
                summary["matched"],
                summary["attempted"],
                Path(file_key).name,
            )
        return summary

    if speaker_bank_config:
        bank_root, bank_profile, bank_profile_dir = _resolve_speaker_bank_paths(
            speaker_bank_config,
            speaker_bank_root,
            hf_cache_root,
        )
        speaker_bank_profile_dir = bank_profile_dir
        speaker_bank = SpeakerBank(
            bank_root,
            profile=bank_profile,
            cluster_method=speaker_bank_config.cluster_method,
            dbscan_eps=speaker_bank_config.cluster_eps,
            dbscan_min_samples=speaker_bank_config.cluster_min_samples,
        )
        bank_info = speaker_bank.summary()
        speaker_bank_summary = {
            "profile": str(bank_profile_dir),
            "initial": bank_info,
            "config": {
                "threshold": speaker_bank_config.threshold,
                "radius_factor": speaker_bank_config.radius_factor,
                "use_existing": speaker_bank_config.use_existing,
                "train_from_stems": speaker_bank_config.train_from_stems,
            },
            "files": {},
        }
        logging.getLogger("transcriber").info(
            "Speaker bank profile=%s (entries=%s, speakers=%s)",
            speaker_bank_profile_dir,
            bank_info.get("entries"),
            len(bank_info.get("speakers", [])),
        )

    if backend == "whisperx":
        # Import after environment setup so Hugging Face picks up HF_* vars
        from .whisperx_backend import (
            transcribe_with_whisperx,
            extract_speaker_embeddings,
            _detect_device as _wx_detect_device,
        )
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
        try:
            import torch  # type: ignore

            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:  # noqa: BLE001
            cuda_available = False
            logger.debug("CUDA availability check failed: %s", exc)
        if device == "cuda" and not cuda_available:
            logger.warning(
                "CUDA requested but unavailable; falling back to CPU for transcription. "
                "Check GPU drivers and visibility."
            )
            device_guess = "cpu"
        eff_bs = batch_size
        if auto_batch:
            eff_bs = _recommend_batch_size(device_guess, model_name, compute_type, user_hint=batch_size)
        iter_files = files
        use_tqdm = _tqdm_enabled()
        mapping_covers_all = bool(mapping_pre) and all(mapping_hits.get(f, False) for f in files)
        diarization_forced = (min_speakers is not None) or (max_speakers is not None)
        multi_track_zip = tmp_root is not None and len(files) > 1
        enable_diarization = True
        if multi_track_zip and not diarization_forced:
            enable_diarization = False
            logger.info(
                "Skipping diarization: multi-track ZIP input treated as per-speaker stems (%d tracks).",
                len(files),
            )
            missing = [
                Path(f).name
                for f in files
                if mapping_pre and not mapping_hits.get(f, False)
            ]
            if missing:
                logger.warning(
                    "Speaker mapping missing entries for %s — using raw stems as labels.",
                    ", ".join(missing),
                )
        elif mapping_covers_all and not diarization_forced:
            enable_diarization = False
            logger.info(
                "Skipping diarization: speaker mapping resolved all input files (%d/%d).",
                len(files),
                len(files),
            )
        elif mapping_covers_all and diarization_forced:
            logger.info(
                "Speaker mapping covers all inputs, but diarization remains enabled due to explicit speaker count constraints."
            )
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
            segs, diar, speaker_embeddings = transcribe_with_whisperx(
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
                strict_cuda=(device == "cuda" and device_guess == "cuda"),
                enable_diarization=enable_diarization,
            )
            if pbar is not None:
                pbar.close()
            dur = int(time.time() - start_ts)
            logger.warning("Finished: %s in %ds (segments=%d)", Path(f).name, dur, len(segs))
            summary = {
                "attempted": len(speaker_embeddings),
                "matched": 0,
                "matches": {},
                "segment_counts": {"matched": 0, "unknown": 0},
            }
            if speaker_embeddings:
                bank_summary = _apply_speaker_bank(segs, speaker_embeddings, f)
                summary.update(bank_summary)
            speaker_bank_debug_by_file[f] = summary
            if speaker_bank_summary is not None and "files" in speaker_bank_summary:
                speaker_bank_summary["files"][Path(f).name] = summary

            if (
                speaker_bank
                and speaker_bank_config
                and speaker_bank_config.train_from_stems
                and multi_track_zip
            ):
                training_label = file_labels.get(f, Path(f).stem)
                training_embeddings = speaker_embeddings
                if not training_embeddings:
                    try:
                        training_embeddings, _ = extract_speaker_embeddings(
                            f,
                            hf_token=hf_token,
                            min_speakers=1,
                            max_speakers=1,
                            pyannote_on_cpu=pyannote_on_cpu,
                            force_device=device_guess,
                            quiet=quiet,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Speaker bank training failed for %s: %s",
                            Path(f).name,
                            exc,
                        )
                        training_embeddings = {}
                added = 0
                for diar_label, vec in training_embeddings.items():
                    speaker_bank_training_entries.append(
                        (
                            training_label,
                            vec,
                            {
                                "file": str(Path(f).name),
                                "diar_label": diar_label,
                                "mode": "train_from_stems",
                            },
                        )
                    )
                    added += 1
                if added:
                    summary.setdefault("training", {})
                    summary["training"]["embeddings_added"] = added
                    summary["training"]["speaker"] = training_label
                    speaker_bank_modified = True

            per_file_segments.append((f, segs))
            if diar:
                diar_by_file[f] = diar

        if speaker_bank and speaker_bank_training_entries:
            enrollment: List[Tuple[str, np.ndarray, Optional[str], Dict[str, object]]] = []
            for speaker_label, vec, meta in speaker_bank_training_entries:
                vector = np.asarray(vec, dtype=np.float32)
                source = meta.get("file") if isinstance(meta, dict) else None
                enrollment.append((speaker_label, vector, source, meta if isinstance(meta, dict) else {}))
            if enrollment:
                speaker_bank.extend(enrollment)
                speaker_bank_modified = True
                speaker_bank_summary.setdefault("training", {})
                speaker_bank_summary["training"]["embeddings_added"] = speaker_bank_summary.get("training", {}).get("embeddings_added", 0) + len(enrollment)

        if speaker_bank and speaker_bank_modified:
            try:
                speaker_bank.save()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to save speaker bank profile %s: %s", speaker_bank_profile_dir, exc)

        if speaker_bank and speaker_bank_summary is not None:
            speaker_bank_summary["final"] = speaker_bank.summary()

        if (
            speaker_bank
            and speaker_bank_config
            and speaker_bank_config.emit_pca
            and speaker_bank_summary
        ):
            try:
                rendered_pca_path = speaker_bank.render_pca(speaker_bank_profile_dir / "pca.png")
                if rendered_pca_path:
                    speaker_bank_summary.setdefault("artifacts", {})
                    speaker_bank_summary["artifacts"]["pca"] = str(rendered_pca_path)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Speaker bank PCA rendering failed: %s", exc)
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
        # when speaker bank matching is not overriding the diarization labels.
        multi_file = len(files) > 1
        for i, (fname, segs) in enumerate(per_file_segments):
            file_label = choose_speaker(fname, mapping)
            for s in segs:
                if multi_file:
                    matched_by_bank = s.get("speaker_match_source") == "speaker_bank"
                    if matched_by_bank and s.get("speaker") not in (None, "unknown"):
                        continue
                    if matched_by_bank and s.get("speaker") == "unknown":
                        continue
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

    if speaker_bank_summary:
        try:
            speaker_bank_summary["files_debug"] = {
                Path(fname).name: data for fname, data in speaker_bank_debug_by_file.items()
            }
        except Exception:
            speaker_bank_summary.setdefault("files_debug", {})
        debug_path = Path(final_out_dir) / f"{base}.speaker_bank.json"
        try:
            debug_path.write_text(json.dumps(speaker_bank_summary, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to write speaker bank summary %s: %s", debug_path, exc)
        if rendered_pca_path and rendered_pca_path.exists():
            target_pca = Path(final_out_dir) / f"{base}.speaker_bank.pca.png"
            try:
                shutil.copy2(rendered_pca_path, target_pca)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to copy speaker bank PCA plot to %s: %s", target_pca, exc)

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


def watch_and_transcribe(
    args: argparse.Namespace,
    cfg: Dict,
    speaker_bank_config: Optional[SpeakerBankConfig],
) -> None:
    """Continuously scan an input directory for new audio/ZIP files and transcribe them.

    A file is processed when its corresponding output TXT is missing and the file appears
    stable (no recent modification for `watch_stability` seconds).
    """
    logger = logging.getLogger("transcriber")
    interval = int(getattr(args, "watch_interval", 10))
    stability = int(getattr(args, "watch_stability", 5))

    try:
        while True:
            if isinstance(cfg, dict):
                cfg_watch_input = cfg.get("watch_input")
                cfg_input = cfg.get("input")
            else:
                cfg_watch_input = None
                cfg_input = None
            input_root = (
                getattr(args, "watch_input", None)
                or cfg_watch_input
                or args.input
                or cfg_input
            )
            if not input_root:
                logger.error(
                    "Watch: missing INPUT/--watch-input and no 'input'/'watch_input' in config; sleeping %ss",
                    interval,
                )
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
                            hf_cache_root=args.hf_cache_root or args.cache_root,
                            speaker_bank_root=args.speaker_bank_root,
                            cache_mode=args.cache_mode,
                            local_files_only=args.local_files_only,
                            single_file_speaker=args.single_file_speaker,
                            vad_on_cpu=args.vad_on_cpu,
                            pyannote_on_cpu=args.pyannote_on_cpu,
                            quiet=args.quiet,
                            auto_batch=args.auto_batch,
                            device=None if getattr(args, "device", "auto") == "auto" else args.device,
                            speaker_bank_config=speaker_bank_config,
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
        raise


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
    ap.add_argument(
        "--watch-input",
        help="Override the directory to monitor in watch mode (defaults to INPUT/config input)",
    )
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--speaker-mapping", dest="speaker_mapping", help="YAML/JSON mapping of speaker IDs -> names")
    ap.add_argument("--min-speakers", type=int)
    ap.add_argument("--max-speakers", type=int)
    ap.add_argument("--no-srt", action="store_true", help="Don't write combined SRT")
    ap.add_argument("--no-jsonl", action="store_true", help="Don't write JSONL")
    ap.add_argument("--cache-root", help="Directory to reuse for all model caches")
    ap.add_argument(
        "--hf-cache-root",
        dest="hf_cache_root",
        help="Override Hugging Face model cache root (falls back to --cache-root)",
    )
    ap.add_argument(
        "--speaker-bank-root",
        dest="speaker_bank_root",
        help="Base directory for storing speaker bank profiles",
    )
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
    ap.add_argument(
        "--speaker-bank",
        dest="speaker_bank_enabled",
        action="store_true",
        default=None,
        help="Enable speaker bank matching (default: on when configured)",
    )
    ap.add_argument(
        "--no-speaker-bank",
        dest="speaker_bank_enabled",
        action="store_false",
        help="Disable speaker bank matching",
    )
    ap.add_argument("--speaker-bank-path", help="Speaker bank profile name or absolute path")
    ap.add_argument(
        "--speaker-bank-threshold",
        type=float,
        help="Cosine similarity threshold to accept a speaker match",
    )
    ap.add_argument(
        "--speaker-bank-radius-factor",
        type=float,
        help="Radius multiplier used when validating cluster membership",
    )
    ap.add_argument(
        "--speaker-bank-use-existing",
        dest="speaker_bank_use_existing",
        action="store_true",
        default=None,
        help="Apply existing speaker bank profiles without updating them",
    )
    ap.add_argument(
        "--speaker-bank-no-use-existing",
        dest="speaker_bank_use_existing",
        action="store_false",
        help="Skip applying pre-trained speaker bank profiles",
    )
    ap.add_argument(
        "--speaker-bank-train-stems",
        dest="speaker_bank_train_stems",
        action="store_true",
        default=None,
        help="When inputs contain multi-track ZIPs, treat each track as training data",
    )
    ap.add_argument(
        "--speaker-bank-no-train-stems",
        dest="speaker_bank_train_stems",
        action="store_false",
        help="Do not consume multi-track inputs as training data",
    )
    ap.add_argument(
        "--speaker-bank-emit-pca",
        dest="speaker_bank_emit_pca",
        action="store_true",
        default=None,
        help="Export PCA scatter plots for debugging speaker clusters",
    )
    ap.add_argument(
        "--speaker-bank-no-pca",
        dest="speaker_bank_emit_pca",
        action="store_false",
        help="Skip generating PCA scatter plots",
    )
    ap.add_argument(
        "--speaker-bank-cluster-method",
        choices=["dbscan"],
        help="Clustering strategy to derive speaker personas (default: dbscan)",
    )
    ap.add_argument(
        "--speaker-bank-cluster-eps",
        type=float,
        help="DBSCAN eps value used when clustering speaker embeddings",
    )
    ap.add_argument(
        "--speaker-bank-cluster-min-samples",
        type=int,
        help="DBSCAN min_samples value for clustering speaker embeddings",
    )
    ap.add_argument(
        "--speaker-bank-train-only",
        metavar="AUDIO_PATH",
        help="Train or update the speaker bank using audio without running transcription",
    )

    _apply_config_defaults(ap, cfg)
    args = ap.parse_args()

    # Re-apply logging config in case config/CLI changed it
    _setup_logging_and_warnings(args.log_level, args.quiet)

    speaker_bank_config, train_only_path = _resolve_speaker_bank_settings(cfg, args)

    # Input may come from config; ensure it exists
    effective_input = args.input or (cfg.get("input") if isinstance(cfg, dict) else None)
    if not effective_input and not args.watch:
        ap.error("Missing INPUT and no 'input' provided in config")

    if train_only_path:
        run_speaker_bank_training(
            input_path=train_only_path,
            hf_cache_root=args.hf_cache_root or args.cache_root,
            cache_mode=args.cache_mode,
            local_files_only=args.local_files_only,
            backend=args.backend,
            model_name=args.model,
            compute_type=args.compute_type,
            batch_size=args.batch_size,
            auto_batch=args.auto_batch,
            vad_on_cpu=args.vad_on_cpu,
            pyannote_on_cpu=args.pyannote_on_cpu,
            quiet=args.quiet,
            device=None if args.device == "auto" else args.device,
            speaker_bank_config=speaker_bank_config or SpeakerBankConfig(),
            speaker_mapping_path=args.speaker_mapping,
            speaker_bank_root_override=args.speaker_bank_root,
        )
        return

    if args.watch:
        # Keep the service resilient: never exit on uncaught exceptions
        while True:
            try:
                watch_and_transcribe(args, cfg, speaker_bank_config)
            except KeyboardInterrupt:
                logging.getLogger("transcriber").warning("Watch mode interrupted by user; exiting.")
                break
            except BaseException as exc:  # noqa: PIE786
                logging.getLogger("transcriber").error("Watch loop crashed: %s", exc)
                time.sleep(int(getattr(args, "watch_interval", 10)))
                continue
        return

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
        hf_cache_root=args.hf_cache_root or args.cache_root,
        speaker_bank_root=args.speaker_bank_root,
        cache_mode=args.cache_mode,
        local_files_only=args.local_files_only,
        single_file_speaker=args.single_file_speaker,
        vad_on_cpu=args.vad_on_cpu,
        pyannote_on_cpu=args.pyannote_on_cpu,
        quiet=args.quiet,
        device=None if args.device == "auto" else args.device,
        speaker_bank_config=speaker_bank_config,
    )

if __name__ == "__main__":
    main()
