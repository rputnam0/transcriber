from __future__ import annotations

import argparse
import json
import sys
import logging
import os
import shutil
import math
import platform
import numpy as np
from pathlib import Path, PurePosixPath
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import yaml
import time
from tqdm import tqdm

from .audio import cleanup_tmp, gather_inputs, is_audio_file, load_audio_mono
from .audio_augment import AudioAugmentationConfig, build_waveform_augmenter
from .consolidate import consolidate, save_outputs, choose_speaker
from .postprocess import (
    PostProcessConfig,
    expected_completion_marker_path,
    resolve_postprocess_config,
    run_postprocess_for_transcript,
)
from .segment_classifier import (
    load_segment_classifier,
    train_segment_classifier_from_multitrack,
)
from .session_reassignment import apply_profile_to_segments
from .speaker_bank import SpeakerBank, SpeakerBankConfig
from .segments import (
    SegmentWindow,
    TrainingSegment,
    generate_windows_for_segments,
    load_segments_file,
)

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


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _supports_parakeet_cpu_runtime() -> bool:
    return _is_apple_silicon() or sys.platform == "linux"


def _resolve_backend_choice(
    backend: str,
    *,
    files: List[str],
    tmp_root: str | None,
    min_speakers: int | None,
    max_speakers: int | None,
    single_file_speaker: str | None,
    device: str | None,
    speaker_bank_config: SpeakerBankConfig | None,
) -> str:
    requested = (backend or "auto").lower()
    if requested != "auto":
        return requested

    explicit_multi_speaker = min_speakers is not None or max_speakers is not None
    multi_track_zip_input = tmp_root is not None and len(files) > 1
    speaker_aware_run = bool(
        speaker_bank_config and speaker_bank_config.enabled and not single_file_speaker
    )

    if explicit_multi_speaker or speaker_aware_run:
        return "faster"

    if device == "cuda":
        return "faster"

    if _supports_parakeet_cpu_runtime() and (multi_track_zip_input or bool(single_file_speaker)):
        return "parakeet"

    return "faster"


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
        if cfg.get("diarization_model") is not None:
            defaults["diarization_model"] = cfg.get("diarization_model")
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
        if "scoring_margin" in sb_cfg and sb_cfg.get("scoring_margin") is not None:
            defaults["speaker_bank_margin"] = float(sb_cfg.get("scoring_margin"))
        if "match_per_segment" in sb_cfg:
            defaults["speaker_bank_match_per_segment"] = bool(sb_cfg.get("match_per_segment"))
        if "match_aggregation" in sb_cfg and sb_cfg.get("match_aggregation") is not None:
            defaults["speaker_bank_match_aggregation"] = sb_cfg.get("match_aggregation")
        if "min_segments_per_label" in sb_cfg and sb_cfg.get("min_segments_per_label") is not None:
            defaults["speaker_bank_min_segments_per_label"] = int(
                sb_cfg.get("min_segments_per_label")
            )
        cluster_cfg = sb_cfg.get("cluster") or {}
        if "method" in cluster_cfg and cluster_cfg.get("method") is not None:
            defaults["speaker_bank_cluster_method"] = cluster_cfg.get("method")
        if "eps" in cluster_cfg and cluster_cfg.get("eps") is not None:
            defaults["speaker_bank_cluster_eps"] = float(cluster_cfg.get("eps"))
        if "min_samples" in cluster_cfg and cluster_cfg.get("min_samples") is not None:
            defaults["speaker_bank_cluster_min_samples"] = int(cluster_cfg.get("min_samples"))
        scoring_cfg = sb_cfg.get("scoring") or {}
        if isinstance(scoring_cfg, dict):
            if scoring_cfg.get("threshold") is not None:
                defaults["speaker_bank_threshold"] = float(scoring_cfg.get("threshold"))
            if scoring_cfg.get("margin") is not None:
                defaults["speaker_bank_margin"] = float(scoring_cfg.get("margin"))
            as_norm_cfg = scoring_cfg.get("as_norm") or {}
            if isinstance(as_norm_cfg, dict):
                if as_norm_cfg.get("enabled") is not None:
                    defaults["speaker_bank_as_norm"] = bool(as_norm_cfg.get("enabled"))
                if as_norm_cfg.get("cohort_size") is not None:
                    defaults["speaker_bank_as_norm_cohort_size"] = int(
                        as_norm_cfg.get("cohort_size")
                    )
            if scoring_cfg.get("whiten") is not None:
                defaults["speaker_bank_whiten"] = bool(scoring_cfg.get("whiten"))
        proto_cfg = sb_cfg.get("prototypes") or {}
        if isinstance(proto_cfg, dict):
            if proto_cfg.get("enabled") is not None:
                defaults["speaker_bank_prototypes"] = bool(proto_cfg.get("enabled"))
            if proto_cfg.get("per_cluster") is not None:
                defaults["speaker_bank_prototypes_per_cluster"] = int(proto_cfg.get("per_cluster"))
            if proto_cfg.get("method") is not None:
                defaults["speaker_bank_prototypes_method"] = proto_cfg.get("method")
        classifier_cfg = sb_cfg.get("classifier") or {}
        if isinstance(classifier_cfg, dict):
            if classifier_cfg.get("min_confidence") is not None:
                defaults["speaker_bank_classifier_min_confidence"] = float(
                    classifier_cfg.get("min_confidence")
                )
            if classifier_cfg.get("min_margin") is not None:
                defaults["speaker_bank_classifier_min_margin"] = float(
                    classifier_cfg.get("min_margin")
                )
            fusion_cfg = classifier_cfg.get("fusion") or {}
            if isinstance(fusion_cfg, dict):
                if fusion_cfg.get("mode") is not None:
                    defaults["speaker_bank_classifier_fusion_mode"] = str(fusion_cfg.get("mode"))
                if fusion_cfg.get("classifier_weight") is not None:
                    defaults["speaker_bank_classifier_fusion_weight"] = float(
                        fusion_cfg.get("classifier_weight")
                    )
                if fusion_cfg.get("bank_weight") is not None:
                    defaults["speaker_bank_classifier_bank_weight"] = float(
                        fusion_cfg.get("bank_weight")
                    )
            if classifier_cfg.get("model") is not None:
                defaults["speaker_bank_classifier_model"] = classifier_cfg.get("model")
            if classifier_cfg.get("c") is not None:
                defaults["speaker_bank_classifier_c"] = float(classifier_cfg.get("c"))
            if classifier_cfg.get("n_neighbors") is not None:
                defaults["speaker_bank_classifier_n_neighbors"] = int(
                    classifier_cfg.get("n_neighbors")
                )
            if classifier_cfg.get("training_mode") is not None:
                defaults["speaker_bank_classifier_training_mode"] = classifier_cfg.get(
                    "training_mode"
                )
            if classifier_cfg.get("train_enabled") is not None:
                defaults["speaker_bank_classifier_train_enabled"] = bool(
                    classifier_cfg.get("train_enabled")
                )
            if classifier_cfg.get("excluded_speakers") is not None:
                defaults["speaker_bank_classifier_excluded_speakers"] = classifier_cfg.get(
                    "excluded_speakers"
                )
            augmentation_cfg = classifier_cfg.get("augmentation") or {}
            if isinstance(augmentation_cfg, dict):
                if augmentation_cfg.get("profile") is not None:
                    defaults["speaker_bank_classifier_augmentation_profile"] = augmentation_cfg.get(
                        "profile"
                    )
                if augmentation_cfg.get("copies") is not None:
                    defaults["speaker_bank_classifier_augmentation_copies"] = int(
                        augmentation_cfg.get("copies")
                    )
                if augmentation_cfg.get("seed") is not None:
                    defaults["speaker_bank_classifier_augmentation_seed"] = int(
                        augmentation_cfg.get("seed")
                    )
            if classifier_cfg.get("clean_max_records_per_speaker_per_session") is not None:
                defaults["speaker_bank_classifier_clean_max_records_per_speaker_per_session"] = int(
                    classifier_cfg.get("clean_max_records_per_speaker_per_session")
                )
            if classifier_cfg.get("dataset_cache_dir") is not None:
                defaults["speaker_bank_classifier_dataset_cache_dir"] = classifier_cfg.get(
                    "dataset_cache_dir"
                )
            if classifier_cfg.get("input_paths") is not None:
                defaults["speaker_bank_classifier_input_paths"] = classifier_cfg.get("input_paths")
            if classifier_cfg.get("transcript_roots") is not None:
                defaults["speaker_bank_classifier_transcript_roots"] = classifier_cfg.get(
                    "transcript_roots"
                )
        train_cfg = sb_cfg.get("train") or {}
        if isinstance(train_cfg, dict):
            if "from_segments" in train_cfg:
                defaults["speaker_bank_train_from_segments"] = bool(train_cfg.get("from_segments"))
            if "segment_source" in train_cfg and train_cfg.get("segment_source") is not None:
                defaults["speaker_bank_train_segment_source"] = train_cfg.get("segment_source")
            if "min_segment_dur" in train_cfg and train_cfg.get("min_segment_dur") is not None:
                defaults["speaker_bank_min_segment_dur"] = float(train_cfg.get("min_segment_dur"))
            if "max_segment_dur" in train_cfg and train_cfg.get("max_segment_dur") is not None:
                defaults["speaker_bank_max_segment_dur"] = float(train_cfg.get("max_segment_dur"))
            if "window_size" in train_cfg and train_cfg.get("window_size") is not None:
                defaults["speaker_bank_window_size"] = float(train_cfg.get("window_size"))
            if "window_stride" in train_cfg and train_cfg.get("window_stride") is not None:
                defaults["speaker_bank_window_stride"] = float(train_cfg.get("window_stride"))
            if (
                "max_embeddings_per_speaker" in train_cfg
                and train_cfg.get("max_embeddings_per_speaker") is not None
            ):
                defaults["speaker_bank_max_embeddings"] = int(
                    train_cfg.get("max_embeddings_per_speaker")
                )
            if "vad_chunk_stems" in train_cfg:
                defaults["speaker_bank_vad_chunk_stems"] = bool(train_cfg.get("vad_chunk_stems"))
            if "pre_pad" in train_cfg and train_cfg.get("pre_pad") is not None:
                defaults["speaker_bank_pre_pad"] = float(train_cfg.get("pre_pad"))
            if "post_pad" in train_cfg and train_cfg.get("post_pad") is not None:
                defaults["speaker_bank_post_pad"] = float(train_cfg.get("post_pad"))
            if "embed_workers" in train_cfg and train_cfg.get("embed_workers") is not None:
                defaults["speaker_bank_embed_workers"] = int(train_cfg.get("embed_workers"))
            if "embed_batch_size" in train_cfg and train_cfg.get("embed_batch_size") is not None:
                defaults["speaker_bank_embed_batch_size"] = int(train_cfg.get("embed_batch_size"))
            if "segments_path" in train_cfg and train_cfg.get("segments_path") is not None:
                defaults["speaker_bank_segments_json"] = train_cfg.get("segments_path")
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


def _normalize_speaker_bank_base_root(path_value: str | Path) -> Path:
    resolved = Path(path_value).expanduser().resolve()
    if resolved.name == "hub":
        return resolved.parent
    if resolved.name == "speaker_bank":
        return resolved.parent
    return resolved


def _resolve_speaker_bank_paths(
    cfg: SpeakerBankConfig,
    root_override: str | None,
    hf_cache_root: str | None,
) -> Tuple[Path, str, Path]:
    raw_path = Path(cfg.path or "default").expanduser()
    if raw_path.is_absolute():
        profile_dir = raw_path
        profile = raw_path.name or "default"
        root = raw_path.parent
        return root, profile, profile_dir

    candidate_base_roots: List[Path] = []
    if root_override:
        candidate_base_roots.append(_normalize_speaker_bank_base_root(root_override))
    else:
        repo_root = (Path.cwd() / ".hf_cache").resolve()
        if (repo_root / "speaker_bank").exists():
            candidate_base_roots.append(repo_root)
        if hf_cache_root:
            candidate_base_roots.append(_normalize_speaker_bank_base_root(hf_cache_root))
        candidate_base_roots.append((Path.home() / "hf_cache").resolve())

    deduped_roots: List[Path] = []
    for candidate in candidate_base_roots:
        if candidate not in deduped_roots:
            deduped_roots.append(candidate)

    rel = raw_path
    if str(rel).strip() in ("", ".", "./"):
        rel = Path("default")
    profile = rel.name or "default"
    resolved_candidates: List[Tuple[Path, Path]] = []
    for base_root in deduped_roots:
        bank_root = base_root / "speaker_bank"
        if rel.parent != Path("."):
            root = (bank_root / rel.parent).resolve()
        else:
            root = bank_root.resolve()
        profile_dir = root / profile
        resolved_candidates.append((root, profile_dir))

    for root, profile_dir in resolved_candidates:
        if profile_dir.exists():
            return root, profile, profile_dir

    if resolved_candidates:
        root, profile_dir = resolved_candidates[0]
        return root, profile, profile_dir

    bank_root = ((Path.home() / "hf_cache").resolve()) / "speaker_bank"
    root = (bank_root / rel.parent).resolve() if rel.parent != Path(".") else bank_root.resolve()
    profile_dir = root / profile
    return root, profile, profile_dir


def _aggregate_segment_label_candidates(
    segment_indices: List[int],
    seg_matches: Dict[int, Dict[str, object]],
    *,
    aggregation: str,
    threshold: float,
    margin_required: float,
    min_segments_per_label: int,
) -> Tuple[Optional[Dict[str, object]], Dict[str, object]]:
    stats: Dict[str, object] = {
        "segments_embedded": 0,
        "segments_matched": 0,
        "means": {},
        "means_supported": {},
        "vote_counts": {},
        "best_ratio": None,
        "second_ratio": None,
        "margin": None,
        "second_metric": None,
        "score_metric": None,
        "selection": None,
    }

    embedded_total = 0
    candidate_totals: Dict[str, float] = defaultdict(float)
    candidate_support: Dict[str, int] = defaultdict(int)
    vote_counts: Dict[str, int] = defaultdict(int)
    best_candidate_by_speaker: Dict[str, Dict[str, object]] = {}

    for seg_idx in segment_indices:
        seg_info = seg_matches.get(seg_idx) or {}
        candidates = seg_info.get("candidates") or []
        if not candidates:
            continue
        embedded_total += 1
        top_candidate = candidates[0]
        top_speaker = str(top_candidate.get("speaker") or "")
        if top_speaker:
            vote_counts[top_speaker] += 1
        for candidate in candidates:
            speaker = str(candidate.get("speaker") or "")
            if not speaker:
                continue
            score = float(candidate.get("score") or 0.0)
            candidate_totals[speaker] += score
            candidate_support[speaker] += 1
            existing = best_candidate_by_speaker.get(speaker)
            if existing is None or score > float(existing.get("score") or 0.0):
                best_candidate_by_speaker[speaker] = {
                    "segment_index": seg_idx,
                    **candidate,
                }

    stats["segments_embedded"] = embedded_total
    stats["segments_matched"] = sum(
        1 for seg_idx in segment_indices if (seg_matches.get(seg_idx) or {}).get("accepted")
    )

    if embedded_total:
        stats["means"] = {
            speaker: total / embedded_total
            for speaker, total in sorted(
                candidate_totals.items(), key=lambda item: item[1], reverse=True
            )
        }
    stats["means_supported"] = {
        speaker: candidate_totals[speaker] / candidate_support[speaker]
        for speaker in sorted(candidate_support)
        if candidate_support[speaker]
    }
    stats["vote_counts"] = dict(sorted(vote_counts.items(), key=lambda item: item[1], reverse=True))

    if embedded_total < max(1, min_segments_per_label) or not candidate_totals:
        return None, stats

    selection: Optional[Dict[str, object]] = None
    aggregation_name = (aggregation or "mean").lower()
    if aggregation_name == "vote":
        ordered_counts = sorted(
            vote_counts.items(),
            key=lambda item: (item[1], stats["means"].get(item[0], 0.0)),  # type: ignore[union-attr]
            reverse=True,
        )
        if ordered_counts:
            best_speaker, best_count = ordered_counts[0]
            second_count = ordered_counts[1][1] if len(ordered_counts) > 1 else 0
            ratio_best = best_count / embedded_total if embedded_total else 0.0
            ratio_second = second_count / embedded_total if embedded_total else 0.0
            margin_value = ratio_best - ratio_second
            stats["best_ratio"] = ratio_best
            stats["second_ratio"] = ratio_second
            stats["margin"] = margin_value
            stats["second_metric"] = ratio_second
            score_metric = float((stats["means"] or {}).get(best_speaker, 0.0))
            stats["score_metric"] = score_metric
            if score_metric >= threshold and margin_value >= margin_required:
                best_candidate = best_candidate_by_speaker.get(best_speaker)
                if best_candidate:
                    selection = {
                        "speaker": best_speaker,
                        "cluster_id": best_candidate.get("cluster_id"),
                        "score": score_metric,
                        "score_max": best_candidate.get("score"),
                        "distance": best_candidate.get("distance"),
                        "margin": margin_value,
                        "second_best": ratio_second,
                        "source": "segment_vote",
                        "segments_count": best_count,
                    }
    else:
        ordered_means = sorted(
            ((speaker, float(score)) for speaker, score in (stats["means"] or {}).items()),
            key=lambda item: item[1],
            reverse=True,
        )
        if ordered_means:
            best_speaker, best_mean = ordered_means[0]
            second_mean = ordered_means[1][1] if len(ordered_means) > 1 else 0.0
            margin_value = best_mean - second_mean
            stats["margin"] = margin_value
            stats["second_metric"] = second_mean
            stats["score_metric"] = best_mean
            if best_mean >= threshold and margin_value >= margin_required:
                best_candidate = best_candidate_by_speaker.get(best_speaker)
                if best_candidate:
                    selection = {
                        "speaker": best_speaker,
                        "cluster_id": best_candidate.get("cluster_id"),
                        "score": best_mean,
                        "score_max": best_candidate.get("score"),
                        "distance": best_candidate.get("distance"),
                        "margin": margin_value,
                        "second_best": second_mean,
                        "source": "segment_mean",
                        "segments_count": candidate_support.get(best_speaker, 0),
                    }

    stats["selection"] = selection
    return selection, stats


def _segment_classifier_thresholds(
    *,
    duration: float,
    base_confidence: float,
    base_margin: float,
) -> Tuple[float, float]:
    min_confidence = float(base_confidence)
    min_margin = float(base_margin)
    if duration < 0.75:
        min_confidence = max(min_confidence, 0.60)
        min_margin = max(min_margin, 0.20)
    elif duration < 1.25:
        min_confidence = max(min_confidence, 0.50)
        min_margin = max(min_margin, 0.14)
    elif duration < 2.0:
        min_confidence = max(min_confidence, 0.35)
        min_margin = max(min_margin, 0.08)
    return min_confidence, min_margin


def _label_classifier_thresholds(
    *,
    segment_count: int,
    base_confidence: float,
    base_margin: float,
) -> Tuple[float, float]:
    min_confidence = float(base_confidence)
    min_margin = float(base_margin)
    if segment_count >= 6:
        min_confidence = max(min_confidence, 0.50)
        min_margin = max(min_margin, 0.14)
    elif segment_count >= 3:
        min_confidence = max(min_confidence, 0.55)
        min_margin = max(min_margin, 0.16)
    else:
        min_confidence = max(min_confidence, 0.60)
        min_margin = max(min_margin, 0.18)
    return min_confidence, min_margin


def _set_segment_speaker_label(segment: Dict[str, object], speaker: str | None) -> None:
    segment["speaker"] = speaker
    words = segment.get("words")
    if not isinstance(words, list):
        return
    for word in words:
        if not isinstance(word, dict):
            continue
        if "speaker_raw" not in word and word.get("speaker") is not None:
            word["speaker_raw"] = word.get("speaker")
        word["speaker"] = speaker


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
        if sb_cfg.get("diarization_model") is not None:
            config.diarization_model = str(sb_cfg.get("diarization_model"))
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
        if sb_cfg.get("scoring_margin") is not None:
            config.scoring_margin = float(sb_cfg.get("scoring_margin"))
        if sb_cfg.get("match_per_segment") is not None:
            config.match_per_segment = bool(sb_cfg.get("match_per_segment"))
        if sb_cfg.get("match_aggregation") is not None:
            config.match_aggregation = str(sb_cfg.get("match_aggregation"))
        if sb_cfg.get("min_segments_per_label") is not None:
            config.min_segments_per_label = int(sb_cfg.get("min_segments_per_label"))
        repair_cfg = sb_cfg.get("repair") or {}
        if isinstance(repair_cfg, dict) and repair_cfg:
            if repair_cfg.get("enabled") is not None:
                config.repair_enabled = bool(repair_cfg.get("enabled"))
            if repair_cfg.get("merge_same_raw_gap_seconds") is not None:
                config.repair_merge_same_raw_gap_seconds = float(
                    repair_cfg.get("merge_same_raw_gap_seconds")
                )
            if repair_cfg.get("snap_boundary_seconds") is not None:
                config.repair_snap_boundary_seconds = float(repair_cfg.get("snap_boundary_seconds"))
            if repair_cfg.get("max_overlap_trim_seconds") is not None:
                config.repair_max_overlap_trim_seconds = float(
                    repair_cfg.get("max_overlap_trim_seconds")
                )
            if repair_cfg.get("split_on_word_gap_seconds") is not None:
                config.repair_split_on_word_gap_seconds = float(
                    repair_cfg.get("split_on_word_gap_seconds")
                )
            if repair_cfg.get("max_seed_overlap_seconds") is not None:
                config.repair_max_seed_overlap_seconds = float(
                    repair_cfg.get("max_seed_overlap_seconds")
                )
            if repair_cfg.get("min_segment_duration_seconds") is not None:
                config.repair_min_segment_duration_seconds = float(
                    repair_cfg.get("min_segment_duration_seconds")
                )
        session_graph_cfg = sb_cfg.get("session_graph") or {}
        if isinstance(session_graph_cfg, dict) and session_graph_cfg:
            if session_graph_cfg.get("enabled") is not None:
                config.session_graph_enabled = bool(session_graph_cfg.get("enabled"))
            if session_graph_cfg.get("candidate_top_k") is not None:
                config.session_graph_candidate_top_k = int(session_graph_cfg.get("candidate_top_k"))
            if session_graph_cfg.get("candidate_floor") is not None:
                config.session_graph_candidate_floor = float(
                    session_graph_cfg.get("candidate_floor")
                )
            if session_graph_cfg.get("knn") is not None:
                config.session_graph_knn = int(session_graph_cfg.get("knn"))
            if session_graph_cfg.get("min_similarity") is not None:
                config.session_graph_min_similarity = float(session_graph_cfg.get("min_similarity"))
            if session_graph_cfg.get("anchor_weight") is not None:
                config.session_graph_anchor_weight = float(session_graph_cfg.get("anchor_weight"))
            if session_graph_cfg.get("temporal_weight") is not None:
                config.session_graph_temporal_weight = float(
                    session_graph_cfg.get("temporal_weight")
                )
            if session_graph_cfg.get("temporal_tau_seconds") is not None:
                config.session_graph_temporal_tau_seconds = float(
                    session_graph_cfg.get("temporal_tau_seconds")
                )
            if session_graph_cfg.get("temporal_max_gap_seconds") is not None:
                config.session_graph_temporal_max_gap_seconds = float(
                    session_graph_cfg.get("temporal_max_gap_seconds")
                )
            if session_graph_cfg.get("same_raw_label_weight") is not None:
                config.session_graph_same_raw_label_weight = float(
                    session_graph_cfg.get("same_raw_label_weight")
                )
            if session_graph_cfg.get("same_top1_weight") is not None:
                config.session_graph_same_top1_weight = float(
                    session_graph_cfg.get("same_top1_weight")
                )
            if session_graph_cfg.get("alpha") is not None:
                config.session_graph_alpha = float(session_graph_cfg.get("alpha"))
            if session_graph_cfg.get("max_iters") is not None:
                config.session_graph_max_iters = int(session_graph_cfg.get("max_iters"))
            if session_graph_cfg.get("tolerance") is not None:
                config.session_graph_tolerance = float(session_graph_cfg.get("tolerance"))
            if session_graph_cfg.get("strong_seed_score") is not None:
                config.session_graph_strong_seed_score = float(
                    session_graph_cfg.get("strong_seed_score")
                )
            if session_graph_cfg.get("strong_seed_margin") is not None:
                config.session_graph_strong_seed_margin = float(
                    session_graph_cfg.get("strong_seed_margin")
                )
            if session_graph_cfg.get("override_min_confidence") is not None:
                config.session_graph_override_min_confidence = float(
                    session_graph_cfg.get("override_min_confidence")
                )
            if session_graph_cfg.get("override_min_margin") is not None:
                config.session_graph_override_min_margin = float(
                    session_graph_cfg.get("override_min_margin")
                )
            if session_graph_cfg.get("override_min_delta") is not None:
                config.session_graph_override_min_delta = float(
                    session_graph_cfg.get("override_min_delta")
                )
            pair_overrides = session_graph_cfg.get("pair_overrides")
            if isinstance(pair_overrides, dict):
                normalized_pair_overrides = {}
                for pair_key, override_values in pair_overrides.items():
                    if not isinstance(override_values, dict):
                        continue
                    normalized_values = {}
                    for name, value in override_values.items():
                        if value is None:
                            continue
                        try:
                            normalized_values[str(name)] = float(value)
                        except (TypeError, ValueError):
                            continue
                    normalized_pair_overrides[str(pair_key)] = normalized_values
                config.session_graph_pair_overrides = normalized_pair_overrides
        cluster_cfg = sb_cfg.get("cluster") or {}
        if cluster_cfg.get("method"):
            config.cluster_method = str(cluster_cfg.get("method"))
        if cluster_cfg.get("eps") is not None:
            config.cluster_eps = float(cluster_cfg.get("eps"))
        if cluster_cfg.get("min_samples") is not None:
            config.cluster_min_samples = int(cluster_cfg.get("min_samples"))
        scoring_cfg = sb_cfg.get("scoring") or {}
        if isinstance(scoring_cfg, dict) and scoring_cfg:
            if scoring_cfg.get("threshold") is not None:
                config.threshold = float(scoring_cfg.get("threshold"))
            if scoring_cfg.get("margin") is not None:
                config.scoring_margin = float(scoring_cfg.get("margin"))
            if scoring_cfg.get("whiten") is not None:
                config.scoring_whiten = bool(scoring_cfg.get("whiten"))
            as_norm_cfg = scoring_cfg.get("as_norm") or {}
            if isinstance(as_norm_cfg, dict) and as_norm_cfg:
                if as_norm_cfg.get("enabled") is not None:
                    config.scoring_as_norm_enabled = bool(as_norm_cfg.get("enabled"))
                if as_norm_cfg.get("cohort_size") is not None:
                    config.scoring_as_norm_cohort_size = int(as_norm_cfg.get("cohort_size"))
        proto_cfg = sb_cfg.get("prototypes") or {}
        if isinstance(proto_cfg, dict) and proto_cfg:
            if proto_cfg.get("enabled") is not None:
                config.prototypes_enabled = bool(proto_cfg.get("enabled"))
            if proto_cfg.get("per_cluster") is not None:
                config.prototypes_per_cluster = int(proto_cfg.get("per_cluster"))
            if proto_cfg.get("method"):
                config.prototypes_method = str(proto_cfg.get("method"))
        classifier_cfg = sb_cfg.get("classifier") or {}
        if isinstance(classifier_cfg, dict) and classifier_cfg:
            if classifier_cfg.get("min_confidence") is not None:
                config.classifier_min_confidence = float(classifier_cfg.get("min_confidence"))
            if classifier_cfg.get("min_margin") is not None:
                config.classifier_min_margin = float(classifier_cfg.get("min_margin"))
            fusion_cfg = classifier_cfg.get("fusion") or {}
            if isinstance(fusion_cfg, dict) and fusion_cfg:
                if fusion_cfg.get("mode") is not None:
                    config.classifier_fusion_mode = str(fusion_cfg.get("mode"))
                if fusion_cfg.get("classifier_weight") is not None:
                    config.classifier_fusion_weight = float(fusion_cfg.get("classifier_weight"))
                if fusion_cfg.get("bank_weight") is not None:
                    config.classifier_bank_weight = float(fusion_cfg.get("bank_weight"))
            if classifier_cfg.get("model") is not None:
                config.classifier_model = str(classifier_cfg.get("model"))
            if classifier_cfg.get("c") is not None:
                config.classifier_c = float(classifier_cfg.get("c"))
            if classifier_cfg.get("n_neighbors") is not None:
                config.classifier_n_neighbors = int(classifier_cfg.get("n_neighbors"))
            if classifier_cfg.get("training_mode") is not None:
                config.classifier_training_mode = str(classifier_cfg.get("training_mode"))
            if classifier_cfg.get("train_enabled") is not None:
                config.classifier_train_enabled = bool(classifier_cfg.get("train_enabled"))
            if classifier_cfg.get("excluded_speakers") is not None:
                config.classifier_excluded_speakers = [
                    str(item)
                    for item in (classifier_cfg.get("excluded_speakers") or [])
                    if item is not None
                ]
            augmentation_cfg = classifier_cfg.get("augmentation") or {}
            if isinstance(augmentation_cfg, dict) and augmentation_cfg:
                if augmentation_cfg.get("profile") is not None:
                    config.classifier_augmentation_profile = str(augmentation_cfg.get("profile"))
                if augmentation_cfg.get("copies") is not None:
                    config.classifier_augmentation_copies = int(augmentation_cfg.get("copies"))
                if augmentation_cfg.get("seed") is not None:
                    config.classifier_augmentation_seed = int(augmentation_cfg.get("seed"))
            if classifier_cfg.get("clean_max_records_per_speaker_per_session") is not None:
                config.classifier_clean_max_records_per_speaker_per_session = int(
                    classifier_cfg.get("clean_max_records_per_speaker_per_session")
                )
            if classifier_cfg.get("dataset_cache_dir") is not None:
                cache_dir = classifier_cfg.get("dataset_cache_dir")
                config.classifier_dataset_cache_dir = str(cache_dir) if cache_dir else None
            if classifier_cfg.get("input_paths") is not None:
                config.classifier_input_paths = [
                    str(item)
                    for item in (classifier_cfg.get("input_paths") or [])
                    if item is not None
                ]
            if classifier_cfg.get("transcript_roots") is not None:
                config.classifier_transcript_roots = [
                    str(item)
                    for item in (classifier_cfg.get("transcript_roots") or [])
                    if item is not None
                ]
        train_cfg = sb_cfg.get("train") or {}
        if isinstance(train_cfg, dict) and train_cfg:
            if train_cfg.get("from_segments") is not None:
                config.train_from_segments = bool(train_cfg.get("from_segments"))
            if train_cfg.get("segment_source") is not None:
                config.train_segment_source = str(train_cfg.get("segment_source"))
            if train_cfg.get("min_segment_dur") is not None:
                config.min_segment_dur = float(train_cfg.get("min_segment_dur"))
            if train_cfg.get("max_segment_dur") is not None:
                config.max_segment_dur = float(train_cfg.get("max_segment_dur"))
            if train_cfg.get("window_size") is not None:
                config.window_size = float(train_cfg.get("window_size"))
            if train_cfg.get("window_stride") is not None:
                config.window_stride = float(train_cfg.get("window_stride"))
            if train_cfg.get("max_embeddings_per_speaker") is not None:
                config.max_embeddings_per_speaker = int(train_cfg.get("max_embeddings_per_speaker"))
            if train_cfg.get("vad_chunk_stems") is not None:
                config.vad_chunk_stems = bool(train_cfg.get("vad_chunk_stems"))
            if train_cfg.get("pre_pad") is not None:
                config.pre_pad = float(train_cfg.get("pre_pad"))
            if train_cfg.get("post_pad") is not None:
                config.post_pad = float(train_cfg.get("post_pad"))
            if train_cfg.get("embed_workers") is not None:
                config.embed_workers = int(train_cfg.get("embed_workers"))
            if train_cfg.get("embed_batch_size") is not None:
                config.embed_batch_size = int(train_cfg.get("embed_batch_size"))
            if train_cfg.get("segments_path"):
                config.segments_path = str(train_cfg.get("segments_path"))

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
    if getattr(args, "speaker_bank_margin", None) is not None:
        config.scoring_margin = float(args.speaker_bank_margin)
    if getattr(args, "speaker_bank_train_from_segments", None) is not None:
        config.train_from_segments = bool(args.speaker_bank_train_from_segments)
    if getattr(args, "speaker_bank_train_segment_source", None):
        config.train_segment_source = str(args.speaker_bank_train_segment_source)
    if getattr(args, "speaker_bank_min_segment_dur", None) is not None:
        config.min_segment_dur = float(args.speaker_bank_min_segment_dur)
    if getattr(args, "speaker_bank_max_segment_dur", None) is not None:
        config.max_segment_dur = float(args.speaker_bank_max_segment_dur)
    if getattr(args, "speaker_bank_window_size", None) is not None:
        config.window_size = float(args.speaker_bank_window_size)
    if getattr(args, "speaker_bank_window_stride", None) is not None:
        config.window_stride = float(args.speaker_bank_window_stride)
    if getattr(args, "speaker_bank_max_embeddings", None) is not None:
        config.max_embeddings_per_speaker = int(args.speaker_bank_max_embeddings)
    if getattr(args, "speaker_bank_vad_chunk_stems", None) is not None:
        config.vad_chunk_stems = bool(args.speaker_bank_vad_chunk_stems)
    if getattr(args, "speaker_bank_pre_pad", None) is not None:
        config.pre_pad = float(args.speaker_bank_pre_pad)
    if getattr(args, "speaker_bank_post_pad", None) is not None:
        config.post_pad = float(args.speaker_bank_post_pad)
    if getattr(args, "speaker_bank_embed_workers", None) is not None:
        config.embed_workers = int(args.speaker_bank_embed_workers)
    if getattr(args, "speaker_bank_embed_batch_size", None) is not None:
        config.embed_batch_size = int(args.speaker_bank_embed_batch_size)
    if getattr(args, "speaker_bank_segments_json", None):
        config.segments_path = str(args.speaker_bank_segments_json)
    if getattr(args, "speaker_bank_match_per_segment", None) is not None:
        config.match_per_segment = bool(args.speaker_bank_match_per_segment)
    if getattr(args, "speaker_bank_match_aggregation", None):
        config.match_aggregation = str(args.speaker_bank_match_aggregation)
    if getattr(args, "speaker_bank_min_segments_per_label", None) is not None:
        config.min_segments_per_label = int(args.speaker_bank_min_segments_per_label)
    if getattr(args, "speaker_bank_prototypes", None) is not None:
        config.prototypes_enabled = bool(args.speaker_bank_prototypes)
    if getattr(args, "speaker_bank_prototypes_per_cluster", None) is not None:
        config.prototypes_per_cluster = int(args.speaker_bank_prototypes_per_cluster)
    if getattr(args, "speaker_bank_prototypes_method", None):
        config.prototypes_method = str(args.speaker_bank_prototypes_method)

    config.match_aggregation = (config.match_aggregation or "mean").lower()
    config.prototypes_method = (config.prototypes_method or "central").lower()

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
    model_name: str,
    compute_type: str,
    batch_size: int,
    auto_batch: bool,
    pyannote_on_cpu: bool,
    diarization_model: str | None,
    quiet: bool,
    device: str | None,
    speaker_bank_config: SpeakerBankConfig,
    speaker_mapping_path: str | None = None,
    speaker_bank_root_override: str | None = None,
    segments_json: str | None = None,
) -> None:
    from .diarization import (
        _detect_device as _diar_detect_device,
        extract_embeddings_for_segments,
        extract_speaker_embeddings,
    )

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
        logger.error(
            "Speaker bank training: failed to load mapping %s: %s", speaker_mapping_path, exc
        )
        mapping_pre = {}

    def _resolve_bank_profile_paths(cfg: SpeakerBankConfig) -> Tuple[Path, str, Path]:
        base_cache = (
            Path(cache_root_resolved).expanduser()
            if cache_root_resolved
            else Path.home() / "hf_cache"
        )
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
        prototypes_enabled=speaker_bank_config.prototypes_enabled,
        prototypes_per_cluster=speaker_bank_config.prototypes_per_cluster,
        prototypes_method=speaker_bank_config.prototypes_method,
        scoring_whiten=speaker_bank_config.scoring_whiten,
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
    device_guess = device if device in {"cpu", "cuda"} else _diar_detect_device()
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
        "Speaker bank training using device=%s (pyannote_on_cpu=%s)", device_guess, pyannote_on_cpu
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
    segments_override_raw = segments_json or speaker_bank_config.segments_path
    segments_override_path: Optional[Path] = None
    if segments_override_raw:
        candidate = Path(segments_override_raw).expanduser()
        if candidate.exists():
            segments_override_path = candidate
        else:
            logger.warning(
                "Speaker bank training: segments path %s not found; ignoring",
                candidate,
            )

    segment_source_pref = (speaker_bank_config.train_segment_source or "auto").lower()

    excluded_speakers = {
        str(item).strip()
        for item in speaker_bank_config.classifier_excluded_speakers
        if str(item).strip()
    }
    augmentation_config = AudioAugmentationConfig(
        profile=speaker_bank_config.classifier_augmentation_profile,
        copies=max(int(speaker_bank_config.classifier_augmentation_copies or 0), 0),
        seed=int(speaker_bank_config.classifier_augmentation_seed),
    )

    def _segment_filename_candidates(base: str) -> List[str]:
        return [
            f"{base}.jsonl",
            f"{base}.json",
            f"{base}.segments.json",
            f"{base}.diarization.json",
        ]

    def _discover_segment_artifact(audio_file: Path) -> Optional[Path]:
        seen: set[Path] = set()
        candidates: List[Path] = []
        if segments_override_path is not None:
            if segments_override_path.is_file():
                candidates.append(segments_override_path)
            elif segments_override_path.is_dir():
                for name in _segment_filename_candidates(audio_file.stem):
                    candidates.append(segments_override_path / name)
        outputs_dir = Path(".outputs") / audio_file.stem
        if segment_source_pref in {"auto", "session_jsonl"}:
            candidates.append(outputs_dir / f"{audio_file.stem}.jsonl")
        if segment_source_pref in {"auto", "diarization_json"}:
            candidates.append(outputs_dir / f"{audio_file.stem}.diarization.json")
            candidates.append(outputs_dir / f"{audio_file.stem}.json")
        for candidate in candidates:
            expanded = candidate.expanduser()
            if expanded in seen:
                continue
            seen.add(expanded)
            if expanded.exists():
                return expanded
        return None

    def _segment_matches(seg: TrainingSegment, audio_file: Path) -> bool:
        seg_name = Path(seg.audio_file).name
        if not seg_name:
            return False
        return (
            seg_name == audio_file.name
            or Path(seg_name).stem == audio_file.stem
            or seg.audio_file == audio_file.name
            or seg.audio_file == audio_file.stem
        )

    for path in iter_files:
        audio_path = Path(path)
        audio_name = audio_path.name
        file_summary: Dict[str, object] = {
            "mode": "legacy",
            "segments_source": None,
            "windows_total": 0,
        }

        use_segments = speaker_bank_config.train_from_segments or segments_override_path is not None
        segment_artifact: Optional[Path] = None
        windows: List[SegmentWindow] = []

        if use_segments:
            segment_artifact = _discover_segment_artifact(audio_path)
            if segment_artifact:
                try:
                    loaded_segments = load_segments_file(segment_artifact, mapping_pre or {})
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Speaker bank training: failed to load segments from %s: %s",
                        segment_artifact,
                        exc,
                    )
                    loaded_segments = []
                matching_segments = [
                    seg
                    for seg in loaded_segments
                    if _segment_matches(seg, audio_path) and seg.speaker not in excluded_speakers
                ]
                if matching_segments:
                    windows = generate_windows_for_segments(
                        matching_segments,
                        min_duration=speaker_bank_config.min_segment_dur,
                        max_duration=speaker_bank_config.max_segment_dur,
                        window_size=speaker_bank_config.window_size,
                        window_stride=speaker_bank_config.window_stride,
                    )
                    file_summary["mode"] = "segments"
                    file_summary["segments_source"] = str(segment_artifact)
                else:
                    logger.debug(
                        "Speaker bank training: no matching segments for %s in %s",
                        audio_name,
                        segment_artifact,
                    )
            elif speaker_bank_config.train_from_segments:
                logger.warning(
                    "Speaker bank training: segment metadata not found for %s (source=%s)",
                    audio_name,
                    segment_source_pref,
                )

        if not windows and speaker_bank_config.vad_chunk_stems:
            fallback_label, _ = choose_speaker(audio_name, mapping_pre, return_match=True)
            if fallback_label and fallback_label not in excluded_speakers:
                try:
                    waveform = load_audio_mono(path, sample_rate=16000)
                    duration = len(waveform) / 16000.0
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Speaker bank training: failed to load audio %s for chunking: %s",
                        audio_name,
                        exc,
                    )
                    duration = 0.0
                if duration >= speaker_bank_config.min_segment_dur:
                    base_segment = TrainingSegment(
                        audio_file=audio_name,
                        start=0.0,
                        end=duration,
                        speaker=fallback_label,
                        speaker_raw=fallback_label,
                    )
                    windows = generate_windows_for_segments(
                        [base_segment],
                        min_duration=speaker_bank_config.min_segment_dur,
                        max_duration=speaker_bank_config.max_segment_dur,
                        window_size=speaker_bank_config.window_size,
                        window_stride=speaker_bank_config.window_stride,
                    )
                    if windows:
                        file_summary["mode"] = "segments"

        windows_by_speaker: Dict[str, List[SegmentWindow]] = defaultdict(list)
        for win in windows:
            if win.speaker in excluded_speakers:
                continue
            windows_by_speaker[win.speaker].append(win)

        selected_windows: List[SegmentWindow] = []
        limit = speaker_bank_config.max_embeddings_per_speaker
        for speaker_label, win_list in windows_by_speaker.items():
            win_list.sort(key=lambda w: (w.start, w.segment_index, w.window_index))
            if limit and limit > 0 and len(win_list) > limit:
                step = max(1, math.ceil(len(win_list) / limit))
                sampled = win_list[::step]
                if len(sampled) > limit:
                    sampled = sampled[:limit]
                selected_windows.extend(sampled)
            else:
                selected_windows.extend(win_list)

        selected_windows.sort(key=lambda w: (w.start, w.segment_index, w.window_index))
        file_summary["windows_total"] = len(selected_windows)

        if selected_windows:
            payload: List[Tuple[float, float, str]] = []
            window_lookup: Dict[int, SegmentWindow] = {}
            for idx, win in enumerate(selected_windows):
                payload.append((win.start, win.end, win.speaker))
                window_lookup[idx] = win

            embed_results, embed_summary = extract_embeddings_for_segments(
                path,
                payload,
                hf_token=hf_token,
                diarization_model_name=diarization_model or speaker_bank_config.diarization_model,
                force_device=device_guess,
                quiet=quiet,
                pre_pad=speaker_bank_config.pre_pad,
                post_pad=speaker_bank_config.post_pad,
                batch_size=speaker_bank_config.embed_batch_size,
                workers=speaker_bank_config.embed_workers,
            )

            speaker_counts: Dict[str, int] = defaultdict(int)
            added = 0
            for result in embed_results:
                win = window_lookup.get(result.index)
                if win is None:
                    continue
                vector = np.asarray(result.embedding, dtype=np.float32)
                speaker_bank.extend(
                    [
                        (
                            win.speaker,
                            vector,
                            audio_name,
                            {
                                "mode": "train_from_segments",
                                "segment_index": win.segment_index,
                                "window_index": win.window_index,
                                "start": win.start,
                                "end": win.end,
                                "speaker_raw": win.speaker_raw,
                                "segments_source": file_summary["segments_source"],
                            },
                        )
                    ]
                )
                speaker_counts[win.speaker] += 1
                added += 1
                total_added += 1

            if augmentation_config.enabled:
                for pass_index in range(augmentation_config.copies):
                    waveform_augmenter = build_waveform_augmenter(
                        augmentation_config,
                        domain="bank",
                        pass_index=pass_index,
                    )
                    if waveform_augmenter is None:
                        continue
                    aug_results, aug_summary = extract_embeddings_for_segments(
                        path,
                        payload,
                        hf_token=hf_token,
                        diarization_model_name=diarization_model
                        or speaker_bank_config.diarization_model,
                        force_device=device_guess,
                        quiet=quiet,
                        pre_pad=speaker_bank_config.pre_pad,
                        post_pad=speaker_bank_config.post_pad,
                        batch_size=speaker_bank_config.embed_batch_size,
                        workers=speaker_bank_config.embed_workers,
                        waveform_transform=waveform_augmenter,
                    )
                    for result in aug_results:
                        win = window_lookup.get(result.index)
                        if win is None:
                            continue
                        vector = np.asarray(result.embedding, dtype=np.float32)
                        speaker_bank.extend(
                            [
                                (
                                    win.speaker,
                                    vector,
                                    audio_name,
                                    {
                                        "mode": "train_from_segments_aug",
                                        "augmentation_profile": augmentation_config.profile,
                                        "augmentation_pass": pass_index,
                                        "segment_index": win.segment_index,
                                        "window_index": win.window_index,
                                        "start": win.start,
                                        "end": win.end,
                                        "speaker_raw": win.speaker_raw,
                                        "segments_source": file_summary["segments_source"],
                                    },
                                )
                            ]
                        )
                        speaker_counts[win.speaker] += 1
                        added += 1
                        total_added += 1
                    file_summary.setdefault("augmentation_runs", []).append(
                        {
                            "pass_index": pass_index,
                            "embedded": aug_summary.get("embedded"),
                            "skipped": aug_summary.get("skipped"),
                            "total_requests": aug_summary.get("total"),
                        }
                    )

            file_summary.update(
                {
                    "embeddings_added": added,
                    "embedded": embed_summary.get("embedded"),
                    "skipped": embed_summary.get("skipped"),
                    "total_requests": embed_summary.get("total"),
                    "speakers": dict(speaker_counts),
                }
            )

            logger.info(
                "Training: %s -> %d segment embedding(s) appended (total=%d)",
                audio_name,
                added,
                total_added,
            )
            summary["files"][audio_name] = file_summary
            if progress_bar:
                progress_bar.set_postfix_str(f"added={total_added}")
            continue

        # Fallback legacy single-embedding path
        label, _ = choose_speaker(path, mapping_pre, return_match=True)
        if label in excluded_speakers:
            summary["files"][audio_name] = {
                **file_summary,
                "mode": "excluded",
                "speaker": label,
                "embeddings_added": 0,
            }
            continue
        logger.info("Training: extracting fallback embeddings for %s (label=%s)", audio_name, label)
        try:
            embeddings, _ = extract_speaker_embeddings(
                path,
                hf_token=hf_token,
                min_speakers=1,
                max_speakers=1,
                pyannote_on_cpu=pyannote_on_cpu,
                diarization_model_name=diarization_model or speaker_bank_config.diarization_model,
                force_device=device_guess,
                quiet=quiet,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Speaker bank training: failed to extract fallback embeddings for %s: %s",
                audio_name,
                exc,
            )
            summary["files"][audio_name] = file_summary
            continue

        added = 0
        speaker_counts: Dict[str, int] = defaultdict(int)
        for diar_label, vec in embeddings.items():
            target_speaker, _ = choose_speaker(diar_label, mapping_pre, return_match=True)
            vector = np.asarray(vec, dtype=np.float32)
            speaker_bank.extend(
                [
                    (
                        target_speaker,
                        vector,
                        audio_name,
                        {"diar_label": diar_label, "mode": "train_command"},
                    )
                ]
            )
            speaker_counts[target_speaker] += 1
            added += 1
            total_added += 1

        file_summary.update(
            {
                "embeddings_added": added,
                "embedded": added,
                "skipped": 0,
                "total_requests": len(embeddings),
                "speakers": dict(speaker_counts),
                "speaker": label,
            }
        )
        summary["files"][audio_name] = file_summary
        logger.info(
            "Training: %s -> %d embedding(s) appended (total=%d)",
            audio_name,
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
        classifier_summary = None
        if speaker_bank_config.classifier_train_enabled:
            try:
                classifier_summary = train_segment_classifier_from_multitrack(
                    input_path=input_path,
                    profile_dir=bank_profile_dir,
                    speaker_mapping=mapping_pre or {},
                    hf_token=hf_token,
                    force_device=device_guess,
                    quiet=quiet,
                    batch_size=speaker_bank_config.embed_batch_size,
                    workers=speaker_bank_config.embed_workers,
                    speaker_aliases={"zariel torgan": "David Tanglethorn"},
                    extra_input_paths=speaker_bank_config.classifier_input_paths,
                    transcript_search_roots=[
                        Path(path).expanduser()
                        for path in speaker_bank_config.classifier_transcript_roots
                    ]
                    or None,
                    allowed_speakers=sorted(
                        {
                            str(value).strip()
                            for value in (mapping_pre or {}).values()
                            if str(value).strip()
                            and str(value).strip()
                            not in {
                                str(item).strip()
                                for item in speaker_bank_config.classifier_excluded_speakers
                                if str(item).strip()
                            }
                        }
                    )
                    or None,
                    excluded_speakers=speaker_bank_config.classifier_excluded_speakers or None,
                    model_name=speaker_bank_config.classifier_model,
                    classifier_c=speaker_bank_config.classifier_c,
                    classifier_n_neighbors=speaker_bank_config.classifier_n_neighbors,
                    training_mode=speaker_bank_config.classifier_training_mode,
                    augmentation_profile=speaker_bank_config.classifier_augmentation_profile,
                    augmentation_copies=speaker_bank_config.classifier_augmentation_copies,
                    augmentation_seed=speaker_bank_config.classifier_augmentation_seed,
                    clean_max_records_per_speaker_per_session=(
                        speaker_bank_config.classifier_clean_max_records_per_speaker_per_session
                    ),
                    dataset_cache_dir=(
                        Path(speaker_bank_config.classifier_dataset_cache_dir).expanduser()
                        if speaker_bank_config.classifier_dataset_cache_dir
                        else None
                    ),
                    diarization_model_name=diarization_model
                    or speaker_bank_config.diarization_model,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Speaker bank training: segment classifier training failed: %s", exc)
        if classifier_summary:
            summary.setdefault("artifacts", {})
            summary["artifacts"]["segment_classifier"] = classifier_summary.get("artifacts")
            summary["segment_classifier"] = classifier_summary
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
            logger.warning(
                "Speaker bank training: failed to persist summary %s: %s", summary_path, exc
            )
    else:
        logger.info("Speaker bank training: no embeddings extracted from %s", input_path)

    cleanup_tmp(tmp_root)


def run_transcribe(
    input_path: str,
    backend: str = "auto",
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
    pyannote_on_cpu: bool = False,
    diarization_model: str | None = None,
    quiet: bool = True,
    auto_batch: bool = True,
    cache_mode: str | None = None,
    device: str | None = None,
    speaker_bank_config: SpeakerBankConfig | None = None,
    postprocess_config: PostProcessConfig | None = None,
) -> None:
    # Ensure cuDNN/cuBLAS split libraries are visible to the loader
    _ensure_cuda_libs_on_path()
    _preload_cudnn_libs()
    logger = logging.getLogger("transcriber")
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
    resolved_backend = _resolve_backend_choice(
        backend,
        files=files,
        tmp_root=str(tmp_root) if tmp_root is not None else None,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        single_file_speaker=single_file_speaker,
        device=device,
        speaker_bank_config=speaker_bank_config,
    )
    if resolved_backend != backend:
        logger.info("Resolved backend=%s to backend=%s for this run.", backend, resolved_backend)
    backend = resolved_backend
    if not files:
        raise SystemExit(f"No audio files found in: {input_path}")

    logger.info("Found %d audio file(s) under %s", len(files), input_path)

    per_file_segments: List[Tuple[str, List[dict]]] = []
    diar_by_file: Dict[str, List[dict]] = {}
    exclusive_diar_by_file: Dict[str, List[dict]] = {}

    # Compute sub-dirs for non-HF-hub model caches (faster-whisper/ctranslate2 + align models)
    if hf_cache_root:
        root_path = Path(os.path.expanduser(hf_cache_root)).resolve()
        model_cache_dir = str(root_path / "models")
    else:
        model_cache_dir = None

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
    speaker_bank_modified = False
    speaker_bank_debug_by_file: Dict[str, Dict[str, object]] = {}
    rendered_pca_path: Optional[Path] = None
    segment_classifier = None
    extract_embeddings_for_segments_fn = None
    speaker_bank_embed_device: Optional[str] = None

    def _apply_speaker_bank(
        segments: List[dict],
        embeddings: Dict[str, np.ndarray],
        file_key: str,
    ) -> Dict[str, object]:
        file_path = Path(file_key)
        file_name = file_path.name
        logger.debug(
            "Speaker bank apply start file=%s embeddings=%d labels=%s",
            file_name,
            len(embeddings),
            list(embeddings.keys()),
        )

        if not speaker_bank or not speaker_bank_config:
            logger.debug(
                "No speaker bank configured for %s",
                file_name,
            )
            return {
                "attempted": 0,
                "matched": 0,
                "matches": {},
                "segment_counts": {"matched": 0, "unknown": 0},
            }
        if not speaker_bank_config.use_existing:
            logger.debug(
                "Speaker bank configured to skip existing matches for %s",
                file_name,
            )
            return {
                "attempted": 0,
                "matched": 0,
                "matches": {},
                "segment_counts": {"matched": 0, "unknown": 0},
            }
        if extract_embeddings_for_segments_fn is None:
            logger.warning("Segment embedding extraction is unavailable for %s", file_name)
            return {
                "attempted": 0,
                "matched": 0,
                "matches": {},
                "segment_counts": {"matched": 0, "unknown": 0},
            }

        relabeled_segments, summary, _ = apply_profile_to_segments(
            audio_path=file_key,
            segments=segments,
            label_embeddings=embeddings,
            speaker_bank=speaker_bank,
            speaker_bank_config=speaker_bank_config,
            segment_classifier=segment_classifier,
            extract_embeddings_for_segments_fn=extract_embeddings_for_segments_fn,
            hf_token=hf_token,
            diarization_model_name=diarization_model or speaker_bank_config.diarization_model,
            force_device=speaker_bank_embed_device,
            quiet=quiet,
        )
        segments[:] = relabeled_segments
        if summary["attempted"]:
            logging.getLogger("transcriber").info(
                "Speaker bank matched %d/%d diar speakers for %s",
                summary["matched"],
                summary["attempted"],
                file_name,
            )
        logger.debug(
            "Speaker bank apply complete file=%s matched_segments=%d unknown_segments=%d",
            file_name,
            int(summary["segment_counts"].get("matched") or 0),
            int(summary["segment_counts"].get("unknown") or 0),
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
            prototypes_enabled=speaker_bank_config.prototypes_enabled,
            prototypes_per_cluster=speaker_bank_config.prototypes_per_cluster,
            prototypes_method=speaker_bank_config.prototypes_method,
            scoring_whiten=speaker_bank_config.scoring_whiten,
        )
        segment_classifier = load_segment_classifier(bank_profile_dir)
        bank_info = speaker_bank.summary()
        logger.debug(
            "Speaker bank summary initial profile=%s data=%s",
            speaker_bank_profile_dir,
            bank_info,
        )
        speaker_bank_summary = {
            "profile": str(bank_profile_dir),
            "initial": bank_info,
            "config": {
                "threshold": speaker_bank_config.threshold,
                "radius_factor": speaker_bank_config.radius_factor,
                "use_existing": speaker_bank_config.use_existing,
                "train_from_stems": speaker_bank_config.train_from_stems,
                "margin": speaker_bank_config.scoring_margin,
                "match_per_segment": speaker_bank_config.match_per_segment,
                "match_aggregation": speaker_bank_config.match_aggregation,
                "min_segments_per_label": speaker_bank_config.min_segments_per_label,
                "repair": {
                    "enabled": speaker_bank_config.repair_enabled,
                    "merge_same_raw_gap_seconds": speaker_bank_config.repair_merge_same_raw_gap_seconds,
                    "snap_boundary_seconds": speaker_bank_config.repair_snap_boundary_seconds,
                    "max_overlap_trim_seconds": speaker_bank_config.repair_max_overlap_trim_seconds,
                    "split_on_word_gap_seconds": speaker_bank_config.repair_split_on_word_gap_seconds,
                    "max_seed_overlap_seconds": speaker_bank_config.repair_max_seed_overlap_seconds,
                    "min_segment_duration_seconds": speaker_bank_config.repair_min_segment_duration_seconds,
                },
                "session_graph": {
                    "enabled": speaker_bank_config.session_graph_enabled,
                    "candidate_top_k": speaker_bank_config.session_graph_candidate_top_k,
                    "candidate_floor": speaker_bank_config.session_graph_candidate_floor,
                    "knn": speaker_bank_config.session_graph_knn,
                    "min_similarity": speaker_bank_config.session_graph_min_similarity,
                    "anchor_weight": speaker_bank_config.session_graph_anchor_weight,
                    "temporal_weight": speaker_bank_config.session_graph_temporal_weight,
                    "temporal_tau_seconds": speaker_bank_config.session_graph_temporal_tau_seconds,
                    "temporal_max_gap_seconds": speaker_bank_config.session_graph_temporal_max_gap_seconds,
                    "same_raw_label_weight": speaker_bank_config.session_graph_same_raw_label_weight,
                    "same_top1_weight": speaker_bank_config.session_graph_same_top1_weight,
                    "alpha": speaker_bank_config.session_graph_alpha,
                    "max_iters": speaker_bank_config.session_graph_max_iters,
                    "tolerance": speaker_bank_config.session_graph_tolerance,
                    "strong_seed_score": speaker_bank_config.session_graph_strong_seed_score,
                    "strong_seed_margin": speaker_bank_config.session_graph_strong_seed_margin,
                    "override_min_confidence": speaker_bank_config.session_graph_override_min_confidence,
                    "override_min_margin": speaker_bank_config.session_graph_override_min_margin,
                    "override_min_delta": speaker_bank_config.session_graph_override_min_delta,
                    "pair_overrides": speaker_bank_config.session_graph_pair_overrides,
                },
                "prototypes_enabled": speaker_bank_config.prototypes_enabled,
                "prototypes_per_cluster": speaker_bank_config.prototypes_per_cluster,
                "prototypes_method": speaker_bank_config.prototypes_method,
                "whiten": speaker_bank_config.scoring_whiten,
                "as_norm_enabled": speaker_bank_config.scoring_as_norm_enabled,
                "as_norm_cohort_size": speaker_bank_config.scoring_as_norm_cohort_size,
            },
            "files": {},
        }
        if segment_classifier is not None:
            speaker_bank_summary["segment_classifier"] = segment_classifier.summary()
        logging.getLogger("transcriber").info(
            "Speaker bank profile=%s (entries=%s, speakers=%s)",
            speaker_bank_profile_dir,
            bank_info.get("entries"),
            len(bank_info.get("speakers", [])),
        )

    if backend == "parakeet":
        logger.info("Using parakeet-mlx backend.")
        from .parakeet_backend import (
            load_model as parakeet_load,
            resolve_model_name as parakeet_resolve_model_name,
            transcribe_file as parakeet_transcribe,
        )

        if min_speakers is not None or max_speakers is not None:
            logger.warning(
                "Parakeet backend does not support diarization; ignoring min/max speaker hints."
            )
        if pyannote_on_cpu:
            logger.warning(
                "Parakeet backend does not use direct pyannote controls; ignoring --pyannote-on-cpu."
            )

        parakeet_model_name = parakeet_resolve_model_name(model_name)
        if parakeet_model_name != model_name:
            logger.warning(
                "Parakeet backend remapped model=%s to %s.",
                model_name,
                parakeet_model_name,
            )

        logger.warning(
            "ASR Backend (parakeet): model=%s compute=%s device=%s",
            parakeet_model_name,
            compute_type,
            device or "auto",
        )
        model = parakeet_load(
            parakeet_model_name,
            compute_type=compute_type,
            device=device or "auto",
            download_root=hf_cache_root,
            local_files_only=local_files_only,
        )
        use_tqdm = _tqdm_enabled()
        iter_files = (
            tqdm(
                files,
                desc="Transcribing (parakeet)",
                unit="file",
                bar_format="{l_bar}{bar} | ETA: {remaining} | {n_fmt}/{total_fmt}",
                dynamic_ncols=True,
            )
            if use_tqdm
            else files
        )
        for f in iter_files:
            label = file_labels.get(f, Path(f).stem)
            logger.warning("Start: %s", Path(f).name)
            start_ts = time.time()
            if use_tqdm:
                from tqdm import tqdm as _tqdm

                with _tqdm(total=1, desc=f"{label}", leave=False, unit="phase") as pbar:
                    segs = parakeet_transcribe(f, model, batch_size=batch_size)
                    pbar.update(1)
                    pbar.set_postfix_str("asr")
            else:
                segs = parakeet_transcribe(f, model, batch_size=batch_size)
            dur = int(time.time() - start_ts)
            logger.warning("Finished: %s in %ds (segments=%d)", Path(f).name, dur, len(segs))
            per_file_segments.append((f, segs))
    else:
        logger.info("Using faster-whisper + direct pyannote backend.")
        from .diarization import (
            _detect_device as _diar_detect_device,
            extract_embeddings_for_segments as diar_extract_embeddings_for_segments,
            extract_speaker_embeddings,
        )
        from .transcript_pipeline import transcribe_with_faster_pipeline

        extract_embeddings_for_segments_fn = diar_extract_embeddings_for_segments

        hf_token = (
            os.getenv("HUGGING_FACE_HUB_TOKEN")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
        )
        fw_device = device if device in {"cpu", "cuda"} else _diar_detect_device()
        fw_compute = compute_type
        cpu_supported = {"int8", "int8_float32", "int8_np"}
        if fw_device == "cpu" and fw_compute not in cpu_supported:
            logger.info(
                "Compute type %s is not supported on CPU; falling back to int8 for faster-whisper.",
                fw_compute,
            )
            fw_compute = "int8"
        speaker_bank_embed_device = fw_device
        logger.warning("ASR Device (faster-whisper): %s compute=%s", fw_device, fw_compute)
        eff_bs = batch_size
        if auto_batch:
            eff_bs = _recommend_batch_size(fw_device, model_name, fw_compute, user_hint=batch_size)
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
        elif mapping_covers_all and not diarization_forced:
            enable_diarization = False
            logger.info(
                "Skipping diarization: speaker mapping resolved all input files (%d/%d).",
                len(files),
                len(files),
            )
        use_tqdm = _tqdm_enabled()
        iter_files = (
            tqdm(
                files,
                desc="Transcribing (faster+pyannote)",
                unit="file",
                bar_format="{l_bar}{bar} | ETA: {remaining} | {n_fmt}/{total_fmt}",
                dynamic_ncols=True,
            )
            if use_tqdm
            else files
        )
        for f in iter_files:
            label = file_labels.get(f, Path(f).stem)
            logger.warning("Start: %s", Path(f).name)
            start_ts = time.time()
            if use_tqdm:
                from tqdm import tqdm as _tqdm

                phase_total = 2 if enable_diarization else 1
                with _tqdm(total=phase_total, desc=f"{label}", leave=False, unit="phase") as pbar:
                    result = transcribe_with_faster_pipeline(
                        f,
                        model_name=model_name,
                        compute_type=fw_compute,
                        batch_size=eff_bs,
                        hf_token=hf_token,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        model_cache_dir=model_cache_dir,
                        local_files_only=local_files_only,
                        pyannote_on_cpu=pyannote_on_cpu,
                        diarization_model_name=diarization_model
                        or (speaker_bank_config.diarization_model if speaker_bank_config else None),
                        force_device=fw_device,
                        quiet=quiet,
                        enable_diarization=enable_diarization,
                    )
                    pbar.update(1)
                    pbar.set_postfix_str("asr")
                    if enable_diarization:
                        pbar.update(1)
                        pbar.set_postfix_str("diar")
            else:
                result = transcribe_with_faster_pipeline(
                    f,
                    model_name=model_name,
                    compute_type=fw_compute,
                    batch_size=eff_bs,
                    hf_token=hf_token,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    model_cache_dir=model_cache_dir,
                    local_files_only=local_files_only,
                    pyannote_on_cpu=pyannote_on_cpu,
                    diarization_model_name=diarization_model
                    or (speaker_bank_config.diarization_model if speaker_bank_config else None),
                    force_device=fw_device,
                    quiet=quiet,
                    enable_diarization=enable_diarization,
                )
            segs = result.segments
            dur = int(time.time() - start_ts)
            logger.warning("Finished: %s in %ds (segments=%d)", Path(f).name, dur, len(segs))
            summary = {
                "attempted": len(result.speaker_embeddings),
                "matched": 0,
                "matches": {},
                "segment_counts": {"matched": 0, "unknown": 0},
            }
            if result.speaker_embeddings:
                bank_summary = _apply_speaker_bank(segs, result.speaker_embeddings, f)
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
                training_embeddings = result.speaker_embeddings
                if not training_embeddings:
                    try:
                        training_embeddings, _ = extract_speaker_embeddings(
                            f,
                            hf_token=hf_token,
                            min_speakers=1,
                            max_speakers=1,
                            pyannote_on_cpu=pyannote_on_cpu,
                            diarization_model_name=diarization_model
                            or (
                                speaker_bank_config.diarization_model
                                if speaker_bank_config
                                else None
                            ),
                            force_device=fw_device,
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
                training_entries: List[Tuple[str, np.ndarray, str, Dict[str, object]]] = []
                for diar_label, vec in training_embeddings.items():
                    training_entries.append(
                        (
                            training_label,
                            np.asarray(vec, dtype=np.float32),
                            str(Path(f).name),
                            {
                                "file": str(Path(f).name),
                                "diar_label": diar_label,
                                "mode": "train_from_stems",
                            },
                        )
                    )
                    added += 1
                if added:
                    speaker_bank.extend(training_entries)
                    summary.setdefault("training", {})
                    summary["training"]["embeddings_added"] = added
                    summary["training"]["speaker"] = training_label
                    speaker_bank_modified = True

            per_file_segments.append((f, segs))
            if result.diarization_segments:
                diar_by_file[f] = result.diarization_segments
            if result.exclusive_diarization_segments:
                exclusive_diar_by_file[f] = result.exclusive_diarization_segments

    if speaker_bank and speaker_bank_modified:
        try:
            speaker_bank.save()
            speaker_bank_summary["final"] = speaker_bank.summary()
            if speaker_bank_config and speaker_bank_config.emit_pca and speaker_bank_profile_dir:
                rendered_pca_path = speaker_bank.render_pca(speaker_bank_profile_dir / "pca.png")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist speaker bank updates: %s", exc)

    base = Path(input_path).stem

    def _normalise_per_file_segments(
        payload: List[Tuple[str, List[dict]]],
    ) -> Tuple[List[Tuple[str, List[dict]]], set[str]]:
        normalised: List[Tuple[str, List[dict]]] = []
        kept: set[str] = set()
        for idx, entry in enumerate(payload):
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                fname, segs_obj = entry
            elif isinstance(entry, dict) and len(entry) == 1:
                ((fname, segs_obj),) = entry.items()
            else:
                logger.warning(
                    "Unexpected segment payload at index %d: %r (skipping)",
                    idx,
                    entry,
                )
                continue

            fname_str = str(fname)
            seg_iterable: List[dict] = []
            if isinstance(segs_obj, dict) and "segments" in segs_obj:
                candidate = segs_obj.get("segments") or []
                if isinstance(candidate, (list, tuple)):
                    seg_iterable = list(candidate)
                elif hasattr(candidate, "__iter__") and not isinstance(candidate, (str, bytes)):
                    seg_iterable = list(candidate)
            elif isinstance(segs_obj, (list, tuple)):
                seg_iterable = list(segs_obj)
            elif hasattr(segs_obj, "__iter__") and not isinstance(segs_obj, (str, bytes)):
                seg_iterable = list(segs_obj)
            else:
                logger.warning(
                    "Unsupported segment container for %s (type=%s); skipping file.",
                    fname_str,
                    type(segs_obj).__name__,
                )
                continue

            cleaned_segments: List[dict] = []
            for seg in seg_iterable:
                if isinstance(seg, dict):
                    cleaned_segments.append(seg)
                elif hasattr(seg, "to_dict"):
                    try:
                        cleaned_segments.append(seg.to_dict())
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Failed to convert segment to dict for %s: %s", fname_str, exc)
                else:
                    logger.debug("Dropping non-dict segment for %s: %r", fname_str, seg)
            normalised.append((fname_str, cleaned_segments))
            kept.add(fname_str)
        return normalised, kept

    per_file_segments, _kept_files = _normalise_per_file_segments(per_file_segments)

    def _normalise_diarization(
        payload: Dict[str, List[dict]] | Dict[str, object] | None,
        kept: set[str],
    ) -> Dict[str, List[dict]]:
        if not payload:
            return {}
        normalised: Dict[str, List[dict]] = {}
        for key, value in payload.items():
            fname = str(key)
            if fname not in kept:
                continue
            entries: List[dict] = []
            candidates: List[object] = []
            if isinstance(value, dict) and "segments" in value:
                segment_block = value.get("segments") or []
                if isinstance(segment_block, (list, tuple)):
                    candidates = list(segment_block)
                elif hasattr(segment_block, "__iter__") and not isinstance(
                    segment_block, (str, bytes)
                ):
                    candidates = list(segment_block)
            elif isinstance(value, (list, tuple)):
                candidates = list(value)
            elif hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                candidates = list(value)
            else:
                logger.debug(
                    "Unsupported diarization payload for %s (type=%s); skipping.",
                    fname,
                    type(value).__name__,
                )
                continue
            for item in candidates:
                if isinstance(item, dict):
                    entries.append(item)
                elif hasattr(item, "to_dict"):
                    try:
                        entries.append(item.to_dict())
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Failed to convert diarization entry for %s: %s", fname, exc)
            if entries:
                normalised[fname] = entries
        return normalised

    diar_by_file = _normalise_diarization(diar_by_file, _kept_files)
    exclusive_diar_by_file = _normalise_diarization(exclusive_diar_by_file, _kept_files)

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
                    if s.get("speaker_match_source"):
                        continue
                    _set_segment_speaker_label(s, file_label)
                else:
                    # Single-file: only fill missing and allow cluster -> name override
                    if not s.get("speaker"):
                        _set_segment_speaker_label(s, file_label)
                    if s.get("speaker") in mapping:
                        _set_segment_speaker_label(s, mapping[s["speaker"]])
            per_file_segments[i] = (fname, segs)
    # Prototype single-file behavior: allow forcing a generic label
    if single_file_speaker and len(files) == 1:
        target = files[0]
        for i, (fname, segs) in enumerate(per_file_segments):
            if fname == target:
                for s in segs:
                    if single_file_speaker or not s.get("speaker"):
                        _set_segment_speaker_label(s, single_file_speaker)
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
        exclusive_diar_by_file=exclusive_diar_by_file or None,
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

    # Clean up extracted ZIP contents before any downstream processing.
    cleanup_tmp(tmp_root)

    transcript_output_path = Path(final_out_dir) / f"{base}.txt"
    if postprocess_config:
        run_postprocess_for_transcript(transcript_output_path, postprocess_config)

    logging.getLogger("transcriber").warning(
        "Done. Outputs in %s", str(Path(final_out_dir).resolve())
    )


def _recommend_batch_size(
    device: str, model_name: str, compute_type: str, user_hint: int | None = None
) -> int:
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


def _watch_task_kind(
    input_path: str,
    output_dir: str,
    postprocess_config: PostProcessConfig | None,
) -> str | None:
    transcript_path = _expected_txt_path_for_input(input_path, output_dir)
    if not transcript_path.exists():
        return "transcribe"
    if postprocess_config is None:
        return None
    marker_path = expected_completion_marker_path(transcript_path, postprocess_config)
    if marker_path.exists():
        return None
    return "postprocess"


def _iter_candidate_media(
    root_dir: Path, exclude_globs: Optional[Sequence[str]] = None
) -> list[str]:
    ignore_dirs = {"quarantine", ".cache", "outputs"}
    exclude_patterns = [pattern for pattern in (exclude_globs or []) if pattern]
    files: list[str] = []
    for f in root_dir.rglob("*"):
        if not f.is_file():
            continue
        # Skip ignored directories
        parts = set(p.name for p in f.parents)
        if parts & ignore_dirs:
            continue
        if exclude_patterns:
            rel_path = f.relative_to(root_dir).as_posix()
            rel = PurePosixPath(rel_path)
            if any(rel.match(pattern) for pattern in exclude_patterns):
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
    postprocess_config: Optional[PostProcessConfig],
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
                getattr(args, "watch_input", None) or cfg_watch_input or args.input or cfg_input
            )
            if not input_root:
                logger.error(
                    "Watch: missing INPUT/--watch-input and no 'input'/'watch_input' in config; sleeping %ss",
                    interval,
                )
                time.sleep(interval)
                continue
            root = Path(input_root)
            try:
                root_exists = root.exists()
            except OSError as exc:
                logger.error("Watch: cannot access %s: %s; sleeping %ss", root, exc, interval)
                time.sleep(interval)
                continue
            if not root_exists:
                logger.error("Watch: path does not exist: %s; sleeping %ss", root, interval)
                time.sleep(interval)
                continue
            try:
                root_is_dir = root.is_dir()
                root_is_file = root.is_file()
            except OSError as exc:
                logger.error("Watch: cannot inspect %s: %s; sleeping %ss", root, exc, interval)
                time.sleep(interval)
                continue
            if not root_is_dir:
                if root_is_file:
                    parent = root.parent
                    logger.warning("Watch: INPUT is a file; watching parent directory: %s", parent)
                    root = parent
                else:
                    logger.error(
                        "Watch: %s is neither file nor directory; sleeping %ss", root, interval
                    )
                    time.sleep(interval)
                    continue

            # Use WARNING so it appears even in quiet mode
            logger.warning(
                "Watch: monitoring %s (every %ss, stability %ss)", root, interval, stability
            )

            watch_exclude_globs = []
            if isinstance(cfg, dict):
                watch_exclude_globs = list(cfg.get("watch_exclude_globs") or [])

            candidates = _iter_candidate_media(root, watch_exclude_globs)
            pending: list[tuple[str, str]] = []
            for f in candidates:
                task_kind = _watch_task_kind(f, args.output_dir, postprocess_config)
                if task_kind is None:
                    continue
                if not _file_is_stable(Path(f), stability):
                    continue
                pending.append((task_kind, f))

            if pending:
                logger.warning(
                    "Watch: found %d pending item(s); processing (e.g., %s)",
                    len(pending),
                    Path(pending[0][1]).name,
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
                for task_kind, f in outer:
                    try:
                        if task_kind == "postprocess":
                            transcript_path = _expected_txt_path_for_input(f, args.output_dir)
                            run_postprocess_for_transcript(transcript_path, postprocess_config)
                        else:
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
                                pyannote_on_cpu=args.pyannote_on_cpu,
                                diarization_model=getattr(args, "diarization_model", None),
                                quiet=args.quiet,
                                auto_batch=args.auto_batch,
                                device=(
                                    None
                                    if getattr(args, "device", "auto") == "auto"
                                    else args.device
                                ),
                                speaker_bank_config=speaker_bank_config,
                                postprocess_config=postprocess_config,
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
    ap = argparse.ArgumentParser(
        description="Transcription with ASR, diarization, and speaker naming"
    )
    ap.add_argument("input", nargs="?", help="Path to audio file, directory, or .zip of audios")
    ap.add_argument("--backend", choices=["auto", "faster", "parakeet"], default="auto")
    ap.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=cfg.get("device", "auto"),
        help="Force device selection (default: auto)",
    )
    ap.add_argument("--model", default="large-v3", help="Model name (e.g., large-v3)")
    ap.add_argument(
        "--compute-type", default="float16", help="Compute type (e.g., float16, int8_float16)"
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--auto-batch",
        dest="auto_batch",
        action="store_true",
        default=True,
        help="Choose batch size automatically for your GPU (default)",
    )
    ap.add_argument(
        "--no-auto-batch",
        dest="auto_batch",
        action="store_false",
        help="Disable auto batch-size selection",
    )
    ap.add_argument(
        "--watch",
        action="store_true",
        help="Continuously watch INPUT directory for new audio/ZIP files",
    )
    ap.add_argument(
        "--watch-interval", type=int, default=10, help="Polling interval (seconds) in watch mode"
    )
    ap.add_argument(
        "--watch-stability",
        type=int,
        default=5,
        help="Minimum age (seconds) before a new file is considered stable for processing",
    )
    ap.add_argument(
        "--watch-input",
        help="Override the directory to monitor in watch mode (defaults to INPUT/config input)",
    )
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument(
        "--speaker-mapping",
        dest="speaker_mapping",
        help="YAML/JSON mapping of speaker IDs -> names",
    )
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
    ap.add_argument(
        "--quiet",
        dest="quiet",
        action="store_true",
        default=prelim.quiet,
        help="Reduce log noise and hide warnings (default)",
    )
    ap.add_argument(
        "--no-quiet", dest="quiet", action="store_false", help="Show library warnings and info logs"
    )
    ap.add_argument(
        "--single-file-speaker",
        help="If input resolves to one audio file, label segments with this speaker name (no diarization)",
    )
    ap.add_argument(
        "--pyannote-on-cpu",
        action="store_true",
        help="Force direct pyannote diarization and embedding stages to run on CPU",
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
        "--speaker-bank-margin",
        dest="speaker_bank_margin",
        type=float,
        help="Minimum margin between top-1 and top-2 scores to accept a speaker match.",
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
        "--speaker-bank-prototypes",
        dest="speaker_bank_prototypes",
        action="store_true",
        default=None,
        help="Enable prototype-based speaker matching (in addition to cluster centroids).",
    )
    ap.add_argument(
        "--speaker-bank-no-prototypes",
        dest="speaker_bank_prototypes",
        action="store_false",
        help="Disable prototype-based speaker matching.",
    )
    ap.add_argument(
        "--speaker-bank-prototypes-per-cluster",
        dest="speaker_bank_prototypes_per_cluster",
        type=int,
        help="Number of prototype embeddings to retain per speaker cluster.",
    )
    ap.add_argument(
        "--speaker-bank-prototypes-method",
        dest="speaker_bank_prototypes_method",
        choices=["central", "kmeans"],
        help="Prototype selection strategy (default: central).",
    )
    ap.add_argument(
        "--speaker-bank-train-only",
        metavar="AUDIO_PATH",
        help="Train or update the speaker bank using audio without running transcription",
    )
    ap.add_argument(
        "--segments-json",
        dest="speaker_bank_segments_json",
        help="JSON/JSONL file or directory containing labeled segments for speaker-bank training",
    )
    ap.add_argument(
        "--speaker-bank-train-from-segments",
        dest="speaker_bank_train_from_segments",
        action="store_true",
        default=None,
        help="Enable training from pre-computed segments (JSON/JSONL).",
    )
    ap.add_argument(
        "--speaker-bank-no-train-from-segments",
        dest="speaker_bank_train_from_segments",
        action="store_false",
        help="Disable training from pre-computed segments.",
    )
    ap.add_argument(
        "--speaker-bank-train-segment-source",
        dest="speaker_bank_train_segment_source",
        choices=["auto", "session_jsonl", "diarization_json"],
        help="Preferred artifact when auto-discovering segment metadata (default: auto).",
    )
    ap.add_argument(
        "--speaker-bank-min-segment-dur",
        dest="speaker_bank_min_segment_dur",
        type=float,
        help="Minimum segment/window duration in seconds when training from segments.",
    )
    ap.add_argument(
        "--speaker-bank-max-segment-dur",
        dest="speaker_bank_max_segment_dur",
        type=float,
        help="Maximum segment/window duration in seconds when training from segments.",
    )
    ap.add_argument(
        "--speaker-bank-window-size",
        dest="speaker_bank_window_size",
        type=float,
        help="Sliding window size in seconds for segment chunking (<=0 uses the full segment).",
    )
    ap.add_argument(
        "--speaker-bank-window-stride",
        dest="speaker_bank_window_stride",
        type=float,
        help="Stride in seconds between sliding windows (<=0 defaults to the window size).",
    )
    ap.add_argument(
        "--speaker-bank-max-embeddings",
        dest="speaker_bank_max_embeddings",
        type=int,
        help="Maximum number of embeddings to retain per speaker during training (<=0 = unlimited).",
    )
    ap.add_argument(
        "--speaker-bank-train-chunk-stems",
        dest="speaker_bank_vad_chunk_stems",
        action="store_true",
        default=None,
        help="Chunk single-speaker stems into windows when segment metadata is unavailable.",
    )
    ap.add_argument(
        "--speaker-bank-no-train-chunk-stems",
        dest="speaker_bank_vad_chunk_stems",
        action="store_false",
        help="Do not chunk stems when segment metadata is unavailable.",
    )
    ap.add_argument(
        "--speaker-bank-pre-pad",
        dest="speaker_bank_pre_pad",
        type=float,
        help="Seconds of audio to prepend to each training window when extracting embeddings.",
    )
    ap.add_argument(
        "--speaker-bank-post-pad",
        dest="speaker_bank_post_pad",
        type=float,
        help="Seconds of audio to append to each training window when extracting embeddings.",
    )
    ap.add_argument(
        "--speaker-bank-embed-workers",
        dest="speaker_bank_embed_workers",
        type=int,
        help="Number of CPU workers used to crop training segments.",
    )
    ap.add_argument(
        "--speaker-bank-embed-batch-size",
        dest="speaker_bank_embed_batch_size",
        type=int,
        help="Number of segment windows to embed per GPU forward batch.",
    )
    ap.add_argument(
        "--speaker-bank-match-per-segment",
        dest="speaker_bank_match_per_segment",
        action="store_true",
        default=None,
        help="Match each diarization segment against the speaker bank and aggregate per label.",
    )
    ap.add_argument(
        "--speaker-bank-no-match-per-segment",
        dest="speaker_bank_match_per_segment",
        action="store_false",
        help="Disable per-segment speaker-bank matching.",
    )
    ap.add_argument(
        "--speaker-bank-match-aggregation",
        dest="speaker_bank_match_aggregation",
        choices=["mean", "vote"],
        help="Aggregation strategy for segment-level matches (default: mean).",
    )
    ap.add_argument(
        "--speaker-bank-min-segments-per-label",
        dest="speaker_bank_min_segments_per_label",
        type=int,
        help="Minimum number of matched segments required before assigning a speaker to a diar label.",
    )

    _apply_config_defaults(ap, cfg)
    args = ap.parse_args()

    # Re-apply logging config in case config/CLI changed it
    _setup_logging_and_warnings(args.log_level, args.quiet)

    speaker_bank_config, train_only_path = _resolve_speaker_bank_settings(cfg, args)
    postprocess_config = resolve_postprocess_config(cfg)

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
            model_name=args.model,
            compute_type=args.compute_type,
            batch_size=args.batch_size,
            auto_batch=args.auto_batch,
            pyannote_on_cpu=args.pyannote_on_cpu,
            diarization_model=getattr(args, "diarization_model", None),
            quiet=args.quiet,
            device=None if args.device == "auto" else args.device,
            speaker_bank_config=speaker_bank_config or SpeakerBankConfig(),
            speaker_mapping_path=args.speaker_mapping,
            speaker_bank_root_override=args.speaker_bank_root,
            segments_json=args.speaker_bank_segments_json,
        )
        return

    if args.watch:
        # Keep the service resilient: never exit on uncaught exceptions
        while True:
            try:
                watch_and_transcribe(args, cfg, speaker_bank_config, postprocess_config)
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
        pyannote_on_cpu=args.pyannote_on_cpu,
        diarization_model=getattr(args, "diarization_model", None),
        quiet=args.quiet,
        device=None if args.device == "auto" else args.device,
        speaker_bank_config=speaker_bank_config,
        postprocess_config=postprocess_config,
    )


if __name__ == "__main__":
    main()
