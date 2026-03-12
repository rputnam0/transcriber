from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from .segment_classifier import ClassifierDataset

DEFAULT_CLIPPING_THRESHOLD = 0.999
DEFAULT_SILENCE_THRESHOLD = 1e-3


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {
            str(key): _normalize_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, set):
        return [_normalize_json_value(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    return value


def stable_json_dumps(payload: Mapping[str, object]) -> str:
    return json.dumps(_normalize_json_value(dict(payload)), sort_keys=True, separators=(",", ":"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def artifact_id_for_payload(payload: Mapping[str, object], *, length: int = 16) -> str:
    digest = hashlib.blake2b(
        stable_json_dumps(payload).encode("utf-8"),
        digest_size=max(int(length), 8),
    ).hexdigest()
    return digest[:length]


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_path_identity(path: Path, *, hash_contents: bool = False) -> Dict[str, object]:
    expanded = Path(path).expanduser()
    try:
        resolved = expanded.resolve()
    except OSError:
        resolved = expanded.absolute()

    payload: Dict[str, object] = {"path": str(resolved)}
    if not expanded.exists():
        payload["kind"] = "missing"
        return payload

    stat = expanded.stat()
    if expanded.is_file():
        payload.update(
            {
                "kind": "file",
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
        if hash_contents:
            payload["sha256"] = _hash_file(expanded)
        return payload

    child_files = []
    for child in sorted(expanded.rglob("*")):
        if not child.is_file():
            continue
        child_stat = child.stat()
        child_files.append(
            {
                "path": str(child.relative_to(expanded)),
                "size": int(child_stat.st_size),
                "mtime_ns": int(child_stat.st_mtime_ns),
            }
        )
    payload.update(
        {
            "kind": "directory",
            "files": len(child_files),
            "digest": artifact_id_for_payload({"children": child_files}, length=24),
        }
    )
    return payload


def collect_input_file_identities(
    paths: Sequence[Path],
    *,
    hash_contents: bool = False,
) -> List[Dict[str, object]]:
    return [
        build_path_identity(Path(path), hash_contents=hash_contents)
        for path in sorted(paths, key=lambda item: str(Path(item).expanduser()))
    ]


def current_git_commit(*, cwd: Optional[Path] = None) -> Optional[str]:
    repo_root = cwd or Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def build_artifact_manifest(
    *,
    artifact_type: str,
    diarization_model: str,
    source_sessions: Sequence[str],
    input_file_identities: Sequence[Mapping[str, object]],
    build_params: Mapping[str, object],
    parent_artifacts: Sequence[str],
    created_at: Optional[str] = None,
    git_commit: Optional[str] = None,
    extra: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    base_payload = {
        "artifact_type": str(artifact_type),
        "diarization_model": str(diarization_model),
        "source_sessions": sorted(str(item) for item in source_sessions if str(item).strip()),
        "input_file_identities": [_normalize_json_value(item) for item in input_file_identities],
        "build_params": _normalize_json_value(dict(build_params)),
        "parent_artifacts": sorted(str(item) for item in parent_artifacts if str(item).strip()),
    }
    artifact_id = artifact_id_for_payload(base_payload)
    manifest: Dict[str, object] = {
        "manifest_version": 1,
        "artifact_id": artifact_id,
        "artifact_type": str(artifact_type),
        "created_at": created_at or utc_now_iso(),
        "diarization_model": str(diarization_model),
        "source_sessions": base_payload["source_sessions"],
        "input_file_identities": base_payload["input_file_identities"],
        "build_params": base_payload["build_params"],
        "parent_artifacts": base_payload["parent_artifacts"],
        "git_commit": git_commit or current_git_commit(),
    }
    if extra:
        manifest.update(_normalize_json_value(dict(extra)))
    return manifest


def load_manifest(path: Path) -> Optional[Dict[str, object]]:
    manifest_path = Path(path).expanduser()
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: Mapping[str, object]) -> Path:
    manifest_path = Path(path).expanduser()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_normalize_json_value(dict(manifest)), indent=2), encoding="utf-8")
    return manifest_path


def save_jsonl_records(path: Path, records: Iterable[Mapping[str, object]]) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_normalize_json_value(dict(record))) + "\n")
    return output_path


def load_jsonl_records(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    input_path = Path(path).expanduser()
    if not input_path.exists():
        return records
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _required_paths_exist(paths: Sequence[str]) -> bool:
    for value in paths:
        if not str(value).strip():
            continue
        if not Path(str(value)).expanduser().exists():
            return False
    return True


def build_stage_manifest(
    *,
    stage: str,
    stage_signature: Mapping[str, object],
    outputs: Mapping[str, object],
    required_paths: Sequence[str],
    parent_stages: Sequence[str],
    git_commit: Optional[str] = None,
    created_at: Optional[str] = None,
) -> Dict[str, object]:
    signature_payload = {
        "stage": str(stage),
        "stage_signature": _normalize_json_value(dict(stage_signature)),
        "parent_stages": sorted(str(item) for item in parent_stages if str(item).strip()),
    }
    artifact_id = artifact_id_for_payload(signature_payload)
    return {
        "manifest_version": 1,
        "artifact_id": artifact_id,
        "stage": str(stage),
        "created_at": created_at or utc_now_iso(),
        "stage_signature": signature_payload["stage_signature"],
        "parent_stages": signature_payload["parent_stages"],
        "outputs": _normalize_json_value(dict(outputs)),
        "required_paths": [str(Path(path).expanduser()) for path in required_paths if str(path).strip()],
        "git_commit": git_commit or current_git_commit(),
    }


def stage_manifest_is_reusable(
    manifest_path: Path,
    *,
    stage: str,
    stage_signature: Mapping[str, object],
) -> Tuple[bool, Optional[Dict[str, object]], str]:
    manifest = load_manifest(manifest_path)
    if manifest is None:
        return (False, None, "missing_manifest")
    if str(manifest.get("stage") or "") != str(stage):
        return (False, manifest, "stage_mismatch")
    expected = build_stage_manifest(
        stage=stage,
        stage_signature=stage_signature,
        outputs={},
        required_paths=[],
        parent_stages=manifest.get("parent_stages") or [],
        git_commit=str(manifest.get("git_commit") or "") or None,
        created_at=str(manifest.get("created_at") or "") or None,
    )
    if str(manifest.get("artifact_id") or "") != str(expected["artifact_id"]):
        return (False, manifest, "artifact_id_mismatch")
    required_paths = [str(item) for item in list(manifest.get("required_paths") or [])]
    if not _required_paths_exist(required_paths):
        return (False, manifest, "missing_required_output")
    return (True, manifest, "ok")


def artifact_is_reusable(
    manifest_path: Path,
    *,
    artifact_type: str,
    diarization_model: str,
    artifact_id: Optional[str] = None,
    allow_legacy_reuse: bool = False,
) -> Tuple[bool, str]:
    manifest = load_manifest(manifest_path)
    if manifest is None:
        return (bool(allow_legacy_reuse), "missing_manifest")
    if str(manifest.get("artifact_type") or "") != str(artifact_type):
        return (False, "artifact_type_mismatch")
    if str(manifest.get("diarization_model") or "") != str(diarization_model):
        return (False, "diarization_model_mismatch")
    if artifact_id and str(manifest.get("artifact_id") or "") != str(artifact_id):
        return (False, "artifact_id_mismatch")
    return (True, "ok")


def _query_gpu_memory_bytes() -> Optional[int]:
    commands = [
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        ["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
    ]
    for command in commands:
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            continue
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            continue
        try:
            return int(float(lines[0])) * 1024 * 1024
        except ValueError:
            continue
    return None


@dataclass
class StageMetricsLogger:
    path: Path
    pid: int = field(default_factory=os.getpid)
    _process: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import psutil

            self._process = psutil.Process(self.pid)
            self._process.cpu_percent(interval=None)
        except Exception:
            self._process = None

    def _snapshot(self) -> Dict[str, object]:
        snapshot: Dict[str, object] = {
            "pid": int(self.pid),
            "rss_bytes": None,
            "cpu_percent": None,
            "read_bytes": None,
            "write_bytes": None,
            "gpu_memory_bytes": _query_gpu_memory_bytes(),
        }
        if self._process is None:
            return snapshot
        try:
            snapshot["rss_bytes"] = int(self._process.memory_info().rss)
        except Exception:
            pass
        try:
            snapshot["cpu_percent"] = float(self._process.cpu_percent(interval=None))
        except Exception:
            pass
        try:
            io_counters = self._process.io_counters()
            snapshot["read_bytes"] = int(getattr(io_counters, "read_bytes", 0))
            snapshot["write_bytes"] = int(getattr(io_counters, "write_bytes", 0))
        except Exception:
            pass
        return snapshot

    def log(
        self,
        *,
        stage: str,
        status: str,
        session: Optional[str] = None,
        variant: Optional[str] = None,
        cache_hit: Optional[bool] = None,
        elapsed_seconds: Optional[float] = None,
        extra: Optional[Mapping[str, object]] = None,
    ) -> Dict[str, object]:
        record: Dict[str, object] = {
            "timestamp": utc_now_iso(),
            "monotonic_ns": time.monotonic_ns(),
            "stage": str(stage),
            "status": str(status),
            "session": str(session) if session else None,
            "variant": str(variant) if variant else None,
            "cache_hit": cache_hit,
        }
        if elapsed_seconds is not None:
            record["elapsed_seconds"] = float(elapsed_seconds)
        record.update(self._snapshot())
        if extra:
            record.update(_normalize_json_value(dict(extra)))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_normalize_json_value(record)) + "\n")
        return record

    def bind(self, *, stage: str, variant: Optional[str] = None):
        def _callback(*, status: str, session: Optional[str] = None, cache_hit: Optional[bool] = None, elapsed_seconds: Optional[float] = None, extra: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
            return self.log(
                stage=stage,
                status=status,
                session=session,
                variant=variant,
                cache_hit=cache_hit,
                elapsed_seconds=elapsed_seconds,
                extra=extra,
            )

        return _callback


def build_audio_quality_metrics(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
    clipping_threshold: float = DEFAULT_CLIPPING_THRESHOLD,
) -> Dict[str, object]:
    vector = np.asarray(waveform, dtype=np.float32).reshape(-1)
    decode_ok = bool(vector.size > 0)
    finite_ok = bool(decode_ok and np.isfinite(vector).all())
    if not decode_ok or not finite_ok or sample_rate <= 0:
        return {
            "duration": float(vector.size / sample_rate) if sample_rate > 0 else 0.0,
            "rms": 0.0,
            "peak_abs": 0.0,
            "clipping_fraction": 0.0,
            "silence_fraction": 1.0,
            "decode_ok": decode_ok,
            "finite_ok": finite_ok,
        }

    abs_vector = np.abs(vector)
    return {
        "duration": float(vector.shape[0] / float(sample_rate)),
        "rms": float(np.sqrt(np.mean(np.square(vector, dtype=np.float64)))),
        "peak_abs": float(np.max(abs_vector)) if abs_vector.size else 0.0,
        "clipping_fraction": float(np.mean(abs_vector >= float(clipping_threshold))),
        "silence_fraction": float(np.mean(abs_vector <= float(silence_threshold))),
        "decode_ok": True,
        "finite_ok": True,
    }


def quality_rejection_reason(
    metrics: Mapping[str, object],
    *,
    min_duration: float,
    max_duration: float,
    clipping_fraction_max: float = 0.005,
    silence_fraction_max: float = 0.80,
) -> Optional[str]:
    if not bool(metrics.get("decode_ok", False)):
        return "decode_failed"
    if not bool(metrics.get("finite_ok", False)):
        return "non_finite"

    duration = float(metrics.get("duration") or 0.0)
    if duration < float(min_duration):
        return "too_short"
    if float(max_duration) > 0.0 and duration > float(max_duration):
        return "too_long"
    if float(metrics.get("clipping_fraction") or 0.0) > float(clipping_fraction_max):
        return "clipping"
    if float(metrics.get("silence_fraction") or 0.0) > float(silence_fraction_max):
        return "silence"
    return None


def summarize_quality_records(
    records: Sequence[Mapping[str, object]],
    *,
    clipping_fraction_max: float = 0.005,
    silence_fraction_max: float = 0.80,
) -> Dict[str, object]:
    counts_by_reason: Counter[str] = Counter()
    counts_by_session: Dict[str, Counter[str]] = defaultdict(Counter)
    counts_by_speaker: Dict[str, Counter[str]] = defaultdict(Counter)
    counts_by_source: Dict[str, Counter[str]] = defaultdict(Counter)

    accepted = 0
    for record in records:
        reason = str(record.get("qa_rejection") or "")
        session = str(record.get("session") or "unknown")
        speaker = str(record.get("speaker") or "unknown")
        source = str(record.get("source") or "unknown")
        if reason:
            counts_by_reason[reason] += 1
            counts_by_session[session][reason] += 1
            counts_by_speaker[speaker][reason] += 1
            counts_by_source[source][reason] += 1
        else:
            accepted += 1

    return {
        "records": int(len(records)),
        "accepted": int(accepted),
        "rejections": dict(counts_by_reason),
        "by_session": {key: dict(value) for key, value in sorted(counts_by_session.items())},
        "by_speaker": {key: dict(value) for key, value in sorted(counts_by_speaker.items())},
        "by_source": {key: dict(value) for key, value in sorted(counts_by_source.items())},
        "thresholds": {
            "clipping_fraction_max": float(clipping_fraction_max),
            "silence_fraction_max": float(silence_fraction_max),
        },
    }


def build_source_session_speaker_breakdown(dataset: ClassifierDataset) -> Dict[str, object]:
    by_source: Dict[str, Counter[str]] = defaultdict(Counter)
    by_session: Dict[str, Counter[str]] = defaultdict(Counter)
    by_speaker: Dict[str, Counter[str]] = defaultdict(Counter)

    for source, session, speaker in zip(dataset.sources, dataset.sessions, dataset.labels):
        by_source[str(source)][str(speaker)] += 1
        by_session[str(session)][str(speaker)] += 1
        by_speaker[str(speaker)][str(source)] += 1

    return {
        "by_source": {key: dict(value) for key, value in sorted(by_source.items())},
        "by_session": {key: dict(value) for key, value in sorted(by_session.items())},
        "by_speaker": {key: dict(value) for key, value in sorted(by_speaker.items())},
    }


def build_coverage_report(
    dataset: ClassifierDataset,
    *,
    min_sessions_per_speaker: int = 3,
    min_sample_ratio: float = 0.5,
) -> Dict[str, object]:
    per_domain_counts: Dict[str, Dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    per_domain_sessions: Dict[str, Dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for label, domain, session in zip(dataset.labels, dataset.domains, dataset.sessions):
        speaker = str(label)
        domain_name = str(domain)
        session_name = str(session)
        per_domain_counts[domain_name][speaker]["samples"] += 1
        per_domain_sessions[domain_name][speaker].add(session_name)

    warnings: List[Dict[str, object]] = []
    domains_payload: Dict[str, object] = {}
    for domain_name in sorted(per_domain_counts):
        speaker_counts = {
            speaker: int(counter["samples"])
            for speaker, counter in sorted(per_domain_counts[domain_name].items())
        }
        median_samples = float(np.median(list(speaker_counts.values()))) if speaker_counts else 0.0
        speaker_payload: Dict[str, object] = {}
        for speaker, counts in sorted(per_domain_counts[domain_name].items()):
            sessions = sorted(per_domain_sessions[domain_name][speaker])
            sample_count = int(counts["samples"])
            session_count = len(sessions)
            speaker_payload[speaker] = {
                "samples": sample_count,
                "sessions": sessions,
                "session_count": session_count,
            }
            if session_count < int(min_sessions_per_speaker):
                warnings.append(
                    {
                        "domain": domain_name,
                        "speaker": speaker,
                        "kind": "low_session_count",
                        "session_count": session_count,
                        "required": int(min_sessions_per_speaker),
                    }
                )
            if median_samples > 0.0 and sample_count < (median_samples * float(min_sample_ratio)):
                warnings.append(
                    {
                        "domain": domain_name,
                        "speaker": speaker,
                        "kind": "low_sample_count",
                        "samples": sample_count,
                        "median_samples": median_samples,
                        "ratio": (sample_count / median_samples) if median_samples else 0.0,
                        "required_ratio": float(min_sample_ratio),
                    }
                )
        domains_payload[domain_name] = {
            "median_samples": median_samples,
            "speakers": speaker_payload,
        }

    return {
        "domains": domains_payload,
        "warnings": warnings,
        "thresholds": {
            "min_sessions_per_speaker": int(min_sessions_per_speaker),
            "min_sample_ratio": float(min_sample_ratio),
        },
    }


def save_candidate_pool(
    output_dir: Path,
    *,
    records: Sequence[Mapping[str, object]],
    embeddings: Sequence[np.ndarray],
) -> Dict[str, str]:
    pool_dir = Path(output_dir).expanduser()
    pool_dir.mkdir(parents=True, exist_ok=True)
    records_path = pool_dir / "candidate_pool.jsonl"
    embeddings_path = pool_dir / "candidate_pool_embeddings.npz"
    with records_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            payload = dict(record)
            payload["embedding_index"] = index
            handle.write(json.dumps(_normalize_json_value(payload)) + "\n")
    matrix = (
        np.vstack([np.asarray(item, dtype=np.float32) for item in embeddings]).astype(np.float32)
        if embeddings
        else np.zeros((0, 0), dtype=np.float32)
    )
    np.savez_compressed(embeddings_path, embeddings=matrix)
    return {
        "candidate_pool_records": str(records_path),
        "candidate_pool_embeddings": str(embeddings_path),
    }


def load_candidate_pool(output_dir: Path) -> Tuple[List[Dict[str, object]], np.ndarray]:
    pool_dir = Path(output_dir).expanduser()
    records_path = pool_dir / "candidate_pool.jsonl"
    embeddings_path = pool_dir / "candidate_pool_embeddings.npz"
    records: List[Dict[str, object]] = []
    if records_path.exists():
        with records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    if embeddings_path.exists():
        payload = np.load(embeddings_path, allow_pickle=False)
        matrix = np.asarray(payload["embeddings"], dtype=np.float32)
    else:
        matrix = np.zeros((0, 0), dtype=np.float32)
    return records, matrix


def append_dataset(
    base_dataset: "ClassifierDataset",
    supplement_dataset: Optional["ClassifierDataset"],
) -> "ClassifierDataset":
    if supplement_dataset is None or supplement_dataset.samples <= 0:
        return base_dataset
    from .segment_classifier import merge_classifier_datasets

    return merge_classifier_datasets([base_dataset, supplement_dataset])
