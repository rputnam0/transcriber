from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover - dependency missing handled elsewhere
    DBSCAN = None  # type: ignore[assignment]
    PCA = None  # type: ignore[assignment]

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]


Array = NDArray[np.float32]


@dataclass
class SpeakerBankConfig:
    enabled: bool = True
    path: str = "default"
    threshold: float = 0.65
    radius_factor: float = 2.5
    use_existing: bool = True
    train_from_stems: bool = False
    emit_pca: bool = True
    cluster_method: str = "dbscan"
    cluster_eps: float = 0.28
    cluster_min_samples: int = 5


@dataclass
class SampleMeta:
    """Metadata stored per embedding entry."""

    speaker: str
    source: Optional[str] = None
    created_ts: float = field(default_factory=lambda: time.time())
    extra: Dict[str, object] = field(default_factory=dict)
    cluster_id: Optional[str] = None

    def to_json(self) -> Dict[str, object]:
        payload = {
            "speaker": self.speaker,
            "source": self.source,
            "created_ts": self.created_ts,
            "extra": self.extra,
            "cluster_id": self.cluster_id,
        }
        return payload

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "SampleMeta":
        return cls(
            speaker=str(payload.get("speaker") or "unknown"),
            source=payload.get("source"),  # type: ignore[arg-type]
            created_ts=float(payload.get("created_ts") or 0.0),
            extra=dict(payload.get("extra") or {}),
            cluster_id=payload.get("cluster_id"),  # type: ignore[arg-type]
        )


@dataclass
class ClusterInfo:
    """Represents one cluster (persona) for a speaker."""

    speaker: str
    cluster_id: str
    centroid: Array
    member_indices: List[int]
    variance: float

    def serialize(self) -> Dict[str, object]:
        return {
            "speaker": self.speaker,
            "cluster_id": self.cluster_id,
            "members": self.member_indices,
            "variance": self.variance,
        }


class SpeakerBank:
    """Persist, cluster, and match speaker embeddings."""

    _MANIFEST_NAME = "bank.json"
    _EMBEDDINGS_NAME = "embeddings.npy"

    def __init__(
        self,
        root: Path,
        profile: str = "default",
        *,
        cluster_method: str = "dbscan",
        dbscan_eps: float = 0.28,
        dbscan_min_samples: int = 5,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.profile = profile
        self.profile_dir = self.root / profile
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.profile_dir / self._MANIFEST_NAME
        self.embeddings_path = self.profile_dir / self._EMBEDDINGS_NAME
        self.cluster_method = cluster_method
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self._embeddings: List[Array] = []
        self._metas: List[SampleMeta] = []
        self._clusters: Dict[str, List[ClusterInfo]] = {}
        self._dirty = False
        self._load()

    @property
    def is_empty(self) -> bool:
        return len(self._embeddings) == 0

    @property
    def speakers(self) -> List[str]:
        return sorted({meta.speaker for meta in self._metas})

    def _load(self) -> None:
        if self.manifest_path.exists() and self.embeddings_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                entries = payload.get("entries") or []
                clusters = payload.get("clusters") or {}
                self._metas = [SampleMeta.from_json(item) for item in entries]
                emb_matrix = np.load(self.embeddings_path, allow_pickle=False)
                self._embeddings = [np.array(vec, dtype=np.float32) for vec in emb_matrix]
                self._clusters = self._deserialize_clusters(clusters)
            except Exception as exc:
                logger.warning("Failed to load speaker bank %s: %s", self.profile_dir, exc)
                self._embeddings = []
                self._metas = []
                self._clusters = {}

    def _deserialize_clusters(self, payload: Dict[str, object]) -> Dict[str, List[ClusterInfo]]:
        result: Dict[str, List[ClusterInfo]] = {}
        for speaker, clusters in (payload or {}).items():
            items: List[ClusterInfo] = []
            for info in clusters or []:
                try:
                    indices = [int(i) for i in info.get("members", [])]  # type: ignore[union-attr]
                    centroid = self._compute_centroid(indices)
                    items.append(
                        ClusterInfo(
                            speaker=speaker,
                            cluster_id=str(info.get("cluster_id") or ""),
                            centroid=centroid,
                            member_indices=indices,
                            variance=float(info.get("variance") or 0.0),
                        )
                    )
                except Exception:
                    continue
            if items:
                result[speaker] = items
        return result

    def _compute_centroid(self, indices: Iterable[int]) -> Array:
        vectors = [self._embeddings[idx] for idx in indices if 0 <= idx < len(self._embeddings)]
        if not vectors:
            return np.zeros((1,), dtype=np.float32)
        stacked = np.vstack(vectors)
        centroid = stacked.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid.astype(np.float32)

    def add_embedding(
        self,
        speaker: str,
        embedding: Array,
        *,
        source: Optional[str] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError("Embedding must be 1D vector")
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Zero embedding cannot be added to speaker bank")
        vec = vec / norm
        self._embeddings.append(vec)
        meta = SampleMeta(
            speaker=speaker,
            source=source,
            extra=extra or {},
        )
        self._metas.append(meta)
        self._dirty = True

    def extend(
        self,
        entries: Iterable[Tuple[str, Array, Optional[str], Optional[Dict[str, object]]]],
    ) -> None:
        for speaker, embedding, source, extra in entries:
            self.add_embedding(speaker, embedding, source=source, extra=extra)

    def save(self) -> None:
        if not self._dirty:
            return
        matrix = np.stack(self._embeddings, axis=0) if self._embeddings else np.zeros((0, 1))
        np.save(self.embeddings_path, matrix.astype(np.float32))
        self._clusters = self._build_clusters()
        payload = {
            "version": 1,
            "profile": self.profile,
            "updated_ts": time.time(),
            "entries": [meta.to_json() for meta in self._metas],
            "clusters": self._serialize_clusters(self._clusters),
        }
        tmp_path = self.manifest_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(self.manifest_path)
        self._dirty = False

    def _build_clusters(self) -> Dict[str, List[ClusterInfo]]:
        if not self._embeddings:
            return {}
        clusters: Dict[str, List[ClusterInfo]] = {}
        speaker_to_indices: Dict[str, List[int]] = {}
        for idx, meta in enumerate(self._metas):
            speaker_to_indices.setdefault(meta.speaker, []).append(idx)

        for speaker, indices in speaker_to_indices.items():
            vectors = np.vstack([self._embeddings[i] for i in indices])
            method = (self.cluster_method or "dbscan").lower()
            labels = np.zeros(len(indices), dtype=int)
            if method == "dbscan" and DBSCAN is not None and len(indices) >= 2:
                try:
                    clustering = DBSCAN(
                        eps=self.dbscan_eps,
                        min_samples=self.dbscan_min_samples,
                        metric="cosine",
                    ).fit(vectors)
                    labels = clustering.labels_
                except Exception as exc:
                    logger.debug("DBSCAN failed for %s: %s", speaker, exc)
                    labels = np.zeros(len(indices), dtype=int)
            persona_clusters: Dict[int, List[int]] = {}
            for offset, label in enumerate(labels):
                persona_clusters.setdefault(label, []).append(indices[offset])
            # Ensure -1 noise entries are preserved as unique clusters
            result_clusters: List[ClusterInfo] = []
            for label, member_indices in persona_clusters.items():
                if label == -1:
                    # treat each as its own cluster
                    for idx in member_indices:
                        centroid = self._embeddings[idx]
                        result_clusters.append(
                            ClusterInfo(
                                speaker=speaker,
                                cluster_id=f"{speaker}_solo_{idx}",
                                centroid=centroid,
                                member_indices=[idx],
                                variance=0.0,
                            )
                        )
                else:
                    centroid = self._compute_centroid(member_indices)
                    variance = self._cluster_variance(centroid, member_indices)
                    cluster_id = f"{speaker}_c{label}"
                    result_clusters.append(
                        ClusterInfo(
                            speaker=speaker,
                            cluster_id=cluster_id,
                            centroid=centroid,
                            member_indices=member_indices,
                            variance=variance,
                        )
                    )
                    for idx in member_indices:
                        self._metas[idx].cluster_id = cluster_id
            if result_clusters:
                clusters[speaker] = result_clusters
        return clusters

    def _cluster_variance(self, centroid: Array, indices: Iterable[int]) -> float:
        distances: List[float] = []
        for idx in indices:
            vec = self._embeddings[idx]
            distances.append(float(np.linalg.norm(vec - centroid)))
        if not distances:
            return 0.0
        return float(np.mean(np.square(distances)))

    def _serialize_clusters(self, clusters: Dict[str, List[ClusterInfo]]) -> Dict[str, List[Dict[str, object]]]:
        return {
            speaker: [cluster.serialize() for cluster in cluster_list]
            for speaker, cluster_list in clusters.items()
        }

    def match(
        self,
        embedding: Array,
        *,
        threshold: float,
        radius_factor: float = 2.5,
    ) -> Optional[Dict[str, object]]:
        if not self._clusters:
            self._clusters = self._build_clusters()
        if not self._clusters:
            return None
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError("Embedding to match must be 1D vector")
        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        vec = vec / norm
        best: Optional[Tuple[ClusterInfo, float, float]] = None
        for cluster_list in self._clusters.values():
            for cluster in cluster_list:
                centroid = cluster.centroid
                if centroid.size != vec.size:
                    continue
                score = float(np.dot(vec, centroid))
                if score < threshold:
                    continue
                distance = float(np.linalg.norm(vec - centroid))
                limit = math.sqrt(cluster.variance) * radius_factor if cluster.variance > 0 else radius_factor * 0.1
                if cluster.variance == 0.0:
                    limit = radius_factor * 0.05
                if distance > max(limit, 1e-4):
                    continue
                if not best or score > best[1]:
                    best = (cluster, score, distance)
        if not best:
            return None
        cluster, score, distance = best
        return {
            "speaker": cluster.speaker,
            "cluster_id": cluster.cluster_id,
            "score": score,
            "distance": distance,
        }

    def summary(self) -> Dict[str, object]:
        return {
            "profile": self.profile,
            "speakers": self.speakers,
            "entries": len(self._embeddings),
            "clusters": {
                speaker: [
                    {
                        "cluster_id": cluster.cluster_id,
                        "members": len(cluster.member_indices),
                        "variance": cluster.variance,
                    }
                    for cluster in clusters
                ]
                for speaker, clusters in self._clusters.items()
            },
        }

    def render_pca(self, output_path: Path, *, annotate: bool = False) -> Optional[Path]:
        if not self._embeddings or PCA is None or plt is None:
            return None
        try:
            matrix = np.vstack(self._embeddings)
            if matrix.shape[0] < 2:
                return None
            pca = PCA(n_components=2)
            coords = pca.fit_transform(matrix)
            fig, ax = plt.subplots(figsize=(6, 6))
            colors: Dict[str, Tuple[float, float, float]] = {}
            rng = np.random.default_rng(seed=42)
            for idx, meta in enumerate(self._metas):
                if meta.speaker not in colors:
                    colors[meta.speaker] = tuple(rng.uniform(0.25, 0.95, size=3))
                color = colors[meta.speaker]
                ax.scatter(coords[idx, 0], coords[idx, 1], c=[color], label=meta.speaker, alpha=0.7, s=24)
                if annotate:
                    ax.annotate(meta.cluster_id or idx, (coords[idx, 0], coords[idx, 1]), fontsize=6, alpha=0.6)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                unique = dict(zip(labels, handles))
                ax.legend(unique.values(), unique.keys(), fontsize=8, loc="upper right")
            ax.set_title(f"Speaker PCA – {self.profile}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(output_path)
            plt.close(fig)
            return output_path
        except Exception as exc:
            logger.debug("Failed to render PCA plot: %s", exc)
            return None
