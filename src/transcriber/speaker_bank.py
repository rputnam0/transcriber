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
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover - dependency missing handled elsewhere
    DBSCAN = None  # type: ignore[assignment]
    KMeans = None  # type: ignore[assignment]
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
    threshold: float = 0.35
    radius_factor: float = 2.5
    use_existing: bool = True
    train_from_stems: bool = False
    train_from_segments: bool = True
    train_segment_source: str = "auto"
    min_segment_dur: float = 6.0
    max_segment_dur: float = 30.0
    window_size: float = 15.0
    window_stride: float = 7.5
    max_embeddings_per_speaker: int = 300
    vad_chunk_stems: bool = False
    segments_path: Optional[str] = None
    pre_pad: float = 0.15
    post_pad: float = 0.15
    embed_workers: int = 4
    embed_batch_size: int = 16
    scoring_margin: float = 0.0
    classifier_min_confidence: float = 0.0
    classifier_min_margin: float = 0.03
    classifier_fusion_mode: str = "fallback"
    classifier_fusion_weight: float = 0.70
    classifier_bank_weight: float = 0.30
    classifier_model: str = "knn"
    classifier_c: float = 1.0
    classifier_n_neighbors: int = 7
    classifier_training_mode: str = "mixed"
    classifier_train_enabled: bool = True
    classifier_excluded_speakers: List[str] = field(default_factory=list)
    classifier_augmentation_profile: str = "none"
    classifier_augmentation_copies: int = 0
    classifier_augmentation_seed: int = 13
    classifier_clean_max_records_per_speaker_per_session: int = 80
    classifier_dataset_cache_dir: Optional[str] = None
    classifier_input_paths: List[str] = field(default_factory=list)
    classifier_transcript_roots: List[str] = field(default_factory=list)
    diarization_model: Optional[str] = None
    scoring_as_norm_enabled: bool = False
    scoring_as_norm_cohort_size: int = 50
    scoring_whiten: bool = True
    prototypes_enabled: bool = True
    prototypes_per_cluster: int = 3
    prototypes_method: str = "central"
    match_per_segment: bool = True
    match_aggregation: str = "mean"
    min_segments_per_label: int = 1
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
    prototype_indices: List[int] = field(default_factory=list)

    def serialize(self) -> Dict[str, object]:
        return {
            "speaker": self.speaker,
            "cluster_id": self.cluster_id,
            "members": self.member_indices,
            "variance": self.variance,
            "prototypes": self.prototype_indices,
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
        prototypes_enabled: bool = True,
        prototypes_per_cluster: int = 3,
        prototypes_method: str = "central",
        scoring_whiten: bool = False,
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
        self.prototypes_enabled = prototypes_enabled
        self.prototypes_per_cluster = max(int(prototypes_per_cluster or 0), 0)
        self.prototypes_method = (prototypes_method or "central").lower()
        self.scoring_whiten = bool(scoring_whiten)
        self._embeddings: List[Array] = []
        self._metas: List[SampleMeta] = []
        self._clusters: Dict[str, List[ClusterInfo]] = {}
        self._score_embeddings: Optional[List[Array]] = None
        self._whiten_mean: Optional[NDArray[np.float64]] = None
        self._whiten_matrix: Optional[NDArray[np.float64]] = None
        self._cohort_cache: Dict[str, NDArray[np.float32]] = {}
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
                speaker_count = len({meta.speaker for meta in self._metas})
                cluster_total = sum(len(items) for items in self._clusters.values())
                logger.debug(
                    "Loaded speaker bank profile=%s entries=%d speakers=%d clusters=%d",
                    self.profile_dir,
                    len(self._embeddings),
                    speaker_count,
                    cluster_total,
                )
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
                    variance = self._cluster_variance(centroid, indices)
                    prototypes_raw = info.get("prototypes") or []
                    proto_indices = [int(i) for i in prototypes_raw if isinstance(i, (int, float))]
                    items.append(
                        ClusterInfo(
                            speaker=speaker,
                            cluster_id=str(info.get("cluster_id") or ""),
                            centroid=centroid,
                            member_indices=indices,
                            variance=variance,
                            prototype_indices=proto_indices,
                        )
                    )
                except Exception:
                    continue
            if items:
                result[speaker] = items
        return result

    def _compute_centroid(self, indices: Iterable[int]) -> Array:
        active_embeddings = self._get_active_embeddings()
        vectors = [active_embeddings[idx] for idx in indices if 0 <= idx < len(active_embeddings)]
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
        self._clusters = {}
        self._invalidate_scoring_cache()
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

    def _invalidate_scoring_cache(self) -> None:
        self._score_embeddings = None
        self._whiten_mean = None
        self._whiten_matrix = None
        self._cohort_cache = {}

    def _get_active_embeddings(self) -> List[Array]:
        if not self.scoring_whiten or not self._embeddings:
            return self._embeddings
        if self._score_embeddings is not None:
            return self._score_embeddings

        matrix = np.vstack(self._embeddings).astype(np.float64, copy=False)
        if matrix.shape[0] < 2:
            self._score_embeddings = [vec.copy() for vec in self._embeddings]
            return self._score_embeddings

        mean = matrix.mean(axis=0)
        centered = matrix - mean
        cov = np.cov(centered, rowvar=False, bias=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=np.float64)

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError as exc:
            logger.debug(
                "Whitening eigendecomposition failed; falling back to raw embeddings: %s", exc
            )
            self._score_embeddings = [vec.copy() for vec in self._embeddings]
            return self._score_embeddings

        if eigenvalues.size == 0:
            self._score_embeddings = [vec.copy() for vec in self._embeddings]
            return self._score_embeddings

        eps = max(float(np.max(eigenvalues)) * 1e-6, 1e-8)
        inv_sqrt = np.where(eigenvalues > eps, 1.0 / np.sqrt(eigenvalues), 0.0)
        whitening = eigenvectors @ np.diag(inv_sqrt) @ eigenvectors.T
        transformed = centered @ whitening
        norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        transformed = transformed / safe_norms

        self._whiten_mean = mean.astype(np.float64, copy=False)
        self._whiten_matrix = whitening.astype(np.float64, copy=False)
        self._score_embeddings = [np.asarray(row, dtype=np.float32) for row in transformed]
        return self._score_embeddings

    def _transform_for_scoring(self, vec: Array) -> Array:
        normalized = np.asarray(vec, dtype=np.float32)
        if normalized.ndim != 1:
            raise ValueError("Embedding to transform must be a 1D vector")
        if not self.scoring_whiten:
            return normalized

        active_embeddings = self._get_active_embeddings()
        if (
            active_embeddings is self._embeddings
            or self._whiten_mean is None
            or self._whiten_matrix is None
        ):
            return normalized

        transformed = (normalized.astype(np.float64) - self._whiten_mean) @ self._whiten_matrix
        norm = np.linalg.norm(transformed)
        if norm == 0.0:
            return normalized
        return np.asarray(transformed / norm, dtype=np.float32)

    def _get_cohort_matrix(self, speaker: str) -> NDArray[np.float32]:
        cached = self._cohort_cache.get(speaker)
        if cached is not None:
            return cached

        active_embeddings = self._get_active_embeddings()
        cohort_rows = [
            active_embeddings[idx]
            for idx, meta in enumerate(self._metas)
            if meta.speaker != speaker
        ]
        if cohort_rows:
            cohort = np.vstack(cohort_rows).astype(np.float32, copy=False)
        elif active_embeddings:
            cohort = np.zeros((0, active_embeddings[0].shape[0]), dtype=np.float32)
        else:
            cohort = np.zeros((0, 1), dtype=np.float32)
        self._cohort_cache[speaker] = cohort
        return cohort

    def _top_k_stats(self, scores: NDArray[np.float32], cohort_size: int) -> Tuple[float, float]:
        if scores.size == 0 or cohort_size <= 0:
            return 0.0, 1.0
        limit = min(int(cohort_size), int(scores.size))
        if limit <= 0:
            return 0.0, 1.0
        if limit >= scores.size:
            top_scores = scores
        else:
            partition_index = scores.size - limit
            top_scores = np.partition(scores, partition_index)[partition_index:]
        mean = float(np.mean(top_scores))
        std = float(np.std(top_scores))
        return mean, max(std, 1e-6)

    def _apply_adaptive_s_norm(
        self,
        raw_score: float,
        query_vec: Array,
        candidate_vec: Array,
        *,
        speaker: str,
        cohort_size: int,
    ) -> float:
        cohort = self._get_cohort_matrix(speaker)
        if cohort.size == 0:
            return raw_score

        query_scores = np.asarray(cohort @ query_vec, dtype=np.float32)
        candidate_scores = np.asarray(cohort @ candidate_vec, dtype=np.float32)
        query_mean, query_std = self._top_k_stats(query_scores, cohort_size)
        candidate_mean, candidate_std = self._top_k_stats(candidate_scores, cohort_size)
        return 0.5 * (
            ((raw_score - candidate_mean) / candidate_std) + ((raw_score - query_mean) / query_std)
        )

    def _select_prototypes(self, member_indices: Iterable[int], centroid: Array) -> List[int]:
        if not self.prototypes_enabled or self.prototypes_per_cluster <= 0:
            return []
        indices = [idx for idx in member_indices if 0 <= idx < len(self._embeddings)]
        if not indices:
            return []
        limit = min(self.prototypes_per_cluster, len(indices))
        if limit <= 0:
            return []

        active_embeddings = self._get_active_embeddings()
        vectors = np.vstack([active_embeddings[i] for i in indices])

        if (
            self.prototypes_method == "kmeans"
            and KMeans is not None
            and limit > 1
            and len(indices) >= limit
        ):
            try:
                km = KMeans(n_clusters=limit, n_init=10, random_state=42)
                km.fit(vectors)
                centers = km.cluster_centers_
                chosen: List[int] = []
                for center in centers:
                    distances = np.linalg.norm(vectors - center, axis=1)
                    order = np.argsort(distances)
                    for pos in order:
                        candidate = indices[pos]
                        if candidate not in chosen:
                            chosen.append(candidate)
                            break
                if len(chosen) >= limit:
                    return chosen[:limit]
                indices_remaining = [idx for idx in indices if idx not in chosen]
                vectors_remaining = (
                    np.vstack([active_embeddings[i] for i in indices_remaining])
                    if indices_remaining
                    else np.empty((0, vectors.shape[1]))
                )
                if indices_remaining:
                    distances = np.linalg.norm(vectors_remaining - centroid, axis=1)
                    order = np.argsort(distances)
                    for pos in order:
                        candidate = indices_remaining[pos]
                        if candidate not in chosen:
                            chosen.append(candidate)
                        if len(chosen) >= limit:
                            break
                return chosen[:limit]
            except Exception as exc:  # pragma: no cover - fallback on error
                logger.debug("Prototype kmeans failed: %s", exc)

        distances = np.linalg.norm(vectors - centroid, axis=1)
        order = np.argsort(distances)
        selected = [indices[pos] for pos in order[:limit]]
        return selected

    def _build_clusters(self) -> Dict[str, List[ClusterInfo]]:
        if not self._embeddings:
            return {}
        clusters: Dict[str, List[ClusterInfo]] = {}
        speaker_to_indices: Dict[str, List[int]] = {}
        for idx, meta in enumerate(self._metas):
            speaker_to_indices.setdefault(meta.speaker, []).append(idx)

        active_embeddings = self._get_active_embeddings()
        for speaker, indices in speaker_to_indices.items():
            vectors = np.vstack([active_embeddings[i] for i in indices])
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
                        centroid = active_embeddings[idx]
                        cluster = ClusterInfo(
                            speaker=speaker,
                            cluster_id=f"{speaker}_solo_{idx}",
                            centroid=centroid,
                            member_indices=[idx],
                            variance=0.0,
                            prototype_indices=[idx] if self.prototypes_enabled else [],
                        )
                        result_clusters.append(cluster)
                else:
                    centroid = self._compute_centroid(member_indices)
                    variance = self._cluster_variance(centroid, member_indices)
                    cluster_id = f"{speaker}_c{label}"
                    prototypes = self._select_prototypes(member_indices, centroid)
                    result_clusters.append(
                        ClusterInfo(
                            speaker=speaker,
                            cluster_id=cluster_id,
                            centroid=centroid,
                            member_indices=member_indices,
                            variance=variance,
                            prototype_indices=prototypes,
                        )
                    )
                    for idx in member_indices:
                        self._metas[idx].cluster_id = cluster_id
            if result_clusters:
                clusters[speaker] = result_clusters
                logger.debug(
                    "Clustered speaker=%s clusters=%d members=%d method=%s",
                    speaker,
                    len(result_clusters),
                    sum(len(info.member_indices) for info in result_clusters),
                    method,
                )
        return clusters

    def _cluster_variance(self, centroid: Array, indices: Iterable[int]) -> float:
        distances: List[float] = []
        active_embeddings = self._get_active_embeddings()
        for idx in indices:
            vec = active_embeddings[idx]
            distances.append(float(np.linalg.norm(vec - centroid)))
        if not distances:
            return 0.0
        return float(np.mean(np.square(distances)))

    def _serialize_clusters(
        self, clusters: Dict[str, List[ClusterInfo]]
    ) -> Dict[str, List[Dict[str, object]]]:
        return {
            speaker: [cluster.serialize() for cluster in cluster_list]
            for speaker, cluster_list in clusters.items()
        }

    def _score_candidates_normalized(
        self,
        vec: Array,
        radius_factor: float,
        *,
        as_norm_enabled: bool,
        as_norm_cohort_size: int,
    ) -> List[Dict[str, object]]:
        if not self._clusters:
            self._clusters = self._build_clusters()
        if not self._clusters:
            return []
        results: List[Dict[str, object]] = []
        for speaker, cluster_list in self._clusters.items():
            best_score: Optional[float] = None
            best_raw_score: Optional[float] = None
            best_cluster: Optional[ClusterInfo] = None
            best_source = "centroid"
            best_distance = 0.0

            for cluster in cluster_list:
                if cluster.centroid.size != vec.size:
                    continue
                candidate_vectors: List[Tuple[str, Array]] = [("centroid", cluster.centroid)]
                if self.prototypes_enabled and cluster.prototype_indices:
                    for idx in cluster.prototype_indices:
                        if 0 <= idx < len(self._embeddings):
                            candidate_vectors.append(
                                ("prototype", self._get_active_embeddings()[idx])
                            )

                cluster_best_score: Optional[float] = None
                cluster_best_raw_score: Optional[float] = None
                cluster_best_source = "centroid"
                cluster_best_distance = 0.0
                for source_name, candidate_vec in candidate_vectors:
                    if candidate_vec.size != vec.size:
                        continue
                    candidate_distance = float(np.linalg.norm(vec - candidate_vec))
                    if cluster.variance > 0:
                        distance_limit = math.sqrt(cluster.variance) * radius_factor
                        if candidate_distance > max(distance_limit, 1e-4):
                            continue
                    raw_score = float(np.dot(vec, candidate_vec))
                    score = raw_score
                    if as_norm_enabled:
                        score = self._apply_adaptive_s_norm(
                            raw_score,
                            vec,
                            candidate_vec,
                            speaker=speaker,
                            cohort_size=as_norm_cohort_size,
                        )
                    if cluster_best_score is None or score > cluster_best_score:
                        cluster_best_score = score
                        cluster_best_raw_score = raw_score
                        cluster_best_source = source_name
                        cluster_best_distance = candidate_distance

                if cluster_best_score is None:
                    continue
                if best_score is None or cluster_best_score > best_score:
                    best_score = cluster_best_score
                    best_raw_score = cluster_best_raw_score
                    best_cluster = cluster
                    best_source = cluster_best_source
                    best_distance = cluster_best_distance

            if best_cluster is not None and best_score is not None:
                results.append(
                    {
                        "speaker": speaker,
                        "cluster_id": best_cluster.cluster_id,
                        "score": best_score,
                        "raw_score": best_raw_score if best_raw_score is not None else best_score,
                        "distance": best_distance,
                        "source": best_source,
                        "variance": best_cluster.variance,
                        "score_mode": "as_norm" if as_norm_enabled else "cosine",
                    }
                )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    def score_candidates(
        self,
        embedding: Array,
        *,
        radius_factor: float = 2.5,
        as_norm_enabled: bool = False,
        as_norm_cohort_size: int = 50,
    ) -> List[Dict[str, object]]:
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError("Embedding to match must be 1D vector")
        norm = np.linalg.norm(vec)
        if norm == 0:
            return []
        normalized = vec / norm
        candidates = self._score_candidates_normalized(
            self._transform_for_scoring(normalized),
            radius_factor,
            as_norm_enabled=bool(as_norm_enabled),
            as_norm_cohort_size=max(int(as_norm_cohort_size or 0), 0),
        )
        return candidates

    def match(
        self,
        embedding: Array,
        *,
        threshold: float,
        radius_factor: float = 2.5,
        margin: float = 0.0,
        as_norm_enabled: bool = False,
        as_norm_cohort_size: int = 50,
    ) -> Optional[Dict[str, object]]:
        candidates = self.score_candidates(
            embedding,
            radius_factor=radius_factor,
            as_norm_enabled=as_norm_enabled,
            as_norm_cohort_size=as_norm_cohort_size,
        )
        if not candidates:
            logger.debug(
                "Speaker bank profile=%s has no clusters available for matching.",
                self.profile,
            )
            return None
        top1 = candidates[0]
        top2_score = candidates[1]["score"] if len(candidates) > 1 else None
        margin_value = top1["score"] - (top2_score if top2_score is not None else 0.0)
        logger.debug(
            "Matching embedding -> top1 speaker=%s score=%.4f margin=%.4f threshold=%.4f",
            top1["speaker"],
            top1["score"],
            margin_value,
            threshold,
        )
        if top1["score"] < threshold or margin_value < max(margin, 0.0):
            logger.debug(
                "No speaker bank match met threshold/margin (score=%.4f margin=%.4f threshold=%.4f margin_req=%.4f)",
                top1["score"],
                margin_value,
                threshold,
                margin,
            )
            return None

        result = dict(top1)
        result["margin"] = margin_value
        result["second_best"] = top2_score
        return result

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
                        "prototypes": len(cluster.prototype_indices),
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
                ax.scatter(
                    coords[idx, 0], coords[idx, 1], c=[color], label=meta.speaker, alpha=0.7, s=24
                )
                if annotate:
                    ax.annotate(
                        meta.cluster_id or idx,
                        (coords[idx, 0], coords[idx, 1]),
                        fontsize=6,
                        alpha=0.6,
                    )
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
