from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .asr import AsrResult, transcribe_with_faster_whisper
from .diarization import (
    DiarizationResult,
    DiarizationTurn,
    _detect_device as _detect_diarization_device,
    diarize_audio,
    extract_embeddings_for_segments,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptPipelineResult:
    segments: List[dict]
    diarization_segments: List[dict]
    exclusive_diarization_segments: List[dict]
    speaker_embeddings: Dict[str, np.ndarray]
    metadata: dict


def _turn_to_dict(turn: DiarizationTurn) -> dict:
    return {"start": float(turn.start), "end": float(turn.end), "speaker": turn.speaker}


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _choose_turn_label(
    start: float,
    end: float,
    primary_turns: Sequence[DiarizationTurn],
    fallback_turns: Sequence[DiarizationTurn],
) -> Optional[str]:
    def _pick(turns: Sequence[DiarizationTurn]) -> Optional[str]:
        best_label: Optional[str] = None
        best_overlap = 0.0
        midpoint = (start + end) / 2.0
        midpoint_label: Optional[str] = None
        midpoint_distance = float("inf")
        for turn in turns:
            overlap = _overlap(start, end, turn.start, turn.end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = turn.speaker
            if turn.start <= midpoint <= turn.end:
                midpoint_distance = 0.0
                midpoint_label = turn.speaker
            elif midpoint_distance > 0.0:
                distance = min(abs(midpoint - turn.start), abs(midpoint - turn.end))
                if distance < midpoint_distance:
                    midpoint_distance = distance
                    midpoint_label = turn.speaker
        if best_label is not None and best_overlap > 0.0:
            return best_label
        return midpoint_label

    return _pick(primary_turns) or _pick(fallback_turns)


def _assign_word_speakers(
    asr_result: AsrResult,
    diarization: DiarizationResult,
) -> List[dict]:
    exclusive_turns = diarization.exclusive_segments or []
    fallback_turns = diarization.segments or []
    structured_segments: List[dict] = []

    def _words_to_text(words: Sequence[dict]) -> str:
        text = " ".join(
            str(word.get("word") or word.get("text") or "").strip()
            for word in words
            if str(word.get("word") or word.get("text") or "").strip()
        ).strip()
        return text

    def _append_group(words: Sequence[dict], default_start: float, default_end: float) -> None:
        if not words:
            return
        speaker = words[0].get("speaker")
        start = float(words[0].get("start") or default_start)
        end = float(words[-1].get("end") or default_end)
        structured_segments.append(
            {
                "start": start,
                "end": end,
                "text": _words_to_text(words),
                "speaker": speaker,
                "speaker_raw": speaker,
                "words": list(words),
            }
        )

    for segment in asr_result.segments:
        payload = {
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
            "speaker": None,
            "speaker_raw": None,
        }
        word_payload: List[dict] = []
        speaker_votes: Dict[str, float] = {}

        for word in segment.words:
            label = _choose_turn_label(word.start, word.end, exclusive_turns, fallback_turns)
            word_entry = word.to_dict()
            if label is not None:
                word_entry["speaker"] = label
                word_entry["speaker_raw"] = label
                speaker_votes[label] = speaker_votes.get(label, 0.0) + max(
                    word.end - word.start, 1e-3
                )
            word_payload.append(word_entry)

        if speaker_votes:
            max_vote = max(speaker_votes.values())
            tied = sorted(label for label, vote in speaker_votes.items() if vote == max_vote)
            if len(tied) == 1:
                segment_label = tied[0]
            else:
                segment_label = _choose_turn_label(
                    float(segment.start),
                    float(segment.end),
                    exclusive_turns,
                    fallback_turns,
                )
        else:
            segment_label = _choose_turn_label(
                float(segment.start),
                float(segment.end),
                exclusive_turns,
                fallback_turns,
            )
        if word_payload:
            current_group: List[dict] = []
            for word in word_payload:
                word_speaker = word.get("speaker")
                if current_group and word_speaker != current_group[-1].get("speaker"):
                    _append_group(current_group, float(segment.start), float(segment.end))
                    current_group = []
                current_group.append(word)
            _append_group(current_group, float(segment.start), float(segment.end))
            continue
        if segment_label is not None:
            payload["speaker"] = segment_label
            payload["speaker_raw"] = segment_label
        structured_segments.append(payload)

    return structured_segments


def _merge_adjacent_turns(
    turns: Sequence[DiarizationTurn],
    *,
    max_gap_seconds: float = 0.20,
    min_duration: float = 0.80,
) -> List[DiarizationTurn]:
    if not turns:
        return []
    ordered = sorted(turns, key=lambda item: (item.start, item.end, item.speaker))
    merged: List[DiarizationTurn] = []
    current = ordered[0]
    for turn in ordered[1:]:
        if turn.speaker == current.speaker and turn.start - current.end <= max_gap_seconds:
            current = DiarizationTurn(
                start=current.start, end=max(current.end, turn.end), speaker=current.speaker
            )
            continue
        if current.end - current.start >= min_duration:
            merged.append(current)
        current = turn
    if current.end - current.start >= min_duration:
        merged.append(current)
    return merged


def _aggregate_speaker_embeddings(
    audio_path: str,
    diarization: DiarizationResult,
    *,
    hf_token: Optional[str],
    diarization_model_name: Optional[str],
    force_device: Optional[str],
    quiet: bool,
    pre_pad: float = 0.15,
    post_pad: float = 0.15,
    batch_size: int = 16,
) -> Dict[str, np.ndarray]:
    turns = _merge_adjacent_turns(diarization.exclusive_segments or diarization.segments)
    if not turns:
        turns = list(diarization.exclusive_segments or diarization.segments)
    if not turns:
        return {}
    payload = [(turn.start, turn.end, turn.speaker) for turn in turns]
    embed_results, _ = extract_embeddings_for_segments(
        audio_path,
        payload,
        hf_token,
        diarization_model_name=diarization_model_name,
        force_device=force_device,
        quiet=quiet,
        pre_pad=pre_pad,
        post_pad=post_pad,
        batch_size=batch_size,
    )
    by_label: Dict[str, List[np.ndarray]] = {}
    for result in embed_results:
        by_label.setdefault(result.speaker, []).append(
            np.asarray(result.embedding, dtype=np.float32)
        )

    embeddings: Dict[str, np.ndarray] = {}
    for label, vectors in by_label.items():
        matrix = np.stack(vectors)
        centroid = np.mean(matrix, axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm <= 0.0:
            continue
        embeddings[label] = (centroid / norm).astype(np.float32)
    return embeddings


def transcribe_with_faster_pipeline(
    audio_path: str,
    *,
    model_name: str,
    compute_type: str = "float16",
    batch_size: int = 32,
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    model_cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    pyannote_on_cpu: bool = False,
    diarization_model_name: Optional[str] = None,
    force_device: Optional[str] = None,
    quiet: bool = True,
    enable_diarization: bool = True,
) -> TranscriptPipelineResult:
    asr_device = force_device if force_device in {"cpu", "cuda"} else _detect_diarization_device()
    asr_compute = compute_type
    cpu_supported = {"int8", "int8_float32", "int8_np"}
    if asr_device == "cpu" and asr_compute not in cpu_supported:
        asr_compute = "int8"

    asr_result = transcribe_with_faster_whisper(
        audio_path,
        model_name=model_name,
        compute_type=asr_compute,
        device=asr_device,
        download_root=model_cache_dir,
        local_files_only=local_files_only,
        batch_size=batch_size,
    )
    structured_segments = asr_result.to_dict_segments()
    diarization_segments: List[dict] = []
    exclusive_segments: List[dict] = []
    speaker_embeddings: Dict[str, np.ndarray] = {}
    metadata = {"asr": asr_result.metadata, "language": asr_result.language}

    if enable_diarization:
        diar_device = "cpu" if pyannote_on_cpu else force_device
        try:
            diarization = diarize_audio(
                audio_path,
                model_name=diarization_model_name,
                hf_token=hf_token,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                device=diar_device,
            )
            structured_segments = _assign_word_speakers(asr_result, diarization)
            diarization_segments = [_turn_to_dict(turn) for turn in diarization.segments]
            exclusive_segments = [_turn_to_dict(turn) for turn in diarization.exclusive_segments]
            speaker_embeddings = _aggregate_speaker_embeddings(
                audio_path,
                diarization,
                hf_token=hf_token,
                diarization_model_name=diarization_model_name,
                force_device=diar_device,
                quiet=quiet,
            )
            metadata["diarization"] = diarization.metadata
        except Exception as exc:  # noqa: BLE001
            logger.warning("Direct pyannote diarization failed for %s: %s", audio_path, exc)

    return TranscriptPipelineResult(
        segments=structured_segments,
        diarization_segments=diarization_segments,
        exclusive_diarization_segments=exclusive_segments,
        speaker_embeddings=speaker_embeddings,
        metadata=metadata,
    )
