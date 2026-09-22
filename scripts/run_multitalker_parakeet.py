from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


DEFAULT_ASR_MODEL = "nvidia/multitalker-parakeet-streaming-0.6b-v1"


def _build_config(args: argparse.Namespace):
    from omegaconf import OmegaConf

    conditioning = str(args.conditioning)
    return OmegaConf.create(
        {
            "audio_file": str(args.audio),
            "manifest_file": None,
            "output_path": str(args.output),
            "max_num_of_spks": int(args.max_speakers),
            "parallel_speaker_strategy": True,
            "masked_asr": conditioning != "kernel",
            "mask_preencode": conditioning == "masked-preencode",
            "cache_gating": not args.disable_cache_gating,
            "cache_gating_buffer_size": int(args.cache_gating_buffer_size),
            "single_speaker_mode": False,
            "batch_size": 1,
            "att_context_size": list(args.att_context_size),
            "online_normalization": False,
            "pad_and_drop_preencoded": False,
            "chunk_size": -1,
            "shift_size": -1,
            "left_chunks": 2,
            "word_window": 50,
            "sent_break_sec": 30.0,
            "fix_prev_words_count": 5,
            "update_prev_words_sentence": 5,
            "ignored_initial_frame_steps": 5,
            "discarded_frames": 8,
            "left_frame_shift": -1,
            "right_frame_shift": 0,
            "min_sigmoid_val": 1e-2,
            "binary_diar_preds": bool(args.binary_diarization),
            "generate_realtime_scripts": False,
            "print_sample_indices": [],
            "colored_text": False,
            "real_time_mode": False,
            "verbose": False,
            "log": bool(args.verbose),
            "streaming_mode": True,
            "spkcache_len": int(args.speaker_cache_length),
            "spkcache_refresh_rate": 0,
            "fifo_len": int(args.fifo_length),
            "chunk_len": int(args.diar_chunk_length),
            "chunk_left_context": 0,
            "chunk_right_context": int(args.diar_right_context),
        }
    )


def _load_diarization_model(model_spec: str, device):
    from nemo.collections.asr.models import SortformerEncLabelModel

    path = Path(model_spec)
    if path.suffix == ".nemo" and path.exists():
        return SortformerEncLabelModel.restore_from(str(path), map_location=device)
    if path.suffix == ".ckpt" and path.exists():
        return SortformerEncLabelModel.load_from_checkpoint(
            checkpoint_path=str(path), map_location=device, strict=False
        )
    return SortformerEncLabelModel.from_pretrained(model_spec, map_location=device)


def _load_asr_model(model_spec: str, device):
    from nemo.collections.asr.models import ASRModel

    path = Path(model_spec)
    if path.suffix == ".nemo" and path.exists():
        return ASRModel.restore_from(restore_path=str(path), map_location=device)
    return ASRModel.from_pretrained(model_spec, map_location=device)


def _configure_diarization_model(model, cfg):
    model.streaming_mode = True
    model.sortformer_modules.chunk_len = cfg.chunk_len
    model.sortformer_modules.spkcache_len = cfg.spkcache_len
    model.sortformer_modules.chunk_left_context = cfg.chunk_left_context
    model.sortformer_modules.chunk_right_context = cfg.chunk_right_context
    model.sortformer_modules.fifo_len = cfg.fifo_len
    model.sortformer_modules.log = cfg.log
    model.sortformer_modules.spkcache_refresh_rate = cfg.spkcache_refresh_rate
    model.rttms_mask_mats = None
    return model.eval()


def _set_oracle_rttm_lines_mask(
    *,
    diar_model,
    rttm_lines: Sequence[str],
    offset: float,
    duration: float,
    max_speakers: int,
    collar_seconds: float,
    device,
) -> None:
    import torch.nn.functional as functional
    from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
        collate_matrices,
        extract_frame_info_from_rttm,
        get_frame_targets_from_rttm,
    )

    timestamps, _ = extract_frame_info_from_rttm(offset, duration, list(rttm_lines))
    mask = get_frame_targets_from_rttm(
        rttm_timestamps=timestamps,
        offset=offset,
        duration=duration,
        round_digits=3,
        feat_per_sec=12.5,
        max_spks=max_speakers,
    )
    collar_frames = int(round(max(0.0, collar_seconds) * 12.5))
    if collar_frames:
        channels_first = mask.transpose(0, 1).unsqueeze(0)
        channels_first = functional.max_pool1d(
            channels_first,
            kernel_size=2 * collar_frames + 1,
            stride=1,
            padding=collar_frames,
        )
        mask = channels_first.squeeze(0).transpose(0, 1)
    diar_model.rttms_mask_mats = None
    diar_model.add_rttms_mask_mats(collate_matrices([mask]), device=device)


def _set_oracle_rttm_mask(
    *,
    diar_model,
    rttm_path: Path,
    offset: float,
    duration: float,
    max_speakers: int,
    collar_seconds: float,
    device,
) -> None:
    _set_oracle_rttm_lines_mask(
        diar_model=diar_model,
        rttm_lines=rttm_path.read_text(encoding="utf-8").splitlines(),
        offset=offset,
        duration=duration,
        max_speakers=max_speakers,
        collar_seconds=collar_seconds,
        device=device,
    )


def _stream_audio(*, cfg, asr_model, diar_model) -> list[dict]:
    import torch
    from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=cfg.online_normalization,
        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
    )
    streaming_buffer.append_audio_file(audio_filepath=cfg.audio_file, stream_id=-1)
    streamer = SpeakerTaggedASR(cfg, asr_model, diar_model)

    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer):
        drop_extra = (
            0
            if step_num == 0 and not cfg.pad_and_drop_preencoded
            else asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
        )
        with torch.inference_mode(), torch.amp.autocast(asr_model.device.type, enabled=True):
            streamer.perform_parallel_streaming_stt_spk(
                step_num=step_num,
                chunk_audio=chunk_audio,
                chunk_lengths=chunk_lengths,
                is_buffer_empty=streaming_buffer.is_buffer_empty(),
                drop_extra_pre_encoded=drop_extra,
            )

    samples = [{"audio_filepath": cfg.audio_file}]
    return streamer.generate_seglst_dicts_from_parallel_streaming(samples=samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Sortformer-conditioned NVIDIA multitalker Parakeet on mono audio."
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diar-model", required=True)
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument(
        "--conditioning",
        choices=("kernel", "masked", "masked-preencode"),
        default="kernel",
    )
    parser.add_argument("--max-speakers", type=int, default=4)
    parser.add_argument("--att-context-size", type=int, nargs=2, default=(70, 13))
    parser.add_argument("--diar-chunk-length", type=int, default=6)
    parser.add_argument("--diar-right-context", type=int, default=7)
    parser.add_argument("--speaker-cache-length", type=int, default=188)
    parser.add_argument("--fifo-length", type=int, default=188)
    parser.add_argument("--cache-gating-buffer-size", type=int, default=2)
    parser.add_argument("--disable-cache-gating", action="store_true")
    parser.add_argument("--binary-diarization", action="store_true")
    parser.add_argument(
        "--oracle-rttm",
        type=Path,
        help="Diagnostic only: replace predicted diarization with an RTTM activity mask.",
    )
    parser.add_argument("--rttm-offset", type=float, default=0.0)
    parser.add_argument("--rttm-duration", type=float)
    parser.add_argument("--rttm-collar", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.audio.exists():
        raise FileNotFoundError(args.audio)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Multitalker Parakeet requires a CUDA GPU for this experiment.")

    device = torch.device("cuda")
    cfg = _build_config(args)
    diar_model = _configure_diarization_model(
        _load_diarization_model(args.diar_model, device).to(device), cfg
    )
    asr_model = _load_asr_model(args.asr_model, device).eval().to(device)
    asr_model.encoder.set_default_att_context_size(att_context_size=cfg.att_context_size)
    if args.oracle_rttm:
        if not args.oracle_rttm.exists():
            raise FileNotFoundError(args.oracle_rttm)
        duration = args.rttm_duration
        if duration is None:
            import soundfile as sf

            duration = sf.info(str(args.audio)).duration
        _set_oracle_rttm_mask(
            diar_model=diar_model,
            rttm_path=args.oracle_rttm,
            offset=args.rttm_offset,
            duration=float(duration),
            max_speakers=args.max_speakers,
            collar_seconds=args.rttm_collar,
            device=device,
        )

    segments = _stream_audio(cfg=cfg, asr_model=asr_model, diar_model=diar_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(segments, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "audio": str(args.audio),
                "output": str(args.output),
                "conditioning": args.conditioning,
                "speaker_supervision": "oracle-rttm" if args.oracle_rttm else "sortformer",
                "rttm_collar": args.rttm_collar if args.oracle_rttm else None,
                "segment_count": len(segments),
                "predicted_speakers": sorted({str(item["speaker"]) for item in segments}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
