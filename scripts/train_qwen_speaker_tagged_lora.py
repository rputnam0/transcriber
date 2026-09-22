from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf
import torch

from build_speaker_tagged_asr_dataset import tagged_transcript


DEFAULT_CONTEXT = (
    "Transcribe the English meeting audio and include speaker changes inline. "
    "Use only [S0], [S1], [S2], and [S3] speaker tags. "
    "Assign speakers by first appearance in the clip and repeat the tag before each speaker turn."
)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_wave(path: Path) -> np.ndarray:
    wave, _ = sf.read(path, dtype="float32", always_2d=False)
    array = np.asarray(wave, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _target_text(row: Mapping[str, object], *, max_speakers: int) -> str:
    text = str(row.get("target_text") or "").strip()
    if text:
        return text
    generated, _ = tagged_transcript(row.get("words") or [], max_speakers=max_speakers)
    return generated


def _prepare_supervised_inputs(
    qwen_model,
    row: Mapping[str, object],
    *,
    context: str,
    max_speakers: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    audio_path = Path(str(row.get("audio_path") or ""))
    wave = _load_wave(audio_path)
    prompt = qwen_model._build_text_prompt(context=context, force_language=None)
    full_text = prompt + _target_text(row, max_speakers=max_speakers)
    prompt_inputs = qwen_model.processor(
        text=[prompt],
        audio=[wave],
        return_tensors="pt",
        padding=True,
    )
    full_inputs = qwen_model.processor(
        text=[full_text],
        audio=[wave],
        return_tensors="pt",
        padding=True,
    )
    labels = full_inputs["input_ids"].clone()
    labels[:, : prompt_inputs["input_ids"].shape[1]] = -100
    labels[full_inputs["attention_mask"] == 0] = -100
    full_inputs = full_inputs.to(qwen_model.model.device).to(qwen_model.model.dtype)
    labels = labels.to(qwen_model.model.device)
    return dict(full_inputs), labels


def _fit_lora(
    qwen_model,
    rows: Sequence[Mapping[str, object]],
    *,
    context: str,
    max_speakers: int,
    max_steps: int,
    learning_rate: float,
    seed: int,
    log_every: int,
) -> list[dict]:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(qwen_model.model.thinker.parameters(), lr=learning_rate)
    history = []
    qwen_model.model.thinker.train()
    start_time = time.time()
    for step in range(1, max_steps + 1):
        row = rows[(step - 1) % len(rows)] if step <= len(rows) else rng.choice(rows)
        inputs, labels = _prepare_supervised_inputs(
            qwen_model,
            row,
            context=context,
            max_speakers=max_speakers,
        )
        output = qwen_model.model.thinker(**inputs, labels=labels, use_cache=False)
        loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_value = float(loss.detach().cpu())
        item = {
            "step": step,
            "loss": loss_value,
            "clip_id": row.get("clip_id"),
            "elapsed_seconds": time.time() - start_time,
        }
        history.append(item)
        if log_every > 0 and (step == 1 or step % log_every == 0 or step == max_steps):
            print(json.dumps(item), flush=True)
    qwen_model.model.thinker.eval()
    return history


def _run_predictions(
    qwen_model,
    rows: Sequence[Mapping[str, object]],
    *,
    context: str,
    max_new_tokens: int,
) -> list[dict]:
    qwen_model.max_new_tokens = int(max_new_tokens)
    qwen_model.model.eval()
    predictions = []
    for row in rows:
        clip_id = str(row.get("clip_id") or "")
        audio_path = str(row.get("audio_path") or "")
        started = time.time()
        result = qwen_model.transcribe(
            audio_path,
            context=context,
            language=None,
            return_time_stamps=False,
        )[0]
        predictions.append(
            {
                "clip_id": clip_id,
                "audio_path": audio_path,
                "elapsed_seconds": time.time() - started,
                "language": result.language,
                "text": result.text,
            }
        )
        print(json.dumps(predictions[-1], ensure_ascii=False), flush=True)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA-adapt Qwen3 speaker-tagged ASR on domain forced-word clips."
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-id",
        default="mrfakename/qwen3-asr-1.7b-ami-diarization-fft-r6-20260422",
    )
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    parser.add_argument("--max-speakers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    from peft import LoraConfig, TaskType, get_peft_model
    from qwen_asr import Qwen3ASRModel

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = _read_jsonl(args.train_manifest)
    if args.max_train_rows > 0:
        train_rows = train_rows[: args.max_train_rows]
    eval_rows = _read_jsonl(args.eval_manifest)
    if not train_rows:
        raise ValueError("No train rows available")
    if not eval_rows:
        raise ValueError("No eval rows available")

    qwen_model = Qwen3ASRModel.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens=args.max_new_tokens,
        max_inference_batch_size=1,
    )
    qwen_model.model.thinker.gradient_checkpointing_enable()
    qwen_model.model.thinker.config.use_cache = False
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    qwen_model.model.thinker = get_peft_model(qwen_model.model.thinker, config)
    qwen_model.model.thinker.print_trainable_parameters()

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    history = _fit_lora(
        qwen_model,
        train_rows,
        context=DEFAULT_CONTEXT,
        max_speakers=args.max_speakers,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        log_every=args.log_every,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.output_dir / "adapter"
    qwen_model.model.thinker.save_pretrained(adapter_dir)
    _write_jsonl(args.output_dir / "training_history.jsonl", history)

    predictions = _run_predictions(
        qwen_model,
        eval_rows,
        context=DEFAULT_CONTEXT,
        max_new_tokens=args.max_new_tokens,
    )
    _write_jsonl(args.output_dir / "speaker_tagged_asr_predictions.jsonl", predictions)

    summary = {
        "model_id": args.model_id,
        "train_manifest": str(args.train_manifest),
        "eval_manifest": str(args.eval_manifest),
        "output_dir": str(args.output_dir),
        "adapter_dir": str(adapter_dir),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
        "final_loss": history[-1]["loss"] if history else None,
        "min_loss": min((item["loss"] for item in history), default=None),
        "peak_cuda_memory_gb": (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
        ),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
