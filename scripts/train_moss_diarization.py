from __future__ import annotations

import json
import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from moss_activity_conditioning import install_activity_conditioning, load_activity_weights
from moss_target_speaker_conditioning import (
    install_target_speaker_conditioning,
    load_target_speaker_weights,
)


DEFAULT_MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
DEFAULT_REVISION = "e8681d68e7042738ffca8ac8212bc8fcb1131ab8"
OFFICIAL_CODE_REVISION = "0e3d1403fd8f1f1c674e883ece96b9f630794ebe"


def promote_trainable_parameters_to_float32(model) -> dict[str, int]:
    import torch

    promoted_parameters = 0
    promoted_elements = 0
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype.is_floating_point:
            if parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
                promoted_parameters += 1
                promoted_elements += parameter.numel()
    return {
        "promoted_parameters": promoted_parameters,
        "promoted_elements": promoted_elements,
    }


def loss_weights_from_offsets(
    offsets: Sequence[Sequence[int]],
    spans: Sequence[Mapping[str, object]],
) -> list[float]:
    weights = []
    for start, end in offsets:
        matches = []
        if end > start:
            for span in spans:
                if int(span.get("start") or 0) < end and int(span.get("end") or 0) > start:
                    weight = float(span.get("weight", 1.0))
                    if weight < 0:
                        raise ValueError("Token loss weights must be nonnegative")
                    matches.append(weight)
        weights.append(max(matches, default=1.0))
    return weights


def weighted_causal_loss(logits, labels, loss_weights):
    import torch.nn.functional as functional

    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = loss_weights[..., 1:].contiguous().float()
    token_loss = functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    valid = shift_labels.ne(-100)
    weights = shift_weights * valid
    return (token_loss * weights).sum() / weights.sum().clamp_min(1.0)


def activity_bce_loss(
    logits_by_sample,
    overlap_logits_by_sample,
    targets,
    mask,
    speech_logits_by_sample=None,
):
    import torch
    import torch.nn.functional as functional

    activity_losses = []
    activity_weights = []
    speech_losses = []
    speech_weights = []
    overlap_losses = []
    overlap_weights = []
    for index, logits in enumerate(logits_by_sample):
        values = logits.squeeze(0).float()
        frame_count = min(values.shape[0], targets.shape[1])
        values = values[:frame_count]
        truth = targets[index, :frame_count].float()
        valid = mask[index, :frame_count].float().unsqueeze(-1)
        slot_cost = functional.binary_cross_entropy_with_logits(
            values.unsqueeze(-1).expand(-1, -1, truth.shape[-1]),
            truth.unsqueeze(-2).expand(-1, values.shape[-1], -1),
            reduction="none",
        )
        slot_cost = (slot_cost * valid.unsqueeze(-1)).sum(dim=0)
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(slot_cost.detach().cpu().numpy())
        permuted_truth = torch.zeros_like(truth)
        permuted_truth[:, rows] = truth[:, columns]
        truth = permuted_truth
        overlap_weight = 1.0 + 3.0 * truth.sum(dim=-1, keepdim=True).ge(2).float()
        positive_weight = 1.0 + 2.0 * truth
        element_weight = valid * overlap_weight * positive_weight
        activity_losses.append(
            functional.binary_cross_entropy_with_logits(values, truth, reduction="none")
            * element_weight
        )
        activity_weights.append(element_weight)
        if speech_logits_by_sample:
            speech_values = (
                speech_logits_by_sample[index].squeeze(0).squeeze(-1).float()[:frame_count]
            )
            speech_truth = truth.sum(dim=-1).ge(1).float()
            valid_frames = mask[index, :frame_count].float()
            positive = (speech_truth * valid_frames).sum()
            negative = ((1.0 - speech_truth) * valid_frames).sum()
            positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 10.0)
            direct_speech_weight = valid_frames * (1.0 + (positive_weight - 1.0) * speech_truth)
            speech_losses.append(
                functional.binary_cross_entropy_with_logits(
                    speech_values,
                    speech_truth,
                    reduction="none",
                )
                * direct_speech_weight
            )
            speech_weights.append(direct_speech_weight)
        overlap_values = (
            overlap_logits_by_sample[index].squeeze(0).squeeze(-1).float()[:frame_count]
        )
        overlap_truth = truth.sum(dim=-1).ge(2).float()
        valid_frames = mask[index, :frame_count].float()
        positive = (overlap_truth * valid_frames).sum()
        negative = ((1.0 - overlap_truth) * valid_frames).sum()
        positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 10.0)
        direct_weight = valid_frames * (1.0 + (positive_weight - 1.0) * overlap_truth)
        overlap_losses.append(
            functional.binary_cross_entropy_with_logits(
                overlap_values,
                overlap_truth,
                reduction="none",
            )
            * direct_weight
        )
        overlap_weights.append(direct_weight)
    if not activity_losses:
        return targets.new_zeros((), dtype=torch.float32)
    activity_loss = sum(loss.sum() for loss in activity_losses) / sum(
        weight.sum() for weight in activity_weights
    ).clamp_min(1.0)
    overlap_loss = sum(loss.sum() for loss in overlap_losses) / sum(
        weight.sum() for weight in overlap_weights
    ).clamp_min(1.0)
    if not speech_losses:
        return activity_loss + overlap_loss
    speech_loss = sum(loss.sum() for loss in speech_losses) / sum(
        weight.sum() for weight in speech_weights
    ).clamp_min(1.0)
    return 0.5 * activity_loss + 0.5 * speech_loss + overlap_loss


def target_activity_bce_loss(logits_by_sample, targets, mask):
    import torch.nn.functional as functional

    losses = []
    weights = []
    for index, logits in enumerate(logits_by_sample):
        values = logits.squeeze(0).squeeze(-1).float()
        frame_count = min(values.shape[0], targets.shape[1])
        values = values[:frame_count]
        truth = targets[index, :frame_count, 0].float()
        valid = mask[index, :frame_count].float()
        positive = (truth * valid).sum()
        negative = ((1.0 - truth) * valid).sum()
        positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 10.0)
        element_weight = valid * (1.0 + (positive_weight - 1.0) * truth)
        losses.append(
            functional.binary_cross_entropy_with_logits(values, truth, reduction="none")
            * element_weight
        )
        weights.append(element_weight)
    if not losses:
        return targets.new_zeros((), dtype=targets.dtype)
    return sum(loss.sum() for loss in losses) / sum(weight.sum() for weight in weights).clamp_min(
        1.0
    )


def load_samples(path: str, *, forbidden_sessions: set[str]) -> list[dict]:
    manifest = Path(path).expanduser().resolve()
    samples = []
    with manifest.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = dict(row.get("metadata") or {})
            # An explicit identity in an unsure review is not a training target.
            if (
                metadata.get("verdict") == "unsure"
                or metadata.get("annotation_verdict") == "unsure"
            ):
                continue
            if metadata.get("mono_input_only") is not True:
                raise ValueError(f"Line {line_no}: training input is not declared mono-only")
            session = str(metadata.get("session") or "")
            source_sessions = {session, *(str(s) for s in metadata.get("source_sessions", []))}
            forbidden_ids = {s.removeprefix("Session ").strip() for s in forbidden_sessions}
            leaked = {
                s for s in source_sessions if s.removeprefix("Session ").strip() in forbidden_ids
            }
            if leaked:
                raise ValueError(f"Line {line_no}: forbidden holdout session {sorted(leaked)}")
            conversation = row.get("conversation") or []
            expected = [("user", "text"), ("user", "audio"), ("assistant", "text")]
            actual = [
                (item.get("role"), item.get("message_type"))
                for item in conversation
                if isinstance(item, dict)
            ]
            if not isinstance(conversation, list) or actual != expected:
                raise ValueError(f"Line {line_no}: expected user/text, user/audio, assistant/text")
            prompt, audio_path, target = (item.get("content") for item in conversation)
            if not all(
                isinstance(value, str) and value.strip() for value in (prompt, audio_path, target)
            ):
                raise ValueError(
                    f"Line {line_no}: prompt, audio path, and target must be non-empty"
                )
            audio = Path(audio_path).expanduser()
            if not audio.is_absolute():
                audio = (manifest.parent / audio).resolve()
            if not audio.is_file():
                raise FileNotFoundError(audio)
            samples.append(
                {
                    "audio": str(audio),
                    "prompt": prompt.strip(),
                    "target": target.strip(),
                    "session": session,
                    "loss_spans": list(metadata.get("loss_spans") or []),
                    "activity": list(metadata.get("activity") or []),
                    "activity_supervised": bool(metadata.get("activity_supervised", False)),
                    "activity_valid_start": float(metadata.get("activity_valid_start") or 0.0),
                    "activity_valid_end": float(metadata.get("activity_valid_end") or 0.0),
                    "sample_loss_weight": float(metadata.get("sample_loss_weight") or 1.0),
                    "eos_loss_weight": float(metadata.get("eos_loss_weight", 1.0)),
                }
            )
    if not samples:
        raise ValueError(f"No samples found in {manifest}")
    return samples


@dataclass
class ScriptArguments:
    train_jsonl: str = field(metadata={"help": "Conversation-format training manifest."})
    eval_jsonl: str = ""
    model_name_or_path: str = DEFAULT_MODEL
    model_revision: str = DEFAULT_REVISION
    max_length: int = 8192
    attn_implementation: str = "sdpa"
    forbidden_sessions: str = "Session 34"
    max_activity_speakers: int = 8
    activity_loss_weight: float = 1.0
    activity_head_only: bool = False
    use_activity_conditioning: bool = True
    use_target_speaker_conditioning: bool = False
    target_conditioning_only: bool = False
    target_profile_seconds: float = 4.0
    target_gap_seconds: float = 0.5
    target_activity_loss_weight: float = 2.0
    promote_trainable_parameters_fp32: bool = False
    stable_speaker_names: str = ""
    text_pad_multiple: int = 0


class ConversationDataset:
    def __init__(self, path: str, *, forbidden_sessions: set[str]):
        self.samples = load_samples(path, forbidden_sessions=forbidden_sessions)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class DataCollator:
    def __init__(
        self, processor, max_length: int, *, max_activity_speakers: int, text_pad_multiple: int = 0
    ):
        self.processor = processor
        self.max_length = max_length
        self.max_activity_speakers = max_activity_speakers
        self.sample_rate = int(processor.feature_extractor.sampling_rate)
        self.text_pad_multiple = text_pad_multiple
        self.observed_text_lengths = set()

    def __call__(self, samples: list[dict]):
        import soundfile as sf
        import soxr
        import torch

        from moss_transcribe_diarize.inference_utils import build_transcription_messages

        prompts, texts, audios = [], [], []
        for sample in samples:
            prompt = self.processor.apply_chat_template(
                build_transcription_messages(sample["audio"], sample["prompt"]),
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
            texts.append(prompt + sample["target"] + self.processor.tokenizer.eos_token)
            audio, sample_rate = sf.read(sample["audio"], dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if sample_rate != self.sample_rate:
                audio = soxr.resample(audio, sample_rate, self.sample_rate)
            audios.append(audio)

        batch = self.processor(
            text=texts,
            audio=audios,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if self.text_pad_multiple:
            length = batch["input_ids"].shape[1]
            padded = (
                (length + self.text_pad_multiple - 1) // self.text_pad_multiple
            ) * self.text_pad_multiple
            if padded > self.max_length:
                raise ValueError("Training sequence exceeds the configured text length")
            batch["input_ids"] = torch.nn.functional.pad(
                batch["input_ids"],
                (0, padded - length),
                value=self.processor.tokenizer.pad_token_id,
            )
            batch["attention_mask"] = torch.nn.functional.pad(
                batch["attention_mask"],
                (0, padded - length),
                value=0,
            )
        self.observed_text_lengths.add(batch["input_ids"].shape[1])
        audio_lengths = torch.zeros(len(samples), dtype=torch.long)
        audio_lengths.scatter_add_(
            0,
            batch["audio_chunk_mapping"].cpu(),
            batch["audio_feature_lengths"].cpu(),
        )
        labels = batch["input_ids"].clone()
        loss_weights = torch.zeros_like(labels, dtype=torch.float32)
        for index, (prompt, audio_length) in enumerate(zip(prompts, audio_lengths.tolist())):
            prompt_ids = self.processor.expand_audio_token(
                prompt,
                audio_length,
                self.max_length,
            )
            labels[index, : len(prompt_ids)] = -100
            encoded_target = self.processor.tokenizer(
                samples[index]["target"],
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            target_ids = list(encoded_target["input_ids"])
            actual = batch["input_ids"][
                index,
                len(prompt_ids) : len(prompt_ids) + len(target_ids),
            ].tolist()
            if actual != target_ids:
                raise ValueError("Target token suffix does not match processor input IDs")
            token_weights = loss_weights_from_offsets(
                list(encoded_target["offset_mapping"]),
                samples[index]["loss_spans"],
            )
            token_weights = [
                weight * samples[index]["sample_loss_weight"] for weight in token_weights
            ]
            loss_weights[
                index,
                len(prompt_ids) : len(prompt_ids) + len(target_ids),
            ] = torch.tensor(token_weights, dtype=torch.float32)
            eos_position = len(prompt_ids) + len(target_ids)
            if eos_position < loss_weights.shape[1]:
                loss_weights[index, eos_position] = samples[index]["sample_loss_weight"] * samples[
                    index
                ].get("eos_loss_weight", 1.0)
        labels[batch["attention_mask"] == 0] = -100
        loss_weights[labels == -100] = 0.0

        max_audio_length = int(audio_lengths.max().item())
        activity_labels = torch.zeros(
            len(samples),
            max_audio_length,
            self.max_activity_speakers,
            dtype=torch.float32,
        )
        activity_mask = torch.zeros(
            len(samples),
            max_audio_length,
            dtype=torch.bool,
        )
        tokens_per_second = float(self.processor.audio_tokens_per_second)
        for index, (sample, audio_length) in enumerate(zip(samples, audio_lengths.tolist())):
            if not sample["activity"] and not sample["activity_supervised"]:
                continue
            frame_centers = (torch.arange(audio_length) + 0.5) / tokens_per_second
            valid_start = float(sample["activity_valid_start"])
            valid_end = float(sample["activity_valid_end"])
            valid = frame_centers.ge(valid_start)
            if valid_end > valid_start:
                valid &= frame_centers.lt(valid_end)
            activity_mask[index, :audio_length] = valid
            for interval in sample["activity"]:
                speaker = int(interval.get("speaker_index") or 0)
                if not 0 <= speaker < self.max_activity_speakers:
                    continue
                active = frame_centers.ge(float(interval.get("start") or 0.0)) & frame_centers.lt(
                    float(interval.get("end") or 0.0)
                )
                activity_labels[index, :audio_length, speaker][active] = 1.0
        batch["labels"] = labels
        batch["loss_weights"] = loss_weights
        batch["activity_labels"] = activity_labels
        batch["activity_mask"] = activity_mask
        return dict(batch)


def main() -> None:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        HfArgumentParser,
        Trainer,
        TrainingArguments,
        TrainerCallback,
    )

    from moss_transcribe_diarize.processing_moss_transcribe_diarize import (
        MossTranscribeDiarizeProcessor,
    )

    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_script = Path(__file__).read_bytes()
    manifest_sha256 = hashlib.sha256(Path(script_args.train_jsonl).read_bytes()).hexdigest()
    training_args.remove_unused_columns = False
    training_args.label_names = ["labels"]
    forbidden = {
        value.strip() for value in script_args.forbidden_sessions.split(",") if value.strip()
    }
    dataset = ConversationDataset(
        script_args.train_jsonl,
        forbidden_sessions=forbidden,
    )
    model_path = Path(script_args.model_name_or_path)
    processor_source = DEFAULT_MODEL if model_path.is_dir() else script_args.model_name_or_path
    processor = MossTranscribeDiarizeProcessor.from_pretrained(
        processor_source,
        revision=script_args.model_revision,
        trust_remote_code=True,
    )
    collator = DataCollator(
        processor,
        script_args.max_length,
        max_activity_speakers=script_args.max_activity_speakers,
        text_pad_multiple=script_args.text_pad_multiple,
    )
    dtype = (
        torch.bfloat16
        if training_args.bf16
        else torch.float16 if training_args.fp16 else torch.float32
    )
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": dtype,
        "attn_implementation": script_args.attn_implementation,
    }
    if not model_path.is_dir():
        model_kwargs["revision"] = script_args.model_revision
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        **model_kwargs,
    )
    model.tie_weights()
    if script_args.stable_speaker_names:
        model.config.stable_speaker_names = script_args.stable_speaker_names.split(",")
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    if script_args.activity_head_only and not script_args.use_activity_conditioning:
        raise ValueError("activity_head_only requires use_activity_conditioning")
    if script_args.use_activity_conditioning and script_args.use_target_speaker_conditioning:
        raise ValueError("Generic and target-speaker conditioning are mutually exclusive")
    activity_adaptor = None
    if script_args.use_activity_conditioning:
        has_activity_checkpoint = model_path.is_dir() and bool(
            getattr(model.config, "activity_conditioning", False)
        )
        activity_adaptor = install_activity_conditioning(
            model,
            max_speakers=script_args.max_activity_speakers,
            version=3,
        )
        if has_activity_checkpoint:
            load_activity_weights(model, model_path)
    target_adaptor = None
    if script_args.use_target_speaker_conditioning:
        has_target_checkpoint = model_path.is_dir() and bool(
            getattr(model.config, "target_speaker_conditioning", False)
        )
        target_adaptor = install_target_speaker_conditioning(
            model,
            profile_seconds=script_args.target_profile_seconds,
            gap_seconds=script_args.target_gap_seconds,
        )
        if has_target_checkpoint:
            load_target_speaker_weights(model, model_path)
    if script_args.activity_head_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        head_modules = [activity_adaptor.activity_head, activity_adaptor.overlap_head]
        if hasattr(activity_adaptor, "speech_head"):
            head_modules.append(activity_adaptor.speech_head)
        for module in head_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
    if script_args.target_conditioning_only:
        if target_adaptor is None:
            raise ValueError("target_conditioning_only requires target-speaker conditioning")
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in target_adaptor.target_parameters():
            parameter.requires_grad = True
    promotion = {"promoted_parameters": 0, "promoted_elements": 0}
    if script_args.promote_trainable_parameters_fp32 or script_args.target_conditioning_only:
        promotion = promote_trainable_parameters_to_float32(model)

    class WeightedActivityTrainer(Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            del num_items_in_batch
            labels = inputs.pop("labels")
            loss_weights = inputs.pop("loss_weights")
            activity_labels = inputs.pop("activity_labels")
            activity_mask = inputs.pop("activity_mask")
            if activity_adaptor is not None:
                activity_adaptor.clear_activity_logits()
            if target_adaptor is not None:
                target_adaptor.clear_target_activity_logits()
            if script_args.activity_head_only:
                model.model.get_audio_features(
                    input_features=inputs["input_features"],
                    audio_feature_lengths=inputs["audio_feature_lengths"],
                    audio_chunk_mapping=inputs["audio_chunk_mapping"],
                )
                outputs = None
                language_loss = activity_labels.new_zeros((), dtype=torch.float32)
            else:
                outputs = model(**inputs)
                language_loss = weighted_causal_loss(outputs.logits, labels, loss_weights)
            auxiliary_loss = (
                activity_bce_loss(
                    activity_adaptor.last_activity_logits,
                    activity_adaptor.last_overlap_logits,
                    activity_labels,
                    activity_mask,
                    getattr(activity_adaptor, "last_speech_logits", None),
                )
                if activity_adaptor is not None
                else activity_labels.new_zeros((), dtype=torch.float32)
            )
            target_loss = (
                target_activity_bce_loss(
                    target_adaptor.last_target_activity_logits,
                    activity_labels,
                    activity_mask,
                )
                if target_adaptor is not None
                else activity_labels.new_zeros((), dtype=torch.float32)
            )
            loss = (
                language_loss
                + script_args.activity_loss_weight * auxiliary_loss
                + script_args.target_activity_loss_weight * target_loss
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite MOSS training loss; checkpoint rejected")
            returned_outputs = outputs if outputs is not None else {"loss": loss}
            return (loss, returned_outputs) if return_outputs else loss

    class MemoryTelemetry(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if args.device.type == "mps" and state.global_step % 10 == 0:
                print(
                    json.dumps(
                        dict(
                            step=state.global_step,
                            mps_tensor_gib=round(torch.mps.current_allocated_memory() / 1024**3, 2),
                            mps_driver_gib=round(torch.mps.driver_allocated_memory() / 1024**3, 2),
                            text_lengths=sorted(collator.observed_text_lengths),
                        )
                    ),
                    flush=True,
                )

    trainer = WeightedActivityTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
        callbacks=[MemoryTelemetry()],
    )
    development = {}
    if script_args.eval_jsonl:
        from torch.utils.data import Subset

        evaluation = ConversationDataset(script_args.eval_jsonl, forbidden_sessions=set())
        cases = [
            json.loads(line)["metadata"]["case"]
            for line in Path(script_args.eval_jsonl).read_text().splitlines()
            if line.strip()
        ]
        if len(cases) != len(evaluation):
            raise ValueError("Development manifest includes excluded labels")
        train_sources = {
            str(s)
            for line in Path(script_args.train_jsonl).read_text().splitlines()
            if line.strip()
            for s in json.loads(line)["metadata"].get(
                "source_sessions", [json.loads(line)["metadata"]["session"]]
            )
        }
        dev_sources = {
            str(s)
            for line in Path(script_args.eval_jsonl).read_text().splitlines()
            if line.strip()
            for s in json.loads(line)["metadata"].get(
                "source_sessions", [json.loads(line)["metadata"]["session"]]
            )
        }
        if train_sources & dev_sources:
            raise ValueError("Training and development sources overlap")
        development = {
            case: Subset(evaluation, [i for i, c in enumerate(cases) if c == case])
            for case in sorted(set(cases))
        }
        trainer.args.prediction_loss_only = True
        trainer.save_metrics("baseline_dev", trainer.evaluate(eval_dataset=development))
    result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if development:
        trainer.save_metrics("candidate_dev", trainer.evaluate(eval_dataset=development))
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    declared_weights = {}
    for sample in dataset.samples:
        for span in sample["loss_spans"]:
            kind = span.get("kind", "unspecified")
            declared_weights.setdefault(kind, set()).add(float(span.get("weight", 1.0)))
    provenance = {
        "base_model": script_args.model_name_or_path,
        "stable_speaker_names": script_args.stable_speaker_names,
        "text_pad_multiple": script_args.text_pad_multiple,
        "base_model_revision": script_args.model_revision,
        "official_training_code_revision": OFFICIAL_CODE_REVISION,
        "official_runtime_code_revision": subprocess.run(
            [
                "git",
                "-C",
                str(Path(__import__("moss_transcribe_diarize").__file__).resolve().parent.parent),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or None,
        "training_script_sha256": hashlib.sha256(training_script).hexdigest(),
        "training_manifest_sha256": manifest_sha256,
        "train_jsonl": str(Path(script_args.train_jsonl).resolve()),
        "samples": len(dataset),
        "forbidden_sessions": sorted(forbidden),
        "mono_input_only_enforced": True,
        "token_loss_weights": {
            kind: next(iter(values)) if len(values) == 1 else sorted(values)
            for kind, values in declared_weights.items()
        },
        "activity_loss_weight": script_args.activity_loss_weight,
        "activity_max_speakers": script_args.max_activity_speakers,
        "activity_conditioning": "sigmoid multi-label head projected into audio embeddings",
        "activity_conditioning_version": 3,
        "overlap_head": "direct class-balanced binary frame classifier",
        "activity_head_only": script_args.activity_head_only,
        "use_activity_conditioning": script_args.use_activity_conditioning,
        "use_target_speaker_conditioning": script_args.use_target_speaker_conditioning,
        "target_conditioning_only": script_args.target_conditioning_only,
        "target_profile_seconds": script_args.target_profile_seconds,
        "target_gap_seconds": script_args.target_gap_seconds,
        "target_activity_loss_weight": script_args.target_activity_loss_weight,
        "promote_trainable_parameters_fp32": (script_args.promote_trainable_parameters_fp32),
        "optimizer_parameter_promotion": promotion,
    }
    Path(training_args.output_dir, "training_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(training_args.output_dir, "training_script_snapshot.py").write_bytes(training_script)


if __name__ == "__main__":
    main()
