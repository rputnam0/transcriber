from __future__ import annotations

from pathlib import Path


def install_target_speaker_conditioning(
    model,
    *,
    profile_seconds: float,
    gap_seconds: float,
    frame_hz: float = 12.5,
    attention_heads: int = 8,
):
    import torch
    from torch import nn

    current = model.model.vq_adaptor
    if hasattr(current, "target_activity_head"):
        return current

    class TargetSpeakerConditionedAdaptor(nn.Module):
        def __init__(self, base) -> None:
            super().__init__()
            self.layers = base.layers
            hidden_size = int(base.layers[-1].normalized_shape[0])
            if hidden_size % attention_heads:
                raise ValueError(
                    f"Hidden size {hidden_size} is not divisible by {attention_heads} heads"
                )
            self.profile_tokens = int(round(profile_seconds * frame_hz))
            self.context_tokens = int(round((profile_seconds + gap_seconds) * frame_hz))
            self.frame_hz = float(frame_hz)
            self.enrollment_cross_attention = nn.MultiheadAttention(
                hidden_size,
                attention_heads,
                batch_first=True,
            )
            self.target_cross_norm = nn.LayerNorm(hidden_size)
            self.target_activity_head = nn.Linear(hidden_size, 1)
            self.target_activity_projection = nn.Linear(1, hidden_size, bias=False)
            self.enrollment_marker = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.segment_embeddings = nn.Parameter(torch.zeros(2, hidden_size))
            self.cross_gate = nn.Parameter(torch.tensor(0.05))
            nn.init.zeros_(self.target_activity_projection.weight)
            nn.init.normal_(self.segment_embeddings, mean=0.0, std=0.01)
            self.last_target_activity_logits = []

        def clear_target_activity_logits(self) -> None:
            self.last_target_activity_logits.clear()

        def target_parameters(self):
            for name, parameter in self.named_parameters():
                if not name.startswith("layers."):
                    yield parameter

        def forward(self, features):
            adapted = self.layers(features)
            length = adapted.shape[1]
            profile_end = min(self.profile_tokens, length)
            mixture_start = min(self.context_tokens, length)
            if profile_end == 0 or mixture_start >= length:
                logits = adapted.new_zeros(adapted.shape[0], length, 1)
                self.last_target_activity_logits.append(logits)
                return adapted

            enrollment = adapted[:, :profile_end] + self.segment_embeddings[0]
            mixture = adapted[:, mixture_start:] + self.segment_embeddings[1]
            attended, _weights = self.enrollment_cross_attention(
                mixture,
                enrollment,
                enrollment,
                need_weights=False,
            )
            fused = self.target_cross_norm(mixture + torch.tanh(self.cross_gate) * attended)
            mixture_logits = self.target_activity_head(fused)
            conditioned = fused + self.target_activity_projection(torch.sigmoid(mixture_logits)).to(
                fused.dtype
            )

            output = adapted.clone()
            marker = self.enrollment_marker + self.segment_embeddings[0].view(1, 1, -1)
            output[:, :mixture_start] = marker.expand(
                adapted.shape[0], mixture_start, adapted.shape[-1]
            )
            output[:, mixture_start:] = conditioned
            logits = adapted.new_zeros(adapted.shape[0], length, 1)
            logits[:, mixture_start:] = mixture_logits
            self.last_target_activity_logits.append(logits)
            return output

    wrapper = TargetSpeakerConditionedAdaptor(current)
    wrapper.to(device=next(current.parameters()).device, dtype=next(current.parameters()).dtype)
    model.model.vq_adaptor = wrapper
    model.config.target_speaker_conditioning = True
    model.config.target_profile_seconds = float(profile_seconds)
    model.config.target_gap_seconds = float(gap_seconds)
    model.config.target_activity_frame_hz = float(frame_hz)
    model.config.target_attention_heads = int(attention_heads)
    return wrapper


def load_target_speaker_weights(model, checkpoint: Path) -> list[str]:
    from safetensors.torch import load_file

    wrapper = model.model.vq_adaptor
    if not hasattr(wrapper, "target_activity_head"):
        raise ValueError("Target-speaker conditioning must be installed before loading weights")
    model_file = checkpoint / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    state = load_file(str(model_file), device="cpu")
    prefix = "model.vq_adaptor."
    target_prefixes = (
        "enrollment_cross_attention.",
        "target_cross_norm.",
        "target_activity_head.",
        "target_activity_projection.",
        "enrollment_marker",
        "segment_embeddings",
        "cross_gate",
    )
    target_state = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix) and key.removeprefix(prefix).startswith(target_prefixes)
    }
    if not target_state:
        raise ValueError(f"No target-speaker weights found in {model_file}")
    incompatible = wrapper.load_state_dict(target_state, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if key.startswith(target_prefixes)]
    if unexpected:
        raise ValueError(f"Unexpected target-speaker keys: {unexpected}")
    return sorted(target_state)


def serialized_target_activity_summary(adaptor) -> dict | None:
    import torch

    if adaptor is None or not adaptor.last_target_activity_logits:
        return None
    logits = torch.cat(
        [values.squeeze(0).detach().cpu() for values in adaptor.last_target_activity_logits],
        dim=0,
    ).squeeze(-1)
    return {
        "target_activity_frame_hz": float(getattr(adaptor, "frame_hz", 12.5)),
        "target_activity_probabilities": [
            round(float(value), 6) for value in torch.sigmoid(logits.float())
        ],
    }
