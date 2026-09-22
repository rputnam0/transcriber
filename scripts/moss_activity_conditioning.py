from __future__ import annotations

from pathlib import Path


def invariant_activity_probabilities(logits):
    import torch

    probabilities = torch.sigmoid(logits.float())
    no_speaker = torch.prod(1.0 - probabilities, dim=-1)
    exactly_one = torch.zeros_like(no_speaker)
    for speaker in range(probabilities.shape[-1]):
        others = [index for index in range(probabilities.shape[-1]) if index != speaker]
        no_other = (
            torch.prod(1.0 - probabilities[..., others], dim=-1)
            if others
            else torch.ones_like(no_speaker)
        )
        exactly_one = exactly_one + probabilities[..., speaker] * no_other
    return {
        "speech": (1.0 - no_speaker).clamp(0.0, 1.0),
        "overlap": (1.0 - no_speaker - exactly_one).clamp(0.0, 1.0),
    }


def serialized_activity_summary(adaptor) -> dict | None:
    import torch

    if adaptor is None or not adaptor.last_activity_logits:
        return None
    logits = torch.cat(
        [values.squeeze(0).detach().cpu() for values in adaptor.last_activity_logits],
        dim=0,
    )
    probabilities = invariant_activity_probabilities(logits)
    if hasattr(adaptor, "last_speech_logits") and adaptor.last_speech_logits:
        speech_logits = torch.cat(
            [values.squeeze(0).detach().cpu() for values in adaptor.last_speech_logits],
            dim=0,
        ).squeeze(-1)
        probabilities["speech"] = torch.sigmoid(speech_logits.float())
    if hasattr(adaptor, "last_overlap_logits") and adaptor.last_overlap_logits:
        overlap_logits = torch.cat(
            [values.squeeze(0).detach().cpu() for values in adaptor.last_overlap_logits],
            dim=0,
        ).squeeze(-1)
        probabilities["overlap"] = torch.sigmoid(overlap_logits.float())
    return {
        "activity_frame_hz": 12.5,
        "activity_speech_probabilities": [
            round(float(value), 6) for value in probabilities["speech"]
        ],
        "activity_overlap_probabilities": [
            round(float(value), 6) for value in probabilities["overlap"]
        ],
    }


def install_activity_conditioning(model, *, max_speakers: int, version: int | None = None):
    import torch
    from torch import nn

    current = model.model.vq_adaptor
    if hasattr(current, "activity_head"):
        return current

    resolved_version = int(
        version
        if version is not None
        else getattr(
            model.config,
            "activity_conditioning_version",
            1 if bool(getattr(model.config, "activity_conditioning", False)) else 2,
        )
    )

    class LegacyActivityConditionedAdaptor(nn.Module):
        def __init__(self, base, speaker_count: int):
            super().__init__()
            # Keep the official key path model.vq_adaptor.layers.* checkpoint-compatible.
            self.layers = base.layers
            hidden_size = int(base.layers[-1].normalized_shape[0])
            self.activity_head = nn.Linear(hidden_size, speaker_count)
            self.activity_projection = nn.Linear(speaker_count, hidden_size, bias=False)
            nn.init.zeros_(self.activity_projection.weight)
            self.last_activity_logits = []

        def clear_activity_logits(self) -> None:
            self.last_activity_logits.clear()

        def forward(self, features):
            adapted = self.layers(features)
            logits = self.activity_head(adapted.to(self.activity_head.weight.dtype))
            self.last_activity_logits.append(logits)
            probabilities = torch.sigmoid(logits).to(self.activity_projection.weight.dtype)
            conditioning = self.activity_projection(probabilities).to(adapted.dtype)
            return adapted + conditioning

    class DirectOverlapConditionedAdaptor(nn.Module):
        def __init__(self, base, speaker_count: int):
            super().__init__()
            self.layers = base.layers
            hidden_size = int(base.layers[-1].normalized_shape[0])
            self.activity_head = nn.Linear(hidden_size, speaker_count)
            self.overlap_head = nn.Linear(hidden_size, 1)
            self.activity_projection = nn.Linear(2, hidden_size, bias=False)
            nn.init.zeros_(self.activity_projection.weight)
            self.last_activity_logits = []
            self.last_overlap_logits = []

        def clear_activity_logits(self) -> None:
            self.last_activity_logits.clear()
            self.last_overlap_logits.clear()

        def forward(self, features):
            adapted = self.layers(features)
            activity_logits = self.activity_head(adapted.to(self.activity_head.weight.dtype))
            overlap_logits = self.overlap_head(adapted.to(self.overlap_head.weight.dtype))
            self.last_activity_logits.append(activity_logits)
            self.last_overlap_logits.append(overlap_logits)
            activity_probabilities = torch.sigmoid(activity_logits)
            speech_probability = 1.0 - torch.prod(
                1.0 - activity_probabilities,
                dim=-1,
                keepdim=True,
            )
            overlap_probability = torch.sigmoid(overlap_logits)
            invariant = torch.cat((speech_probability, overlap_probability), dim=-1)
            conditioning = self.activity_projection(
                invariant.to(self.activity_projection.weight.dtype)
            ).to(adapted.dtype)
            return adapted + conditioning

    class DirectSpeechOverlapConditionedAdaptor(nn.Module):
        def __init__(self, base, speaker_count: int):
            super().__init__()
            self.layers = base.layers
            hidden_size = int(base.layers[-1].normalized_shape[0])
            self.activity_head = nn.Linear(hidden_size, speaker_count)
            self.speech_head = nn.Linear(hidden_size, 1)
            self.overlap_head = nn.Linear(hidden_size, 1)
            self.activity_projection = nn.Linear(2, hidden_size, bias=False)
            nn.init.zeros_(self.activity_projection.weight)
            self.last_activity_logits = []
            self.last_speech_logits = []
            self.last_overlap_logits = []

        def clear_activity_logits(self) -> None:
            self.last_activity_logits.clear()
            self.last_speech_logits.clear()
            self.last_overlap_logits.clear()

        def forward(self, features):
            adapted = self.layers(features)
            activity_logits = self.activity_head(adapted.to(self.activity_head.weight.dtype))
            speech_logits = self.speech_head(adapted.to(self.speech_head.weight.dtype))
            overlap_logits = self.overlap_head(adapted.to(self.overlap_head.weight.dtype))
            self.last_activity_logits.append(activity_logits)
            self.last_speech_logits.append(speech_logits)
            self.last_overlap_logits.append(overlap_logits)
            invariant = torch.cat(
                (torch.sigmoid(speech_logits), torch.sigmoid(overlap_logits)),
                dim=-1,
            )
            conditioning = self.activity_projection(
                invariant.to(self.activity_projection.weight.dtype)
            ).to(adapted.dtype)
            return adapted + conditioning

    wrapper_class = (
        DirectSpeechOverlapConditionedAdaptor
        if resolved_version >= 3
        else (
            DirectOverlapConditionedAdaptor
            if resolved_version >= 2
            else LegacyActivityConditionedAdaptor
        )
    )
    wrapper = wrapper_class(current, max_speakers)
    wrapper.to(device=next(current.parameters()).device, dtype=next(current.parameters()).dtype)
    model.model.vq_adaptor = wrapper
    model.config.activity_conditioning = True
    model.config.activity_max_speakers = max_speakers
    model.config.activity_conditioning_version = resolved_version
    return wrapper


def load_activity_weights(model, checkpoint: Path) -> list[str]:
    from safetensors.torch import load_file

    wrapper = model.model.vq_adaptor
    if not hasattr(wrapper, "activity_head"):
        raise ValueError("Activity conditioning must be installed before loading weights")
    model_file = checkpoint / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    state = load_file(str(model_file), device="cpu")
    prefix = "model.vq_adaptor."
    activity_state = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix + "activity_")
        or key.startswith(prefix + "speech_")
        or key.startswith(prefix + "overlap_")
    }
    if not activity_state:
        raise ValueError(f"No activity-conditioning weights found in {model_file}")
    incompatible = wrapper.load_state_dict(activity_state, strict=False)
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if key.startswith("activity_") or key.startswith("speech_") or key.startswith("overlap_")
    ]
    if unexpected:
        raise ValueError(f"Unexpected activity keys: {unexpected}")
    return sorted(activity_state)
