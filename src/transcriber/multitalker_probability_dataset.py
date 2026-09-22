from __future__ import annotations

import random
import re
from pathlib import Path
import numpy as np


SLOT_INDEX_RE = re.compile(r"(?:speaker|spk|slot)_?(\d+)$")


def speaker_slot_index(value: object) -> int:
    match = SLOT_INDEX_RE.search(str(value))
    if not match:
        raise ValueError(f"Cannot parse Sortformer slot index from {value!r}")
    return int(match.group(1))


def soft_target_masks(
    probabilities: np.ndarray,
    *,
    target_slot: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"Expected [frames, speakers] probabilities, got {values.shape}")
    if target_slot < 0 or target_slot >= values.shape[1]:
        raise ValueError(f"Target slot {target_slot} is outside probability shape {values.shape}")
    values = np.clip(values, 0.0, 1.0)
    target = values[:, target_slot]
    other_slots = [index for index in range(values.shape[1]) if index != target_slot]
    if other_slots:
        background = 1.0 - np.prod(1.0 - values[:, other_slots], axis=1)
    else:
        background = np.zeros(values.shape[0], dtype=np.float32)
    return target.astype(np.float32), background.astype(np.float32)


def build_probability_dataset_class():
    import torch
    from lhotse.dataset.collation import collate_vectors
    from nemo.collections.asr.data.audio_to_text_lhotse_speaker import (
        LhotseSpeechToTextSpkBpeDataset,
        speaker_to_target,
    )

    class LhotseSpeechToTextSpkProbabilityDataset(LhotseSpeechToTextSpkBpeDataset):
        """NeMo multitalker dataset supporting raw mono-Sortformer activity probabilities."""

        def __getitem__(self, cuts):
            audio, audio_lens, cuts = self.load_audio(cuts)
            if self.inference_mode:
                return audio, audio_lens, None, None, None, None

            tokens = []
            speaker_targets = []
            background_targets = []
            for cut in cuts:
                probability_path = cut.custom.get("sortformer_probabilities_path")
                slot_mapping = cut.custom.get("sortformer_slot_to_speaker")
                if probability_path and slot_mapping:
                    mapping = dict(slot_mapping)
                    target_slot_name = random.choice(sorted(mapping))
                    target_speaker = str(mapping[target_slot_name])
                    text = " ".join(
                        supervision.text or ""
                        for supervision in sorted(
                            cut.supervisions,
                            key=lambda item: (item.start, item.end),
                        )
                        if str(supervision.speaker) == target_speaker
                    ).strip()
                    probabilities = np.load(Path(str(probability_path)), allow_pickle=False)
                    target, background = soft_target_masks(
                        probabilities,
                        target_slot=speaker_slot_index(target_slot_name),
                    )
                    speaker_target = torch.from_numpy(target)
                    background_target = torch.from_numpy(background)
                else:
                    hard_targets, texts = speaker_to_target(
                        a_cut=cut,
                        num_speakers=self.num_speakers,
                        num_sample_per_mel_frame=self.num_sample_per_mel_frame,
                        num_mel_frame_per_asr_frame=self.num_mel_frame_per_asr_frame,
                        return_text=True,
                    )
                    hard_targets = hard_targets.transpose(0, 1)[: len(texts)]
                    target_id = random.choice(range(len(texts)))
                    other_ids = [index for index in range(len(texts)) if index != target_id]
                    text = texts[target_id]
                    speaker_target = hard_targets[target_id]
                    background_target = hard_targets[other_ids].sum(dim=0) > 0

                language = cut.supervisions[0].language if cut.supervisions else "en"
                tokens.append(torch.as_tensor(self.tokenizer(text, language)))
                speaker_targets.append(speaker_target)
                background_targets.append(background_target)

            token_lens = torch.tensor([item.size(0) for item in tokens], dtype=torch.long)
            return (
                audio,
                audio_lens,
                collate_vectors(tokens, padding_value=0),
                token_lens,
                collate_vectors(speaker_targets, padding_value=0),
                collate_vectors(background_targets, padding_value=0),
            )

    return LhotseSpeechToTextSpkProbabilityDataset
