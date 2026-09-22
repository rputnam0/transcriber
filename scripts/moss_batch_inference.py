"""Batched greedy MOSS decoding; only used after comparison with single-clip decoding."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from types import SimpleNamespace


def decode_issues(text):
    """Catch observed runaway vocalizations and incomplete timestamped output."""
    issues = []
    if re.search(r"([A-Za-z])\1{39}", text):
        issues.append("runaway_character_repeat")
    if "[S" in text and not re.search(r"\[\d+(?:\.\d+)?\]$", text.strip()):
        issues.append("unfinished_turn")
    return issues


def repair_speaker_brackets(text):
    """Repair the observed closing-brace typo without altering words or identity IDs."""
    return re.sub(r"\[(S\d+)\}", r"[\1]", text)


def recover_compact_segments(text, parsed):
    """Recover bounded turns after malformed syntax without inventing an identity.

    Keep existing parsed turns; add independently bounded utterances it omitted.
    Untagged utterances get their own cluster for subsequent audio attribution.
    """
    result = list(parsed)
    recovered = []
    # An observed cough annotation can replace the missing identity token.
    # Preserve it as transcript text and attribute the voice from audio later.
    pattern = r"\[(\d+(?:\.\d+)?)\](?:\[(S\d+)\])?" r"((?:\[cough\])?[^\[]+)\[(\d+(?:\.\d+)?)\]"
    for start, speaker, words, end in re.findall(pattern, text):
        start, end, words = float(start), float(end), words.strip()
        if not words or end <= start:
            continue
        if any(p.start <= start and p.end >= end and words in p.text for p in result):
            continue
        item = dict(
            start=start, end=end, speaker=speaker or f"UNTAGGED_{len(recovered)}", text=words
        )
        recovered.append(item)
        result.append(SimpleNamespace(**item))
    return result, recovered


def generate_batch(
    model,
    processor,
    records,
    *,
    prompt_kwargs,
    device,
    max_new_tokens=2048,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
):
    import hashlib
    import soundfile as sf
    import torch
    from moss_transcribe_diarize.inference_utils import build_transcription_messages

    texts, waves = [], []
    for record in records:
        wave, rate = sf.read(record["audio"], dtype="float32")
        if rate != 16000 or wave.ndim != 1:
            raise ValueError("Batch inference requires prepared mono 16 kHz clips")
        if hashlib.sha256(wave.tobytes()).hexdigest() != record["sha256"]:
            raise ValueError("Audio differs from manifest")
        waves.append(wave)
        texts.append(
            processor.apply_chat_template(
                build_transcription_messages(Path(record["audio"]), **prompt_kwargs),
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    inputs = processor(text=texts, audio=waves, max_length=131072, return_tensors="pt")
    # The official processor right-pads training batches. Generation needs left padding,
    # including the final shorter clip; retain audio token/chunk ordering within each row.
    for index, mask in enumerate(inputs["attention_mask"]):
        pad = int((mask == 0).sum())
        if pad:
            inputs["input_ids"][index] = torch.roll(inputs["input_ids"][index], pad)
            inputs["attention_mask"][index] = torch.roll(mask, pad)
    inputs = inputs.to(device)
    config = copy.deepcopy(model.generation_config)
    config.max_new_tokens, config.do_sample = max_new_tokens, False
    config.repetition_penalty = repetition_penalty
    config.no_repeat_ngram_size = no_repeat_ngram_size
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=config)
    prefix_length = inputs["input_ids"].shape[1]
    texts = [
        processor.tokenizer.decode(row[prefix_length:], skip_special_tokens=True).strip()
        for row in output
    ]
    del inputs, output
    if device.type == "mps":
        torch.mps.empty_cache()
    return texts
