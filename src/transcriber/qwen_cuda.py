"""CUDA adapters for the official Qwen3-ASR checkpoints, imported only by the Qwen worker.

They expose the two calls ``qwen_backend.transcribe_wave`` makes of the MLX models, and
build the same decoder prompt as mlx-audio, so both runtimes decode the same request.
"""

from __future__ import annotations

from importlib.metadata import version
from types import SimpleNamespace

RATE = 16000


def prompt(language: str, hotwords: list[str] | None) -> str:
    """Return mlx-audio 0.5.5's Qwen3-ASR prompt, before audio padding is expanded.

    mlx-audio joins hotwords with ", " into the system message and ends that message with a
    newline. The official chat template omits the newline, so the text is built here.
    """
    terms = ", ".join(t.strip() for t in hotwords or [] if t.strip())
    system = f"{terms}\n" if terms else ""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        f"<|im_start|>assistant\nlanguage {language}<asr_text>"
    )


class Recognizer:
    def __init__(self, model, processor):
        self.model, self.processor = model, processor
        tokenizer = processor.tokenizer
        self.stop_ids = {
            tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")
        } | {tokenizer.eos_token_id}

    def generate(self, clip, *, language, hotwords, max_tokens, temperature, verbose=False):
        import torch

        if temperature != 0.0:
            raise ValueError("The CUDA Qwen runtime decodes greedily only")
        inputs = self.processor(
            text=[prompt(language, hotwords)], audio=[clip], return_tensors="pt", padding=True
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        sequence = getattr(output, "sequences", output)[0, inputs["input_ids"].shape[1] :]
        tokens = []
        for token in sequence.tolist():
            if token in self.stop_ids:
                break
            tokens.append(token)
        text = self.processor.tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return SimpleNamespace(text=text, generation_tokens=len(tokens))


class Aligner:
    def __init__(self, aligner):
        self.aligner = aligner

    def generate(self, *, audio, text, language):
        [result] = self.aligner.align(audio=[(audio, RATE)], text=[text], language=[language])
        return SimpleNamespace(
            segments=[dict(text=i.text, start=i.start_time, end=i.end_time) for i in result]
        )


def load(model_path: str, aligner_path: str):
    import torch
    from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner

    if not torch.cuda.is_available():
        raise RuntimeError("The CUDA Qwen runtime needs a visible CUDA device")
    options = dict(dtype=torch.bfloat16, device_map="cuda:0")
    asr = Qwen3ASRModel.from_pretrained(model_path, **options)
    aligner = Qwen3ForcedAligner.from_pretrained(aligner_path, **options)
    versions = dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        qwen_asr=version("qwen-asr"),
        dtype="bfloat16",
    )
    return Recognizer(asr.model, asr.processor), Aligner(aligner), versions
