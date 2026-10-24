from types import SimpleNamespace
import json

import numpy as np
import pytest

from transcriber.qwen_backend import aligned_words, transcribe_wave
from transcriber.vocabulary import load_vocabulary


def test_campaign_vocabulary_preserves_canon_without_boosting_mishearings(tmp_path):
    path = tmp_path / "campaign.json"
    path.write_text(
        json.dumps(
            {
                "parties": [
                    {
                        "name": "Party",
                        "pcs": [
                            {"name": "Kaladen Shash", "aliases": ["Kaladen", "Kaladin", "Calden"]},
                            {"name": "Zariel Torgan", "aliases": ["Zariel", "Zario"]},
                        ],
                        "participants": [
                            {"role": "player", "character": {"name": "Zariel Torgan"}}
                        ],
                    }
                ],
                "terms": [{"name": "Lord Marshal Thurr", "aliases": ["Thurr", "Thor"]}],
                "table_people": [{"name": "Bren", "variants": ["Brent"]}],
            }
        )
    )
    bank = load_vocabulary(path)
    assert set(bank["terms"]) == {
        "Party",
        "Kaladen Shash",
        "Kaladen",
        "Zariel Torgan",
        "Zariel",
        "Lord Marshal Thurr",
        "Thurr",
        "Bren",
    }
    assert len(bank["terms"]) == len(set(bank["terms"]))
    path.write_text(json.dumps({"terms": [{"canonical": "Donnie", "spoken_forms": ["Rat Shit"]}]}))
    updated = load_vocabulary(path)
    assert updated["terms"] == ["Donnie", "Rat Shit"]
    assert updated["sha256"] != bank["sha256"]


def test_qwen_chunk_ownership_vocabulary_and_resume(tmp_path):
    # Word at 30.2 appears in both context windows but is published only once.
    wave = np.arange(40 * 16000, dtype=np.float32) + 1
    calls = []

    class Model:
        def generate(self, clip, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                text="Leopold!" if clip[0] == 1 else "Leopold! Zariel.", generation_tokens=5
            )

    class Aligner:
        def generate(self, *, audio, **kwargs):
            offset = (float(audio[0]) - 1) / 16000
            rows = [dict(text="Leopold", start=30.1 - offset, end=30.3 - offset)]
            if offset:
                rows.append(dict(text="Zariel", start=35 - offset, end=36 - offset))
            return SimpleNamespace(segments=rows)

    bank = {"terms": ["Leopold", "Zariel"]}
    result = transcribe_wave(
        wave, Model(), Aligner(), vocabulary=bank, cache=tmp_path, provenance={"test": 1}
    )
    assert [s["text"] for s in result] == ["Leopold!", "Zariel."]
    assert result[0]["start"] == pytest.approx(30.1)
    assert all(c["hotwords"] == bank["terms"] for c in calls)
    assert all(c["language"] == "English" for c in calls)
    assert (
        transcribe_wave(wave, None, None, vocabulary=bank, cache=tmp_path, provenance={"test": 1})
        == result
    )
    with pytest.raises(ValueError, match="provenance"):
        transcribe_wave(wave, None, None, vocabulary=bank, cache=tmp_path, provenance={"test": 2})


def test_qwen_retains_raw_truncation_and_rejects_changed_alignment(tmp_path):
    model = SimpleNamespace(
        generate=lambda *a, **k: SimpleNamespace(text="partial", generation_tokens=1024)
    )
    result = transcribe_wave(
        np.ones(16000), model, None, vocabulary={"terms": []}, cache=tmp_path, provenance={}
    )
    assert result[0]["review_required"]
    assert result[0]["text"] == "[Transcription needs review]"
    assert result[0]["words"] == []
    assert json.loads((tmp_path / "000000.raw.json").read_text())["text"] == "partial"
    assert json.loads((tmp_path / "000000.json").read_text())["review_reason"]
    with pytest.raises(ValueError, match="lexical"):
        aligned_words("Zariel", [dict(text="Zario", start=0, end=1)], 1)
    # Final timestamp can round beyond the clip on the aligner's 80 ms grid.
    assert aligned_words("valley", [dict(text="valley", start=24.72, end=25.04)], 25) == [
        dict(word="valley", start=24.72, end=25),
    ]
    with pytest.raises(ValueError, match="timestamps"):
        aligned_words("valley", [dict(text="valley", start=24.72, end=26)], 25)


def test_qwen_cli_keeps_overlapping_track_identity(monkeypatch, tmp_path):
    from transcriber import cli

    paths = [str(tmp_path / "one.wav"), str(tmp_path / "two.wav")]
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"one": "Player One", "two": "Player Two"}))
    bank = tmp_path / "names.json"
    bank.write_text('["Zariel"]')
    monkeypatch.setattr(cli, "gather_inputs", lambda _: (paths, None))
    monkeypatch.setattr(cli, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(cli, "_ensure_cuda_libs_on_path", lambda: None)
    monkeypatch.setattr(cli, "_preload_cudnn_libs", lambda: None)

    def decode(files, **kwargs):
        assert files == paths
        assert kwargs["vocabulary"]["terms"] == ["Zariel"]
        return [[dict(start=1, end=2, text=text, speaker=None)] for text in ["Hello", "Zariel"]]

    monkeypatch.setattr("transcriber.qwen_backend.transcribe_files", decode)
    out = tmp_path / "out"
    cli.run_transcribe(
        str(tmp_path / "tracks"),
        backend="auto",
        speaker_mapping_path=str(mapping),
        vocabulary_path=str(bank),
        output_dir=str(out),
        hf_cache_root=str(tmp_path / "hf"),
    )
    rows = [json.loads(line) for line in (out / "tracks" / "tracks.jsonl").read_text().splitlines()]
    assert [(r["speaker"], r["start"], r["text"]) for r in rows] == [
        ("Player One", 1, "Hello"),
        ("Player Two", 1, "Zariel"),
    ]
    marker = json.loads((out / "tracks" / "tracks.transcribe.json").read_text())
    assert marker["asr_backend"] == "qwen"
    assert marker["asr_model"] == "mlx-community/Qwen3-ASR-1.7B-bf16"


@pytest.mark.parametrize("prefix", ["", "The "])
def test_complete_hint_echo_is_retried_without_hints(tmp_path, prefix):
    bank = {"terms": ["Kaladen", "Leopold", "Cyrus", "Zariel"]}
    calls = []

    def generate(clip, **kwargs):
        calls.append(kwargs["hotwords"])
        return SimpleNamespace(
            text=prefix + ", ".join(bank["terms"]) + "." if kwargs["hotwords"] else "Okay.",
            generation_tokens=8,
        )

    aligner = SimpleNamespace(
        generate=lambda **kwargs: SimpleNamespace(segments=[dict(text="Okay", start=0.1, end=0.5)])
    )
    result = transcribe_wave(
        np.ones(16000),
        SimpleNamespace(generate=generate),
        aligner,
        vocabulary=bank,
        cache=tmp_path,
        provenance={},
    )
    assert [r["text"] for r in result] == ["Okay."]
    assert calls == [bank["terms"], None]
    assert "Kaladen" in json.loads((tmp_path / "000000.hint_echo.json").read_text())["text"]
    assert json.loads((tmp_path / "000000.json").read_text())["hint_echo_retry"]


@pytest.mark.parametrize(
    "device, runtime, model, revision",
    [
        (
            "auto",
            "mlx",
            "mlx-community/Qwen3-ASR-1.7B-bf16",
            "e1f6c266914abc5a46e8756e02580f834a6cf8a7",
        ),
        ("cuda", "cuda", "Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5"),
    ],
)
def test_qwen_completion_record_names_runtime(
    monkeypatch, tmp_path, device, runtime, model, revision
):
    from transcriber import cli

    paths = [str(tmp_path / "one.wav"), str(tmp_path / "two.wav")]
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"one": "Player One", "two": "Player Two"}))
    monkeypatch.setattr(cli, "gather_inputs", lambda _: (paths, None))
    monkeypatch.setattr(cli, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(cli, "_ensure_cuda_libs_on_path", lambda: None)
    monkeypatch.setattr(cli, "_preload_cudnn_libs", lambda: None)
    runtimes = []

    def decode(files, **kwargs):
        runtimes.append(kwargs["runtime"])
        return [[dict(start=1, end=2, text="Hello", speaker=None)] for _ in files]

    monkeypatch.setattr("transcriber.qwen_backend.transcribe_files", decode)
    out = tmp_path / "out"
    cli.run_transcribe(
        str(tmp_path / "tracks"),
        backend="qwen",
        model_name="qwen",
        device=device,
        speaker_mapping_path=str(mapping),
        output_dir=str(out),
        hf_cache_root=str(tmp_path / "hf"),
    )
    marker = json.loads((out / "tracks" / "tracks.transcribe.json").read_text())
    assert runtimes == [runtime]
    assert marker["status"] == "completed"
    assert (marker["asr_runtime"], marker["asr_model"], marker["asr_model_revision"]) == (
        runtime,
        model,
        revision,
    )


def test_cuda_qwen_is_explicit_and_unavailable_elsewhere(monkeypatch, tmp_path):
    from transcriber import cli

    paths = [str(tmp_path / "one.wav"), str(tmp_path / "two.wav")]
    monkeypatch.setattr(cli, "gather_inputs", lambda _: (paths, "/tmp/zip"))
    monkeypatch.setattr(cli, "_is_apple_silicon", lambda: False)
    monkeypatch.setattr(cli, "_ensure_cuda_libs_on_path", lambda: None)
    monkeypatch.setattr(cli, "cleanup_tmp", lambda _: None)
    assert (
        cli._resolve_backend_choice(
            "auto",
            files=paths,
            tmp_root="/tmp/zip",
            min_speakers=None,
            max_speakers=None,
            single_file_speaker=None,
            device="cuda",
            speaker_bank_config=None,
        )
        == "faster"
    )
    with pytest.raises(ValueError, match="--device cuda"):
        cli.run_transcribe(
            str(tmp_path / "s.zip"),
            backend="qwen",
            device="auto",
            output_dir=str(tmp_path / "out"),
            hf_cache_root=str(tmp_path / "hf"),
        )


def test_cuda_recognizer_matches_mlx_prompt_and_counts_budget():
    import torch

    from transcriber.qwen_cuda import Recognizer

    seen = {}

    class Tokenizer:
        eos_token_id, pad_token_id = 0, 2

        def convert_tokens_to_ids(self, token):
            return {"<|im_end|>": 1, "<|endoftext|>": 2}[token]

        def decode(self, tokens, **kwargs):
            return " ".join(f"w{t}" for t in tokens)

    class Inputs(dict):
        def to(self, _):
            return self

    class Processor:
        tokenizer = Tokenizer()

        def __call__(self, *, text, audio, **kwargs):
            seen["prompt"] = text[0]
            return Inputs(input_ids=torch.zeros((1, 3), dtype=torch.long))

    class Model:
        device, dtype = "cpu", torch.float32

        def generate(self, *, max_new_tokens, do_sample, pad_token_id, **inputs):
            new = list(range(10, 10 + max_new_tokens))
            return SimpleNamespace(sequences=torch.tensor([[0, 0, 0] + new[:5] + [1, 9]]))

    recognizer = Recognizer(Model(), Processor())
    result = recognizer.generate(
        np.zeros(16000, dtype=np.float32),
        language="English",
        hotwords=["Kaladen", "Zariel"],
        max_tokens=1024,
        temperature=0.0,
    )
    assert seen["prompt"] == (
        "<|im_start|>system\nKaladen, Zariel\n<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\nlanguage English<asr_text>"
    )
    assert (result.text, result.generation_tokens) == ("w10 w11 w12 w13 w14", 5)
    # Without an end token the whole budget was used, which triggers the review path.
    assert (
        recognizer.generate(
            np.zeros(16000, dtype=np.float32),
            language="English",
            hotwords=None,
            max_tokens=3,
            temperature=0.0,
        ).generation_tokens
        == 3
    )
    assert "system\n<|im_end|>" in seen["prompt"]
