from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

import transcriber.segment_classifier as segment_classifier
from transcriber.segment_classifier import (
    AudioAugmentationConfig,
    ClassifierDataset,
    SegmentClassifier,
    _build_dataset_cache_signature,
    _collect_clean_stem_segments,
    _discover_session_transcript,
    _find_session_sources,
    filter_classifier_dataset,
    balance_classifier_dataset,
    build_classifier_dataset_from_bank,
    load_labeled_records,
    load_classifier_dataset,
    materialize_classifier_dataset_from_mixed_base,
    merge_classifier_datasets,
    relabel_classifier_dataset_sources,
    save_classifier_dataset,
    train_segment_classifier_from_bank,
)


def test_segment_classifier_fit_save_load_and_predict(tmp_path):
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    labels = ["Alice", "Alice", "Bob", "Bob"]

    classifier = SegmentClassifier.fit(embeddings, labels)
    artifacts = classifier.save(tmp_path)

    assert Path(artifacts["model"]).exists()
    assert Path(artifacts["meta"]).exists()

    loaded = SegmentClassifier.load(tmp_path)
    assert loaded is not None

    prediction = loaded.predict(np.array([0.95, 0.05], dtype=np.float32), min_margin=0.01)
    assert prediction is not None
    assert prediction.speaker == "Alice"
    assert prediction.score > prediction.second_best


def test_segment_classifier_supports_knn_model(tmp_path):
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
        ],
        dtype=np.float32,
    )
    labels = ["Alice", "Alice", "Bob", "Bob"]

    classifier = SegmentClassifier.fit(embeddings, labels, model_name="knn", n_neighbors=1)
    artifacts = classifier.save(tmp_path)

    assert Path(artifacts["model"]).exists()

    loaded = SegmentClassifier.load(tmp_path)
    assert loaded is not None
    assert loaded.summary()["model_name"] == "knn"

    prediction = loaded.predict(np.array([0.98, 0.02], dtype=np.float32), min_margin=0.01)
    assert prediction is not None
    assert prediction.speaker == "Alice"


def test_discover_session_transcript_prefers_matching_session_stem(tmp_path):
    outputs_root = tmp_path / "outputs"
    session_dir = outputs_root / "Session_32"
    session_dir.mkdir(parents=True)
    session_jsonl = session_dir / "Session_32.jsonl"
    session_jsonl.write_text("{}", encoding="utf-8")

    discovered = _discover_session_transcript(tmp_path / "Session_32.zip", [outputs_root])

    assert discovered == session_jsonl


def test_discover_session_transcript_falls_back_to_fuzzy_match(tmp_path):
    outputs_root = tmp_path / "outputs"
    session_dir = outputs_root / "Session 13"
    session_dir.mkdir(parents=True)
    wrong_txt = session_dir / "Session 9.txt"
    wrong_txt.write_text("ignored", encoding="utf-8")
    transcript = session_dir / "Session 13 Transcript.txt"
    transcript.write_text("match", encoding="utf-8")

    discovered = _discover_session_transcript(tmp_path / "Session 13.zip", [outputs_root])

    assert discovered == transcript


def test_discover_session_transcript_rejects_unrelated_fuzzy_match(tmp_path):
    outputs_root = tmp_path / "outputs"
    outputs_root.mkdir(parents=True)
    wrong_txt = outputs_root / "Session 1.txt"
    wrong_txt.write_text("ignored", encoding="utf-8")

    discovered = _discover_session_transcript(tmp_path / "Session_26.zip", [outputs_root])

    assert discovered is None


def test_discover_session_transcript_rejects_non_session_source_names(tmp_path):
    outputs_root = tmp_path / "outputs"
    outputs_root.mkdir(parents=True)
    wrong_txt = outputs_root / "Session 1.txt"
    wrong_txt.write_text("ignored", encoding="utf-8")

    discovered = _discover_session_transcript(tmp_path / "ALgqyzm1qUOD_data", [outputs_root])

    assert discovered is None


def test_discover_session_transcript_skips_path_exists_on_unstable_mount(tmp_path, monkeypatch):
    outputs_root = tmp_path / "outputs"
    session_dir = outputs_root / "Session 58"
    session_dir.mkdir(parents=True)
    transcript = session_dir / "Session 58.txt"
    transcript.write_text("match", encoding="utf-8")

    def raise_no_device(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise OSError(19, "No such device")

    monkeypatch.setattr(Path, "exists", raise_no_device)

    discovered = _discover_session_transcript(tmp_path / "Session_58.zip", [outputs_root])

    assert discovered == transcript


def test_load_labeled_records_parses_timed_transcript_and_aliases(tmp_path):
    transcript = tmp_path / "Session 6.txt"
    transcript.write_text(
        "\n".join(
            [
                "Zariel Torgan 00:00:07 Oh, shit.",
                "Dungeon Master 00:00:10 Continue.",
            ]
        ),
        encoding="utf-8",
    )

    records = load_labeled_records(transcript)

    assert len(records) == 2
    assert records[0]["speaker"] == "David Tanglethorn"
    assert records[0]["start"] == 7
    assert records[0]["end"] == 10


def test_load_labeled_records_maps_raw_track_ids_from_jsonl(tmp_path):
    transcript = tmp_path / "Session_32.jsonl"
    transcript.write_text(
        '{"speaker":"1-traceritops_0","file":"1-traceritops_0.ogg","start":0.0,"end":1.0,"text":"hello"}\n',
        encoding="utf-8",
    )

    records = load_labeled_records(
        transcript,
        speaker_mapping={"traceritops_0": "Cletus Cobbington"},
    )

    assert len(records) == 1
    assert records[0]["speaker"] == "Cletus Cobbington"


def test_segment_classifier_supports_unbalanced_logreg_model():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
        ],
        dtype=np.float32,
    )
    labels = ["Alice", "Alice", "Bob", "Bob"]

    classifier = SegmentClassifier.fit(embeddings, labels, model_name="logreg_unbalanced")
    prediction = classifier.predict(np.array([0.98, 0.02], dtype=np.float32), min_margin=0.01)

    assert prediction is not None
    assert prediction.speaker == "Alice"


def test_segment_classifier_supports_torch_linear_model(tmp_path):
    pytest.importorskip("torch")
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
        ],
        dtype=np.float32,
    )
    labels = ["Alice", "Alice", "Bob", "Bob"]

    classifier = SegmentClassifier.fit(embeddings, labels, model_name="torch_linear", max_iter=200)
    artifacts = classifier.save(tmp_path)

    assert Path(artifacts["model"]).exists()
    loaded = SegmentClassifier.load(tmp_path)
    assert loaded is not None
    assert loaded.summary()["model_name"] == "torch_linear"

    prediction = loaded.predict(np.array([0.98, 0.02], dtype=np.float32), min_margin=0.01)
    assert prediction is not None
    assert prediction.speaker == "Alice"


def test_segment_classifier_supports_lda_knn_model(tmp_path):
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.05, 0.95, 0.0],
            [0.0, 0.0, 1.0],
            [0.05, 0.0, 0.95],
        ],
        dtype=np.float32,
    )
    labels = ["Alice", "Alice", "Bob", "Bob", "Carol", "Carol"]

    classifier = SegmentClassifier.fit(embeddings, labels, model_name="lda_knn", n_neighbors=1)
    artifacts = classifier.save(tmp_path)

    assert Path(artifacts["model"]).exists()
    loaded = SegmentClassifier.load(tmp_path)
    assert loaded is not None
    assert loaded.summary()["model_name"] == "lda_knn"

    prediction = loaded.predict(np.array([0.98, 0.02, 0.0], dtype=np.float32), min_margin=0.01)
    assert prediction is not None
    assert prediction.speaker == "Alice"


def test_segment_classifier_supports_lda_logreg_model():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.05, 0.95, 0.0],
            [0.0, 0.0, 1.0],
            [0.05, 0.0, 0.95],
        ],
        dtype=np.float32,
    )
    labels = ["Alice", "Alice", "Bob", "Bob", "Carol", "Carol"]

    classifier = SegmentClassifier.fit(embeddings, labels, model_name="lda_logreg")
    prediction = classifier.predict(np.array([0.0, 0.0, 0.98], dtype=np.float32), min_margin=0.01)

    assert prediction is not None
    assert prediction.speaker == "Carol"


def test_load_labeled_records_parses_timestamp_first_transcript_and_aliases(tmp_path):
    transcript = tmp_path / "Session 1.txt"
    transcript.write_text(
        "\n".join(
            [
                "0:00:01: DM: Hello there.",
                "0:00:04: Kaladin Shash: Testing.",
            ]
        ),
        encoding="utf-8",
    )

    records = load_labeled_records(transcript)

    assert len(records) == 2
    assert records[0]["speaker"] == "Dungeon Master"
    assert records[0]["start"] == 1
    assert records[0]["end"] == 4
    assert records[1]["speaker"] == "Kaladen Shash"


def test_collect_clean_stem_segments_samples_evenly_and_caps_duration(tmp_path):
    stem_a = tmp_path / "alice.wav"
    stem_b = tmp_path / "bob.wav"
    records = [
        {"speaker": "Alice", "start": 0.0, "end": 30.0, "text": "one"},
        {"speaker": "Alice", "start": 40.0, "end": 44.0, "text": "two"},
        {"speaker": "Alice", "start": 50.0, "end": 55.0, "text": "three"},
        {"speaker": "Bob", "start": 10.0, "end": 10.4, "text": "too short"},
        {"speaker": "Bob", "start": 12.0, "end": 18.0, "text": "valid"},
    ]

    segments = _collect_clean_stem_segments(
        records=records,
        stem_paths_by_speaker={"Alice": stem_a, "Bob": stem_b},
        min_segment_dur=1.0,
        max_segment_dur=8.0,
        max_records_per_speaker=2,
    )

    assert list(segments[stem_a]) == [
        (0.0, 8.0, "Alice"),
        (50.0, 55.0, "Alice"),
    ]
    assert list(segments[stem_b]) == [(12.0, 18.0, "Bob")]


def test_find_session_sources_accepts_extracted_session_directories(tmp_path):
    extracted_root = tmp_path / "extracted"
    session_dir = extracted_root / "Session_58"
    session_dir.mkdir(parents=True)
    (session_dir / "speaker1.ogg").write_bytes(b"audio")
    (extracted_root / "empty").mkdir()

    sources = _find_session_sources(extracted_root)

    assert sources == [session_dir]


def test_find_session_sources_skips_non_session_audio_directories(tmp_path):
    extracted_root = tmp_path / "extracted"
    session_dir = extracted_root / "Session_58"
    session_dir.mkdir(parents=True)
    (session_dir / "speaker1.ogg").write_bytes(b"audio")
    junk_dir = extracted_root / "ALgqyzm1qUOD_data"
    junk_dir.mkdir(parents=True)
    (junk_dir / "speaker2.ogg").write_bytes(b"audio")

    sources = _find_session_sources(extracted_root)

    assert sources == [session_dir]


def test_dataset_cache_signature_includes_transcript_roots(tmp_path):
    session_dir = tmp_path / "Session_58"
    session_dir.mkdir()
    transcripts_a = tmp_path / "transcripts_a"
    transcripts_b = tmp_path / "transcripts_b"
    transcripts_a.mkdir()
    transcripts_b.mkdir()

    signature_a = _build_dataset_cache_signature(
        session_sources=[session_dir],
        transcript_search_roots=[transcripts_a],
        training_mode="mixed",
        top_k=3,
        hop_seconds=120.0,
        min_speakers=4,
        min_share=0.72,
        min_power=2e-4,
        min_segment_dur=0.6,
        max_segment_dur=20.0,
        max_samples_per_speaker=700,
        window_seconds=300.0,
        clean_max_records_per_speaker_per_session=80,
        clean_window_size=0.0,
        clean_window_stride=0.0,
        allowed_speakers=["Alice"],
        excluded_speakers=[],
        augmentation=AudioAugmentationConfig(),
        diarization_model_name="pyannote/speaker-diarization-community-1",
        include_base_samples=True,
    )
    signature_b = _build_dataset_cache_signature(
        session_sources=[session_dir],
        transcript_search_roots=[transcripts_b],
        training_mode="mixed",
        top_k=3,
        hop_seconds=120.0,
        min_speakers=4,
        min_share=0.72,
        min_power=2e-4,
        min_segment_dur=0.6,
        max_segment_dur=20.0,
        max_samples_per_speaker=700,
        window_seconds=300.0,
        clean_max_records_per_speaker_per_session=80,
        clean_window_size=0.0,
        clean_window_stride=0.0,
        allowed_speakers=["Alice"],
        excluded_speakers=[],
        augmentation=AudioAugmentationConfig(),
        diarization_model_name="pyannote/speaker-diarization-community-1",
        include_base_samples=True,
    )

    assert signature_a != signature_b


def test_filter_classifier_dataset_applies_purity_and_duration_filters():
    dataset = ClassifierDataset(
        embeddings=np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        labels=["Alice", "Alice", "Bob"],
        domains=["mixed", "mixed", "mixed"],
        sources=["mixed_raw", "mixed_raw", "mixed_raw"],
        sessions=["s1", "s1", "s2"],
        durations=np.asarray([1.0, 3.0, 2.0], dtype=np.float32),
        dominant_shares=np.asarray([0.92, 0.91, 0.87], dtype=np.float32),
        top1_powers=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        top2_powers=np.asarray([0.1, 0.2, 0.1], dtype=np.float32),
        active_speakers=np.asarray([1, 3, 2], dtype=np.int32),
    )

    filtered, summary = filter_classifier_dataset(
        dataset,
        allowed_speakers=["Alice", "Bob"],
        allowed_sources=["mixed_raw"],
        min_dominant_share=0.85,
        max_active_speakers=2,
        min_duration=0.8,
        max_duration=2.5,
    )

    assert filtered.labels == ["Alice", "Bob"]
    assert filtered.samples == 2
    assert summary["rejections"] == {"active_speakers": 1}


def test_train_segment_classifier_from_bank_uses_saved_embeddings(tmp_path):
    profile_dir = tmp_path / "speaker_bank"
    profile_dir.mkdir()
    (profile_dir / "bank.json").write_text(
        '{"entries": ['
        '{"speaker": "Alice", "source": "alice_1.ogg"},'
        '{"speaker": "Alice", "source": "alice_2.ogg"},'
        '{"speaker": "Bob", "source": "bob_1.ogg"},'
        '{"speaker": "Bob", "source": "bob_2.ogg"}'
        "]}",
        encoding="utf-8",
    )
    np.save(
        profile_dir / "embeddings.npy",
        np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        ),
    )

    summary = train_segment_classifier_from_bank(
        profile_dir=profile_dir,
        model_name="knn",
        classifier_n_neighbors=1,
    )

    assert summary is not None
    assert summary["source"] == "speaker_bank"
    assert summary["training_samples"] == 4
    assert summary["speakers"] == {"Alice": 2, "Bob": 2}
    loaded = SegmentClassifier.load(profile_dir)
    assert loaded is not None
    prediction = loaded.predict(np.array([0.98, 0.02], dtype=np.float32), min_margin=0.01)
    assert prediction is not None
    assert prediction.speaker == "Alice"


def test_save_and_load_classifier_dataset_round_trip(tmp_path):
    dataset = ClassifierDataset(
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        labels=["Alice", "Bob"],
        domains=["mixed", "mixed_aug"],
        sources=["mixed_raw", "mixed_aug_total"],
        sessions=["Session 1", "Session 2"],
        durations=np.array([1.2, 2.4], dtype=np.float32),
        dominant_shares=np.array([0.91, 0.72], dtype=np.float32),
        top1_powers=np.array([0.4, 0.3], dtype=np.float32),
        top2_powers=np.array([0.1, 0.08], dtype=np.float32),
        active_speakers=np.array([1, 2], dtype=np.int32),
    )

    artifacts = save_classifier_dataset(tmp_path, dataset, summary={"name": "roundtrip"})
    loaded, summary = load_classifier_dataset(tmp_path)

    assert Path(artifacts["dataset"]).exists()
    assert Path(artifacts["summary"]).exists()
    assert summary["name"] == "roundtrip"
    assert loaded.labels == dataset.labels
    assert loaded.sources == dataset.sources
    assert np.allclose(loaded.embeddings, dataset.embeddings)


def test_balance_classifier_dataset_caps_augmented_source_to_raw_counts():
    bank = ClassifierDataset(
        embeddings=np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        ),
        labels=["Alice", "Alice", "Bob", "Bob"],
        domains=["bank"] * 4,
        sources=["bank"] * 4,
        sessions=["bank"] * 4,
        durations=np.full(4, np.nan, dtype=np.float32),
        dominant_shares=np.full(4, np.nan, dtype=np.float32),
        top1_powers=np.full(4, np.nan, dtype=np.float32),
        top2_powers=np.full(4, np.nan, dtype=np.float32),
        active_speakers=np.full(4, -1, dtype=np.int32),
    )
    raw = ClassifierDataset(
        embeddings=np.array([[1.0, 0.1], [0.1, 1.0]], dtype=np.float32),
        labels=["Alice", "Bob"],
        domains=["mixed", "mixed"],
        sources=["mixed_raw", "mixed_raw"],
        sessions=["Session 1", "Session 1"],
        durations=np.array([1.0, 1.0], dtype=np.float32),
        dominant_shares=np.array([0.9, 0.9], dtype=np.float32),
        top1_powers=np.array([0.3, 0.3], dtype=np.float32),
        top2_powers=np.array([0.05, 0.05], dtype=np.float32),
        active_speakers=np.array([1, 1], dtype=np.int32),
    )
    aug = ClassifierDataset(
        embeddings=np.array(
            [
                [1.0, 0.2],
                [1.0, 0.3],
                [0.2, 1.0],
                [0.3, 1.0],
            ],
            dtype=np.float32,
        ),
        labels=["Alice", "Alice", "Bob", "Bob"],
        domains=["mixed_aug"] * 4,
        sources=["light_x1"] * 4,
        sessions=["Session 1"] * 4,
        durations=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        dominant_shares=np.array([0.75, 0.76, 0.77, 0.78], dtype=np.float32),
        top1_powers=np.array([0.3, 0.3, 0.3, 0.3], dtype=np.float32),
        top2_powers=np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float32),
        active_speakers=np.array([2, 2, 2, 2], dtype=np.int32),
    )

    merged = merge_classifier_datasets(
        [
            relabel_classifier_dataset_sources(bank, "bank"),
            raw,
            relabel_classifier_dataset_sources(aug, "mixed_aug_total"),
        ]
    )
    balanced, summary = balance_classifier_dataset(
        merged,
        target_speakers=["Alice", "Bob"],
        max_samples_per_cell=10,
    )

    assert summary["source_groups"]["mixed_aug_total"]["selected"] == {"Alice": 1, "Bob": 1}
    assert balanced.sources.count("mixed_aug_total") == 2


def test_build_classifier_dataset_from_bank_returns_dataset(tmp_path):
    profile_dir = tmp_path / "speaker_bank"
    profile_dir.mkdir()
    (profile_dir / "bank.json").write_text(
        '{"entries": ['
        '{"speaker": "Alice", "source": "alice_1.ogg"},'
        '{"speaker": "Bob", "source": "bob_1.ogg"}'
        "]}",
        encoding="utf-8",
    )
    np.save(
        profile_dir / "embeddings.npy",
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    built = build_classifier_dataset_from_bank(profile_dir=profile_dir)

    assert built is not None
    dataset, summary = built
    assert dataset.samples == 2
    assert summary["source"] == "speaker_bank"


def test_prepare_extracted_session_source_reuses_cached_extraction(tmp_path):
    session_zip = tmp_path / "Session 10.zip"
    with ZipFile(session_zip, "w") as archive:
        archive.writestr("alice.ogg", b"audio")

    cache_root = tmp_path / "cache"
    events: list[dict] = []

    first = segment_classifier._prepare_extracted_session_source(
        session_zip,
        cache_root=cache_root,
        progress_callback=lambda **kwargs: events.append(dict(kwargs)),
    )
    second = segment_classifier._prepare_extracted_session_source(
        session_zip,
        cache_root=cache_root,
        progress_callback=lambda **kwargs: events.append(dict(kwargs)),
    )

    assert first == second
    assert (first / "alice.ogg").exists()
    assert [event["status"] for event in events] == [
        "extraction_cache_miss",
        "extraction_cache_hit",
    ]


def test_materialize_classifier_dataset_from_mixed_base_reuses_cached_variant(
    tmp_path, monkeypatch
):
    mixed_base_dir = tmp_path / "mixed_base"
    mixed_base_dir.mkdir()
    (mixed_base_dir / "mixed.wav").write_bytes(b"fake")
    (mixed_base_dir / "prepared_windows.jsonl").write_text(
        json.dumps(
            {
                "session": "Session_10",
                "mixed_path": str(mixed_base_dir / "mixed.wav"),
                "accepted_segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "duration": 1.0,
                        "speaker": "Alice",
                        "raw_label": "SPEAKER_00",
                        "dominant_share": 0.9,
                        "top1_power": 0.3,
                        "top2_power": 0.05,
                        "active_speakers": 1,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (mixed_base_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "artifact_id": "mixed-base-artifact",
                "quality_filters": {
                    "clipping_fraction_max": 0.005,
                    "silence_fraction_max": 0.80,
                },
                "source_groups": {},
            }
        ),
        encoding="utf-8",
    )

    calls = {"extract": 0}

    monkeypatch.setattr(
        segment_classifier,
        "_load_audio_array",
        lambda path: (np.zeros(16000, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(segment_classifier, "build_waveform_augmenter", lambda *args, **kwargs: object())

    import transcriber.diarization as diarization

    def fake_extract_embeddings_for_segments(_audio_path, segments, **kwargs):
        calls["extract"] += 1
        return (
            [
                SimpleNamespace(index=index, embedding=np.asarray([1.0, 0.0], dtype=np.float32))
                for index, _segment in enumerate(segments)
            ],
            {},
        )

    monkeypatch.setattr(diarization, "extract_embeddings_for_segments", fake_extract_embeddings_for_segments)

    output_dir = tmp_path / "light_variant"
    first_dataset, first_summary = materialize_classifier_dataset_from_mixed_base(
        mixed_base_dir=mixed_base_dir,
        dataset_cache_dir=output_dir,
        hf_token=None,
        force_device="cpu",
        quiet=True,
        batch_size=4,
        workers=1,
        augmentation_profile="light",
        augmentation_copies=1,
        include_base_samples=False,
        reuse_cached_dataset=True,
    )
    second_dataset, second_summary = materialize_classifier_dataset_from_mixed_base(
        mixed_base_dir=mixed_base_dir,
        dataset_cache_dir=output_dir,
        hf_token=None,
        force_device="cpu",
        quiet=True,
        batch_size=4,
        workers=1,
        augmentation_profile="light",
        augmentation_copies=1,
        include_base_samples=False,
        reuse_cached_dataset=True,
    )

    assert first_dataset.samples == 1
    assert first_dataset.domains == ["mixed_aug"]
    assert first_dataset.sources == ["mixed_aug"]
    assert second_dataset.samples == 1
    assert calls["extract"] == 1
    assert first_summary["materialization_mode"] == "mixed_base_derived"
    assert second_summary["materialization_signature"] == first_summary["materialization_signature"]
