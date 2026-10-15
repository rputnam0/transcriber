import json
from transcriber.review_player import build_review


def test_review_preserves_audio_identity_offsets_and_escapes_script_data(tmp_path, monkeypatch):
    source = tmp_path / "player.wav"
    source.write_bytes(b"audio")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"files": [str(source)]}))
    response = tmp_path / "response.json"
    response.write_text(
        json.dumps(
            {
                "files": [
                    [
                        {
                            "start": 90,
                            "end": 91,
                            "text": "</script><script>bad</script>",
                            "words": [{"word": "Example", "start": 90.1, "end": 90.5}],
                        }
                    ]
                ],
                "activity": [
                    {
                        "audio_seconds": 100,
                        "retained_seconds": 1,
                        "regions": [{"start": 90, "end": 91}],
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(
        "transcriber.review_player.subprocess.run",
        lambda command, **kw: __import__("pathlib").Path(command[-1]).write_bytes(b"aac"),
    )
    result = build_review([["Session", str(request), str(response)]], tmp_path / "review")
    assert "</script><script>bad" not in result.read_text()
    manifest = json.loads(result.with_name("manifest.json").read_text())
    track = manifest["tracks"][0]
    assert track["segments"][0]["words"][0]["start"] == 90.1
    assert (result.parent / track["audio"]).exists()
    assert len(manifest["dataset_id"]) == 64


def test_review_flags_survive_transcript_export(tmp_path):
    from transcriber.consolidate import save_outputs

    segment = dict(
        start=90,
        end=91,
        speaker="Player",
        text="[Transcription needs review]",
        words=[],
        review_required=True,
        review_reason="Output limit",
    )
    save_outputs("sample", str(tmp_path), [("player.wav", [segment])], [], {}, write_srt_file=False)
    record = json.loads((tmp_path / "sample.jsonl").read_text())
    assert record["review_required"] is True
    assert record["review_reason"] == "Output limit"


def test_carry_feedback_uses_audio_hash_not_changed_segment_indexes(tmp_path, monkeypatch):
    source = tmp_path / "player.wav"
    source.write_bytes(b"audio")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"files": [str(source)]}))
    response = tmp_path / "response.json"
    response.write_text(
        json.dumps(
            {
                "files": [[dict(start=10, end=11, text="New", words=[])]],
                "activity": [
                    dict(audio_seconds=20, retained_seconds=1, regions=[dict(start=10, end=11)])
                ],
            }
        )
    )
    monkeypatch.setattr(
        "transcriber.review_player.subprocess.run",
        lambda command, **kw: __import__("pathlib").Path(command[-1]).write_bytes(b"aac"),
    )
    review = build_review([["Session", str(request), str(response)]], tmp_path / "review")
    manifest = json.loads(review.with_name("manifest.json").read_text())
    prior = tmp_path / "feedback.json"
    prior.write_text(
        json.dumps(
            dict(
                dataset_id="old",
                tracks=manifest["tracks"],
                feedback=[
                    dict(
                        id="0-0:segment:99",
                        track_id="0-0",
                        segment_index=99,
                        start=10,
                        end=11,
                        original_text="Old",
                        corrected_text="Corrected",
                        issue="wording",
                        notes="",
                    )
                ],
            )
        )
    )
    build_review(
        [["Session", str(request), str(response)]], tmp_path / "review", feedback_path=prior
    )
    updated = json.loads(review.with_name("manifest.json").read_text())
    row = updated["seed_feedback"][0]
    assert row["segment_index"] is None
    assert row["corrected_text"] == "Corrected" and row["original_text"] == "Old"
    assert row["source_dataset_id"] == "old"
    assert updated["dataset_id"] == manifest["dataset_id"]
