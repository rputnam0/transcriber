import io
import pytest
from transcriber.review_server import ReviewHandler


@pytest.mark.parametrize(
    "header,expected,body",
    [
        ("bytes=2-5", 206, b"2345"),
        ("bytes=7-", 206, b"789"),
        ("bytes=-3", 206, b"789"),
        ("bytes=99-", 416, b""),
        ("bytes=-0", 416, b""),
        ("bytes=3-1", 416, b""),
    ],
)
def test_media_range_requests(tmp_path, header, expected, body):
    path = tmp_path / "audio.m4a"
    path.write_bytes(b"0123456789")
    handler = object.__new__(ReviewHandler)
    handler.headers = {"Range": header}
    handler.path = "/audio.m4a"
    handler.translate_path = lambda p: str(path)
    status = []
    headers = {}
    handler.send_response = status.append
    handler.send_header = lambda k, v: headers.update({k: v})
    handler.end_headers = lambda: None
    source = handler.send_head()
    out = io.BytesIO()
    if source:
        with source:
            handler.copyfile(source, out)
    assert status == [expected]
    assert out.getvalue() == body
    assert int(headers["Content-Length"]) == len(body)
