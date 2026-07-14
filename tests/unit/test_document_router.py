from rag.document_router import extract_text


def test_plain_text(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello research world")
    assert "hello research world" in extract_text(str(f))


def test_unknown_format_returns_string(tmp_path):
    f = tmp_path / "file.xyz"
    f.write_text("some content")
    assert isinstance(extract_text(str(f)), str)
