import pytest

from react_baby import paths


def test_resolve_video_path_rejects_outside_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("REACT_BABY_VIDEO_DIR", str(tmp_path / "videos"))
    (tmp_path / "videos").mkdir()
    inside = tmp_path / "videos" / "a.mp4"
    inside.write_bytes(b"")
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(b"")

    assert paths.resolve_video_path(str(inside)) == inside.resolve()
    with pytest.raises(ValueError):
        paths.resolve_video_path(str(outside))
    with pytest.raises(ValueError):
        paths.resolve_video_path(str(tmp_path / "videos" / ".." / "secret.mp4"))
    with pytest.raises(FileNotFoundError):
        paths.resolve_video_path(str(tmp_path / "videos" / "missing.mp4"))


def test_worker_count_env_override(monkeypatch):
    monkeypatch.setenv("REACT_BABY_WORKERS", "3")
    assert paths.get_worker_count() == 3
    monkeypatch.setenv("REACT_BABY_WORKERS", "0")
    assert paths.get_worker_count() == 1
