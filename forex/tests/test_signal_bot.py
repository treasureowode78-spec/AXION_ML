from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scripts import signal_bot


def test_resolve_model_path_prefers_existing_model(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    expected = model_dir / "crypto_signal_v1.joblib"
    expected.write_bytes(b"model")

    resolved = signal_bot.resolve_model_path(tmp_path, None)

    assert resolved == expected


def test_resolve_model_path_respects_explicit_path(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    explicit = model_dir / "custom.joblib"
    explicit.write_bytes(b"model")

    resolved = signal_bot.resolve_model_path(tmp_path, str(explicit))

    assert resolved == explicit
