"""Read and write audio files."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_audio(
    path: str | Path,
    *,
    start: float = 0.0,
    end: float | None = None,
    sample_rate: int | None = None,
) -> tuple[np.ndarray, int]:
    import librosa

    audio, rate = librosa.load(
        path,
        sr=sample_rate,
        mono=True,
        offset=start,
        duration=None if end is None else end - start,
    )
    return audio.astype(np.float32), rate


def write_audio(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Write audio to WAV or MP3, inferred from the file extension."""
    import soundfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), np.asarray(audio, dtype=np.float32), int(sample_rate))
    return path
