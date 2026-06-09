from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def load_audio(
    path: str | Path,
    *,
    start: float = 0.0,
    end: Optional[float] = None,
    sample_rate: Optional[int] = None,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load audio as float32 mono by default.

    `sample_rate=None` preserves the source sample rate.
    `start` and `end` are seconds in the original file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if start < 0:
        raise ValueError("start cannot be negative")
    if end is not None and end <= start:
        raise ValueError("end must be greater than start")

    duration = None if end is None else end - start

    try:
        import librosa

        audio, sr = librosa.load(
            str(path),
            sr=sample_rate,
            mono=mono,
            offset=float(start),
            duration=duration,
        )
        return np.ascontiguousarray(audio.astype(np.float32)), int(sr)
    except Exception as librosa_exc:
        # Fallback for WAV/FLAC files if librosa/audioread cannot decode.
        try:
            import soundfile as sf

            info = sf.info(str(path))
            sr = int(info.samplerate)
            start_frame = int(round(start * sr))
            frames = -1 if duration is None else int(round(duration * sr))
            audio, sr = sf.read(str(path), start=start_frame, frames=frames, dtype="float32", always_2d=False)
            if mono and getattr(audio, "ndim", 1) == 2:
                audio = audio.mean(axis=1)
            if sample_rate is not None and sample_rate != sr:
                import librosa

                audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=sample_rate)
                sr = sample_rate
            return np.ascontiguousarray(np.asarray(audio, dtype=np.float32)), int(sr)
        except Exception as sf_exc:
            raise RuntimeError(
                f"Could not load audio file {path}. librosa error: {librosa_exc!r}; soundfile error: {sf_exc!r}"
            ) from sf_exc


def write_temp_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), int(sample_rate))
    return path


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_f0_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
