"""Render pitch runs as audible sine waves."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def sonify_pitch_run(data: pd.DataFrame, sample_rate: int = 44_100, amplitude: float = 0.1) -> np.ndarray:
    """Synthesize a sine wave following the pitch run's frequency track."""
    times = data["time"].to_numpy(dtype=float)
    frequencies = data["frequency_hz"].to_numpy(dtype=float)
    hop = np.median(np.diff(times))
    sample_times = np.arange(int((times[-1] - times[0] + hop) * sample_rate)) / sample_rate + times[0]
    interpolated = np.interp(sample_times, times, frequencies)
    phase = 2 * np.pi * np.cumsum(interpolated) / sample_rate
    audio = amplitude * np.sin(phase)
    audio[interpolated <= 0] = 0
    return audio.astype(np.float32)


def write_sonified_audio(
    data: pd.DataFrame,
    output_path: str | Path,
    sample_rate: int = 44_100,
    amplitude: float = 0.1,
) -> Path:
    """Sonify a pitch run and write it to MP3 or WAV (chosen by file extension)."""
    import soundfile

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = sonify_pitch_run(data, sample_rate=sample_rate, amplitude=amplitude)
    soundfile.write(str(output_path), audio, int(sample_rate))
    return output_path
