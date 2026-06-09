from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def sonify_f0_dataframe(
    df: pd.DataFrame,
    *,
    sample_rate: int = 44100,
    amplitude: float = 0.1,
    frequency_column: str = "frequency_hz",
) -> np.ndarray:
    if "time" not in df.columns:
        raise ValueError("DataFrame must contain a 'time' column")
    if frequency_column not in df.columns:
        raise ValueError(f"DataFrame must contain {frequency_column!r}")

    times = df["time"].to_numpy(dtype=float)
    freqs = df[frequency_column].to_numpy(dtype=float)
    if len(times) < 2:
        return np.zeros(0, dtype=np.float32)

    duration = max(float(times[-1] - times[0]), 0.0)
    n = int(np.ceil(duration * sample_rate))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    t = np.arange(n, dtype=float) / sample_rate + times[0]
    interp_freq = np.interp(t, times, np.where(freqs > 0, freqs, 0.0))
    phase = 2.0 * np.pi * np.cumsum(interp_freq) / sample_rate
    audio = amplitude * np.sin(phase)
    audio[interp_freq <= 0] = 0.0
    return audio.astype(np.float32)


def sonify_f0_csv(csv_path: str | Path, output_wav: str | Path, *, sample_rate: int = 44100) -> Path:
    import soundfile as sf

    df = pd.read_csv(csv_path)
    audio = sonify_f0_dataframe(df, sample_rate=sample_rate)
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), audio, sample_rate)
    return output_wav
