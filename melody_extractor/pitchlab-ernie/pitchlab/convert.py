from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .audio_io import save_dataframe, write_wav
from .utils import hz_to_midi


def convert_data(data: Any, desired_form: str | None = None, parameters: Mapping[str, Any] | None = None) -> Any:
    form = _canonical_form(desired_form)
    params = dict(parameters or {})

    if form in {"default", "raw"}:
        return data
    if form in {"pandas_df", "dataframe", "df", "pandas"}:
        return _to_dataframe(data)
    if form == "csv":
        df = _to_dataframe(data)
        output_path = params.get("output_path") or params.get("csv_path")
        if output_path:
            return save_dataframe(df, output_path)
        return df.to_csv(index=False)
    if form in {"dict", "records"}:
        return _to_dataframe(data).to_dict(orient="records")
    if form in {"numpy", "array"}:
        df = _to_dataframe(data)
        if "frequency_hz" in df:
            return df["frequency_hz"].to_numpy(dtype=float)
        if "pitch_midi" in df:
            return df["pitch_midi"].to_numpy(dtype=float)
        return df.to_numpy()
    if form in {"sonified_audio", "sonified", "audio", "sine"}:
        df = _to_dataframe(data)
        sample_rate = int(params.get("sonify_sample_rate", params.get("output_sample_rate", 44100)))
        audio = sonify_f0_dataframe(df, sample_rate=sample_rate, amplitude=float(params.get("amplitude", 0.1)))
        output_path = params.get("output_path") or params.get("wav_path")
        if output_path:
            return write_wav(output_path, audio, sample_rate)
        return audio
    if form == "midi":
        df = _to_dataframe(data)
        output_path = params.get("output_path") or params.get("midi_path")
        return dataframe_to_midi(df, output_path=output_path)
    raise ValueError(f"Unknown desired_output {desired_form!r}")


def sonify_f0_dataframe(
    df: pd.DataFrame,
    *,
    sample_rate: int = 44100,
    amplitude: float = 0.1,
    frequency_column: str = "frequency_hz",
) -> np.ndarray:
    if "time" not in df.columns:
        raise ValueError("Sonification needs a DataFrame with a 'time' column.")
    if frequency_column not in df.columns:
        raise ValueError(f"Sonification needs a {frequency_column!r} column.")

    times = df["time"].to_numpy(dtype=float)
    freqs = df[frequency_column].to_numpy(dtype=float)
    if len(times) < 2:
        return np.zeros(0, dtype=np.float32)

    hop = float(np.median(np.diff(times))) if len(times) > 2 else float(times[-1] - times[0])
    duration = max(float(times[-1] - times[0] + hop), 0.0)
    n = int(np.ceil(duration * sample_rate))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    t = np.arange(n, dtype=float) / sample_rate + times[0]
    voiced_freqs = np.where(freqs > 0, freqs, 0.0)
    interp_freq = np.interp(t, times, voiced_freqs)
    phase = 2.0 * np.pi * np.cumsum(interp_freq) / sample_rate
    audio = amplitude * np.sin(phase)
    audio[interp_freq <= 0] = 0.0
    return audio.astype(np.float32)


def dataframe_to_midi(df: pd.DataFrame, *, output_path: str | Path | None = None) -> Path:
    pretty_midi = _require_pretty_midi()
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    if {"start_time", "end_time", "pitch_midi"}.issubset(df.columns):
        for row in df.itertuples(index=False):
            start = float(getattr(row, "start_time"))
            end = float(getattr(row, "end_time"))
            pitch = int(round(float(getattr(row, "pitch_midi"))))
            velocity = getattr(row, "velocity", 100)
            velocity = 100 if pd.isna(velocity) else int(velocity)
            if end > start and 0 <= pitch <= 127:
                instrument.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
    elif {"time", "frequency_hz"}.issubset(df.columns):
        for start, end, pitch in _f0_to_note_segments(df):
            instrument.notes.append(pretty_midi.Note(velocity=100, pitch=int(pitch), start=float(start), end=float(end)))
    else:
        raise ValueError("MIDI conversion needs note columns or time/frequency_hz columns.")

    pm.instruments.append(instrument)
    if output_path is None:
        fd, name = tempfile.mkstemp(prefix="pitchlab_", suffix=".mid")
        import os

        os.close(fd)
        Path(name).unlink(missing_ok=True)
        output_path = Path(name)
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(output_path))
    return output_path


def _f0_to_note_segments(df: pd.DataFrame) -> list[tuple[float, float, int]]:
    times = df["time"].to_numpy(dtype=float)
    freqs = df["frequency_hz"].to_numpy(dtype=float)
    if len(times) < 2:
        return []
    hop = float(np.median(np.diff(times))) if len(times) > 2 else float(times[-1] - times[0])
    midi = np.rint(hz_to_midi(freqs)).astype(float)
    segments: list[tuple[float, float, int]] = []
    current_pitch = None
    current_start = None

    for idx, (time, freq, pitch) in enumerate(zip(times, freqs, midi)):
        voiced = np.isfinite(freq) and freq > 0 and np.isfinite(pitch)
        if not voiced:
            if current_pitch is not None:
                segments.append((float(current_start), float(time), int(current_pitch)))
                current_pitch = None
                current_start = None
            continue
        pitch_i = int(max(0, min(127, round(float(pitch)))))
        if current_pitch is None:
            current_pitch = pitch_i
            current_start = float(time)
        elif pitch_i != current_pitch:
            segments.append((float(current_start), float(time), int(current_pitch)))
            current_pitch = pitch_i
            current_start = float(time)
        if idx == len(times) - 1 and current_pitch is not None:
            segments.append((float(current_start), float(time + hop), int(current_pitch)))

    return [(start, end, pitch) for start, end, pitch in segments if end > start]


def _to_dataframe(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, Mapping):
        return pd.DataFrame(data)
    return pd.DataFrame(data)


def _canonical_form(desired_form: str | None) -> str:
    if desired_form is None or str(desired_form).strip() == "":
        return "default"
    form = str(desired_form).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "pandas_dataframe": "pandas_df",
        "sonified_sine_wave": "sonified_audio",
        "sonified_audio": "sonified_audio",
        "midi_file": "midi",
    }
    return aliases.get(form, form)


def _require_pretty_midi():
    try:
        import pretty_midi
    except ImportError as exc:
        raise ImportError("MIDI conversion requires pretty_midi. Install with: pip install pretty_midi") from exc
    return pretty_midi
