"""Turn pitch runs into MIDI, music21 scores, and notation PDFs."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from .quantize import hz_to_midi


def dataframe_to_midi(data: pd.DataFrame, output_path: str | Path) -> Path:
    import pretty_midi

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    if {"start_time", "end_time", "pitch_midi"}.issubset(data.columns):
        notes = data[["start_time", "end_time", "pitch_midi"]].itertuples(index=False, name=None)
    else:
        notes = _pitch_segments(data)
    for start, end, pitch in notes:
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=int(pitch), start=float(start), end=float(end)))
    midi.instruments.append(instrument)
    output_path = Path(output_path)
    midi.write(str(output_path))
    return output_path


def _configure_score(score, piece_name: str, model_name: str):
    from music21 import clef, metadata

    score.metadata = metadata.Metadata()
    score.metadata.title = f"{piece_name} ({model_name})"
    for part in score.parts:
        for existing in list(part.recurse().getElementsByClass(clef.Clef)):
            existing.activeSite.remove(existing)
        part.insert(0, clef.TrebleClef())
    return score


def score_from_pitch_run(pitch_run, piece_name: str):
    """Build a monophonic score from f0 segments without rhythmic quantization."""
    from music21 import meter, note, stream, tempo

    model_name = str(pitch_run["model"].iloc[0])
    segments = _pitch_segments(pitch_run)
    times = pitch_run["time"].to_numpy(dtype=float)
    hop_seconds = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    # Scale tempo so one analysis frame equals a 32nd note (export-safe, unquantized lengths).
    seconds_per_quarter = hop_seconds / 0.125
    bpm = 60.0 / seconds_per_quarter

    part = stream.Part()
    part.insert(0, tempo.MetronomeMark(number=bpm))
    part.insert(0, meter.TimeSignature("4/4"))
    for start, end, pitch_midi in segments:
        duration_quarters = (end - start) / seconds_per_quarter
        if duration_quarters <= 0:
            continue
        pitched = note.Note(int(pitch_midi))
        pitched.quarterLength = duration_quarters
        part.insert(start / seconds_per_quarter, pitched)

    score = stream.Score()
    score.append(part)
    return _configure_score(score, piece_name, model_name)


@contextmanager
def _offscreen_musescore():
    """Temporarily force MuseScore/Qt to run headless, then restore the previous env."""
    previous_platform = os.environ.get("QT_QPA_PLATFORM")
    previous_musescore_platform = os.environ.get("MU_QT_QPA_PLATFORM")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["MU_QT_QPA_PLATFORM"] = "offscreen"
    try:
        yield
    finally:
        if previous_platform is None:
            os.environ.pop("QT_QPA_PLATFORM", None)
        else:
            os.environ["QT_QPA_PLATFORM"] = previous_platform
        if previous_musescore_platform is None:
            os.environ.pop("MU_QT_QPA_PLATFORM", None)
        else:
            os.environ["MU_QT_QPA_PLATFORM"] = previous_musescore_platform


def write_score_pdf(score, output_path: str | Path) -> Path:
    """Write notation to a PDF via MuseScore without requiring a desktop display."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _offscreen_musescore():
        score.write("musicxml.pdf", fp=str(output_path))
    return output_path


def _pitch_segments(data: pd.DataFrame):
    times = data["time"].to_numpy(dtype=float)
    frequencies = data["frequency_hz"].to_numpy(dtype=float)
    pitches = np.rint(hz_to_midi(frequencies))
    hop = np.median(np.diff(times))
    segments, active_pitch, start = [], None, None
    for time, frequency, pitch in zip(times, frequencies, pitches):
        pitch = int(pitch) if frequency > 0 else None
        if pitch != active_pitch:
            if active_pitch is not None:
                segments.append((start, time, active_pitch))
            active_pitch, start = pitch, time
    if active_pitch is not None:
        segments.append((start, times[-1] + hop, active_pitch))
    return segments
