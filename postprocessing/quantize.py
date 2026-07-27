"""Snap extracted pitches to a defined musical pitch-class set."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


CHROMATIC_PITCH_CLASSES = tuple(range(12))
_PITCH_CLASS_NAMES = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
    "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
    "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
}


def hz_to_midi(frequency_hz):
    return 69 + 12 * np.log2(np.asarray(frequency_hz) / 440)


def midi_to_hz(midi_note):
    return 440 * 2 ** ((np.asarray(midi_note) - 69) / 12)


def quantize_pitch_run(pitch_run, cfg):
    """Return a copy of one pitch run quantized in its ``frequency_hz`` column.

    Uses ``cfg.pitch_set``, or the chromatic set when that is ``null``.
    """
    pitch_set = cfg.pitch_set or CHROMATIC_PITCH_CLASSES
    quantized = pitch_run.copy()
    quantized["frequency_hz"] = quantize_frequencies(
        quantized["frequency_hz"].to_numpy(), pitch_set=pitch_set
    )
    return quantized


def quantize_frequencies(frequencies_hz, pitch_set: Iterable[int | str] = CHROMATIC_PITCH_CLASSES):
    """Quantize voiced frequencies to their nearest allowed pitch in Hz."""
    pitch_classes = _normalize_pitch_set(pitch_set)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    result = np.zeros_like(frequencies, dtype=float)
    voiced = np.isfinite(frequencies) & (frequencies > 0)
    if not np.any(voiced):
        return result

    midi = hz_to_midi(frequencies[voiced])
    lower_octaves = np.floor(midi / 12).astype(int)
    candidates = (
        (lower_octaves[:, None, None] + np.arange(-1, 2)[None, :, None]) * 12
        + pitch_classes[None, None, :]
    ).reshape(len(midi), -1)
    nearest = candidates[np.arange(len(midi)), np.abs(candidates - midi[:, None]).argmin(axis=1)]
    result[voiced] = midi_to_hz(nearest)
    return result


def _normalize_pitch_set(pitch_set: Iterable[int | str]) -> np.ndarray:
    pitch_classes = []
    for pitch in pitch_set:
        if isinstance(pitch, str):
            normalized = pitch.strip().upper().replace("♯", "#").replace("♭", "B")
            if normalized not in _PITCH_CLASS_NAMES:
                raise ValueError(f"Unknown pitch class: {pitch!r}")
            pitch_classes.append(_PITCH_CLASS_NAMES[normalized])
        elif isinstance(pitch, (int, np.integer)):
            pitch_classes.append(int(pitch) % 12)
        else:
            raise TypeError("pitch_set entries must be note names or integer pitch classes")
    if not pitch_classes:
        raise ValueError("pitch_set must contain at least one pitch class")
    return np.array(sorted(set(pitch_classes)), dtype=float)
