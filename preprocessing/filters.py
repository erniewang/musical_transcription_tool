"""Transform in-memory audio waveforms.

Each filter takes the step settings object from ``transcribe_settings.json``
so callers can pass that config through unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

_BUTTERWORTH_ORDER = 5


def normalize_audio_array(audio) -> np.ndarray:
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def extract_harmonic_part(audio, sample_rate, cfg=None) -> np.ndarray:
    import librosa

    harmonic, _ = librosa.effects.hpss(normalize_audio_array(audio))
    return harmonic.astype(np.float32)


def high_pass(audio, sample_rate: int, cfg) -> np.ndarray:
    sos = butter(
        _BUTTERWORTH_ORDER, cfg.cutoff_hz, btype="highpass", fs=sample_rate, output="sos"
    )
    return sosfiltfilt(sos, normalize_audio_array(audio)).astype(np.float32)


def low_pass(audio, sample_rate: int, cfg) -> np.ndarray:
    sos = butter(
        _BUTTERWORTH_ORDER, cfg.cutoff_hz, btype="lowpass", fs=sample_rate, output="sos"
    )
    return sosfiltfilt(sos, normalize_audio_array(audio)).astype(np.float32)
