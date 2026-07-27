from __future__ import annotations

import numpy as np

from ..utils import clean_model_parameters, coerce_frequency_range, f0_dataframe, hop_length_from_ms, normalize_audio_array, require_dependency


MODEL_NAME = "librosa-pyin"
ACCEPTED_PARAMETERS = {"hop_ms", "frame_length", "resolution"}


def run_librosa_pyin(input_audio, parameters=None):
    params = clean_model_parameters(MODEL_NAME, parameters, ACCEPTED_PARAMETERS)
    librosa = require_dependency("librosa", "pip install librosa")

    sample_rate = int(params["sample_rate"])
    hop_ms = float(params.get("hop_ms", 10.0))
    frame_length = int(params.get("frame_length", 2048))
    resolution = float(params.get("resolution", 0.1))
    rng = coerce_frequency_range(params)

    audio = normalize_audio_array(input_audio)
    hop_length = hop_length_from_ms(sample_rate, hop_ms)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=rng.low_hz,
        fmax=rng.high_hz,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        resolution=resolution,
    )
    f0 = np.where(np.asarray(voiced_flag, dtype=bool), f0, 0.0)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sample_rate, hop_length=hop_length)
    return f0_dataframe(times=times, frequency_hz=f0, confidence=voiced_prob, model=MODEL_NAME, freq_range=rng)
