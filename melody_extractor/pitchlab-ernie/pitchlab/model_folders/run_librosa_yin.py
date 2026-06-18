from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "pitchlab.model_folders"

import numpy as np

from ..utils import clean_model_parameters, coerce_frequency_range, f0_dataframe, hop_length_from_ms, normalize_audio_array, require_dependency


MODEL_NAME = "librosa-yin"
ACCEPTED_PARAMETERS = {"hop_ms", "frame_length", "trough_threshold"}


def run_librosa_yin(input_audio, parameters=None):
    params = clean_model_parameters(MODEL_NAME, parameters, ACCEPTED_PARAMETERS)
    librosa = require_dependency("librosa", "pip install librosa")

    sample_rate = int(params["sample_rate"])
    hop_ms = float(params.get("hop_ms", 10.0))
    frame_length = int(params.get("frame_length", 2048))
    trough_threshold = float(params.get("trough_threshold", 0.1))
    rng = coerce_frequency_range(params)

    audio = normalize_audio_array(input_audio)
    hop_length = hop_length_from_ms(sample_rate, hop_ms)
    f0 = librosa.yin(
        audio,
        fmin=rng.low_hz,
        fmax=rng.high_hz,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        trough_threshold=trough_threshold,
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sample_rate, hop_length=hop_length)
    confidence = np.where(np.isfinite(f0), 1.0, 0.0)
    return f0_dataframe(times=times, frequency_hz=f0, confidence=confidence, model=MODEL_NAME, freq_range=rng)


if __name__ == "__main__":
    from pitchlab.model_folders import run_model_smoke_test

    run_model_smoke_test(run_librosa_yin, MODEL_NAME)
