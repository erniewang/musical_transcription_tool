from __future__ import annotations

import numpy as np

from ..utils import clean_model_parameters, coerce_frequency_range, f0_dataframe, normalize_audio_array, remove_range_kwargs, require_dependency


MODEL_NAME = "yaapt"
ACCEPTED_PARAMETERS = {"hop_ms", "frame_length_ms"}


def run_yaapt(input_audio, parameters=None):
    params = clean_model_parameters(MODEL_NAME, parameters, ACCEPTED_PARAMETERS)
    basic_tools = require_dependency("amfm_decompy.basic_tools", "pip install amfm-decompy")
    pyaapt = require_dependency("amfm_decompy.pYAAPT", "pip install amfm-decompy")

    sample_rate = int(params["sample_rate"])
    hop_ms = float(params.get("hop_ms", 10.0))
    frame_length_ms = float(params.get("frame_length_ms", 35.0))
    rng = coerce_frequency_range(params)

    audio = normalize_audio_array(input_audio).astype(np.float64)
    signal = basic_tools.SignalObj(audio, sample_rate)
    yaapt_params = remove_range_kwargs(params)
    yaapt_params.pop("sample_rate", None)
    yaapt_params.update(
        {
            "frame_length": frame_length_ms,
            "frame_space": hop_ms,
            "f0_min": float(rng.low_hz),
            "f0_max": float(rng.high_hz),
        }
    )
    try:
        pitch = pyaapt.yaapt(signal, **yaapt_params)
    except TypeError:
        yaapt_params.pop("f0_min", None)
        yaapt_params.pop("f0_max", None)
        pitch = pyaapt.yaapt(signal, **yaapt_params)

    f0 = np.asarray(getattr(pitch, "samp_values", getattr(pitch, "values", [])), dtype=float)
    if f0.size == 0 and hasattr(pitch, "frames"):
        f0 = np.asarray(pitch.frames, dtype=float)
    times = np.arange(len(f0), dtype=float) * hop_ms / 1000.0
    confidence = np.where(f0 > 0.0, 1.0, 0.0)
    return f0_dataframe(times=times, frequency_hz=f0, confidence=confidence, model=MODEL_NAME, freq_range=rng)
