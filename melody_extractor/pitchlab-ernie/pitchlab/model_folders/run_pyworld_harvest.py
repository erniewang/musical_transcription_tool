from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "pitchlab.model_folders"

import numpy as np

from ..utils import clean_model_parameters, coerce_frequency_range, f0_dataframe, normalize_audio_array, require_dependency


MODEL_NAME = "pyworld-harvest"
ACCEPTED_PARAMETERS = {"hop_ms", "refine"}


def run_pyworld_harvest(input_audio, parameters=None):
    params = clean_model_parameters(MODEL_NAME, parameters, ACCEPTED_PARAMETERS)
    pw = require_dependency("pyworld", "pip install pyworld")

    sample_rate = int(params["sample_rate"])
    hop_ms = float(params.get("hop_ms", 10.0))
    refine = bool(params.get("refine", True))
    rng = coerce_frequency_range(params)

    audio = normalize_audio_array(input_audio).astype(np.float64)
    f0, time_axis = pw.harvest(
        audio,
        sample_rate,
        f0_floor=float(rng.low_hz),
        f0_ceil=float(rng.high_hz),
        frame_period=hop_ms,
    )
    if refine:
        f0 = pw.stonemask(audio, f0, time_axis, sample_rate)
    confidence = np.where(np.asarray(f0) > 0.0, 1.0, 0.0)
    return f0_dataframe(times=time_axis, frequency_hz=f0, confidence=confidence, model=MODEL_NAME, freq_range=rng)


if __name__ == "__main__":
    from pitchlab.model_folders import run_model_smoke_test

    run_model_smoke_test(run_pyworld_harvest, MODEL_NAME)
