from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "pitchlab.model_folders"

import numpy as np

from ..utils import clean_model_parameters, coerce_frequency_range, f0_dataframe, hop_length_from_ms, normalize_audio_array, remove_range_kwargs, require_dependency


MODEL_NAME = "essentia-melodia"
ACCEPTED_PARAMETERS = {"hop_ms", "frame_size", "guess_unvoiced", "min_confidence"}


def run_essentia_melodia(input_audio, parameters=None):
    params = clean_model_parameters(MODEL_NAME, parameters, ACCEPTED_PARAMETERS)
    es = require_dependency("essentia.standard", "pip install essentia")

    sample_rate = int(params["sample_rate"])
    hop_ms = float(params.get("hop_ms", 10.0))
    frame_size = int(params.get("frame_size", 2048))
    guess_unvoiced = bool(params.get("guess_unvoiced", False))
    min_confidence = params.get("min_confidence")
    rng = coerce_frequency_range(params)

    audio = normalize_audio_array(input_audio).astype(np.float32)
    hop_size = hop_length_from_ms(sample_rate, hop_ms)
    alg_kwargs = remove_range_kwargs(params)
    for key in ("sample_rate", "hop_ms", "frame_size", "guess_unvoiced", "min_confidence"):
        alg_kwargs.pop(key, None)
    alg_kwargs.update(
        {
            "sampleRate": sample_rate,
            "frameSize": frame_size,
            "hopSize": hop_size,
            "minFrequency": float(rng.low_hz),
            "maxFrequency": float(rng.high_hz),
            "guessUnvoiced": guess_unvoiced,
        }
    )

    pitch, confidence = es.PredominantPitchMelodia(**alg_kwargs)(audio)
    pitch = np.asarray(pitch, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    if min_confidence is not None:
        pitch[confidence < float(min_confidence)] = 0.0
    times = np.arange(len(pitch), dtype=float) * hop_size / float(sample_rate)
    return f0_dataframe(times=times, frequency_hz=pitch, confidence=confidence, model=MODEL_NAME, freq_range=rng)


if __name__ == "__main__":
    from pitchlab.model_folders import run_model_smoke_test
    run_model_smoke_test(run_essentia_melodia, MODEL_NAME)
