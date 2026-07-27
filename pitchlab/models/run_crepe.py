from __future__ import annotations

import numpy as np

from ..utils import clean_model_parameters, coerce_frequency_range, f0_dataframe, normalize_audio_array, require_dependency


MODEL_NAME = "crepe"
ACCEPTED_PARAMETERS = {"hop_ms", "model_capacity", "model", "viterbi", "center", "verbose", "min_confidence"}


def run_crepe(input_audio, parameters=None):
    params = clean_model_parameters(MODEL_NAME, parameters, ACCEPTED_PARAMETERS)
    crepe = require_dependency("crepe", "pip install crepe tensorflow")

    sample_rate = int(params["sample_rate"])
    hop_ms = float(params.get("hop_ms", 10.0))
    model_capacity = params.get("model", params.get("model_capacity", "full"))
    viterbi = bool(params.get("viterbi", False))
    center = bool(params.get("center", True))
    verbose = int(params.get("verbose", 0))
    min_confidence = params.get("min_confidence", 0.0)
    rng = coerce_frequency_range(params)

    audio = normalize_audio_array(input_audio)
    try:
        from crepe.core import get_activation, to_local_average_cents, to_viterbi_cents

        activation = get_activation(
            audio,
            sample_rate,
            model_capacity=model_capacity,
            center=center,
            step_size=hop_ms,
            verbose=verbose,
        )
        bin_freqs = _crepe_bin_frequencies()
        allowed = (bin_freqs >= rng.low_hz) & (bin_freqs <= rng.high_hz)
        if not np.any(allowed):
            raise ValueError(f"No CREPE pitch bins are inside range {rng.as_tuple()} Hz")

        constrained = np.array(activation, copy=True)
        constrained[:, ~allowed] = 0.0
        confidence = constrained.max(axis=1)
        cents = to_viterbi_cents(constrained) if viterbi else to_local_average_cents(constrained)
        frequency = _cents_to_hz(cents)
        times = np.arange(len(frequency), dtype=float) * hop_ms / 1000.0
    except Exception:
        times, frequency, confidence, _activation = crepe.predict(
            audio,
            sample_rate,
            step_size=hop_ms,
            viterbi=viterbi,
            model_capacity=model_capacity,
            center=center,
            verbose=verbose,
        )

    frequency = np.asarray(frequency, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    if min_confidence is not None:
        frequency[confidence < float(min_confidence)] = 0.0
    return f0_dataframe(
        times=times,
        frequency_hz=frequency,
        confidence=confidence,
        model=MODEL_NAME,
        freq_range=rng,
        extra={"backend_model": model_capacity},
    )


def _crepe_bin_frequencies() -> np.ndarray:
    cents = 20.0 * np.arange(360) + 1997.3794084376191
    return 10.0 * np.power(2.0, cents / 1200.0)


def _cents_to_hz(cents: np.ndarray) -> np.ndarray:
    return 10.0 * np.power(2.0, np.asarray(cents, dtype=float) / 1200.0)
