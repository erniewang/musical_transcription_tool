from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "pitchlab.model_folders"

import importlib
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..utils import clean_model_parameters, coerce_frequency_range, f0_dataframe, normalize_audio_array


MODEL_NAME = "swiftf0"
ACCEPTED_PARAMETERS = {"confidence_threshold", "hop_ms"}
MODEL_FMIN = 46.875
MODEL_FMAX = 2093.75


def run_swiftf0(input_audio, parameters=None):
    params = clean_model_parameters(MODEL_NAME, parameters, ACCEPTED_PARAMETERS)
    module = _import_first(("swiftf0", "swift_f0"))
    swift_cls = getattr(module, "SwiftF0", None)
    if swift_cls is None:
        raise AttributeError("swiftf0 backend does not expose a SwiftF0 class")

    sample_rate = int(params["sample_rate"])
    rng = coerce_frequency_range(params, default=(MODEL_FMIN, min(2000.0, MODEL_FMAX)))
    if rng.low_hz < MODEL_FMIN or rng.high_hz > MODEL_FMAX:
        raise ValueError(
            f"swiftf0 only supports {MODEL_FMIN} to {MODEL_FMAX} Hz; "
            f"received {rng.as_tuple()} Hz."
        )
    audio = normalize_audio_array(input_audio)
    detector = swift_cls(
        confidence_threshold=params.get("confidence_threshold"),
        fmin=rng.low_hz,
        fmax=rng.high_hz,
    )
    return _normalize_result(detector.detect_from_array(audio, sample_rate), rng)


def _import_first(module_names: tuple[str, ...]):
    errors = []
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
    raise ImportError("Could not import swiftf0. Tried: " + " | ".join(errors))


def _normalize_result(result: Any, rng):
    pitch = getattr(result, "pitch_hz", None)
    timestamps = getattr(result, "timestamps", None)
    confidence = getattr(result, "confidence", None)
    voicing = getattr(result, "voicing", None)
    if pitch is not None and timestamps is not None:
        extra = {"voiced": np.asarray(voicing, dtype=bool).reshape(-1)} if voicing is not None else None
        return f0_dataframe(
            times=timestamps,
            frequency_hz=pitch,
            confidence=confidence,
            model=MODEL_NAME,
            freq_range=rng,
            extra=extra,
        )

    if isinstance(result, pd.DataFrame):
        df = result.copy()
        freq_col = _find_column(df, ("frequency_hz", "frequency", "f0", "f0_hz", "pitch_hz", "pitch"))
        time_col = _find_column(df, ("time", "times", "timestamp", "t"))
        conf_col = _find_column(df, ("confidence", "periodicity", "probability", "voiced_prob", "score"))
        if freq_col is None:
            raise ValueError("DataFrame result has no recognizable frequency column")
        if time_col is None:
            raise ValueError("DataFrame result has no recognizable timestamp column")
        times = df[time_col].to_numpy(dtype=float)
        conf = df[conf_col].to_numpy(dtype=float) if conf_col else None
        return f0_dataframe(times=times, frequency_hz=df[freq_col].to_numpy(dtype=float), confidence=conf, model=MODEL_NAME, freq_range=rng)

    if isinstance(result, Mapping):
        freq = _first_mapping(result, ("frequency_hz", "frequency", "f0", "f0_hz", "pitch_hz", "pitch"))
        times = _first_mapping(result, ("time", "times", "timestamp", "t"))
        conf = _first_mapping(result, ("confidence", "periodicity", "probability", "voiced_prob", "score"))
        if freq is None:
            raise ValueError("Mapping result has no recognizable frequency key")
        if times is None:
            raise ValueError("Mapping result has no recognizable timestamp key")
        freq = np.asarray(freq, dtype=float).reshape(-1)
        return f0_dataframe(times=times, frequency_hz=freq, confidence=conf, model=MODEL_NAME, freq_range=rng)
    raise ValueError(f"Cannot normalize swiftf0 result of type {type(result).__name__}")


def _find_column(df: pd.DataFrame, names: tuple[str, ...]):
    lookup = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def _first_mapping(data: Mapping[str, Any], names: tuple[str, ...]):
    lower = {str(k).lower(): k for k in data.keys()}
    for name in names:
        key = lower.get(name.lower())
        if key is not None:
            return data[key]
    return None


if __name__ == "__main__":
    from pitchlab.model_folders import run_model_smoke_test

    run_model_smoke_test(run_swiftf0, MODEL_NAME)
