from __future__ import annotations

import importlib
import inspect
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..utils import clean_model_parameters, coerce_frequency_range, f0_dataframe, hop_length_from_ms, normalize_audio_array


MODEL_NAME = "swiftf0"
ACCEPTED_PARAMETERS = {"hop_ms"}


def run_swiftf0(input_audio, parameters=None):
    params = clean_model_parameters(MODEL_NAME, parameters, ACCEPTED_PARAMETERS)
    module = _import_first(("swiftf0", "swift_f0"))
    fn = _find_callable(module, ("predict", "infer", "extract", "estimate", "estimate_f0", "get_f0"))

    sample_rate = int(params["sample_rate"])
    hop_ms = float(params.get("hop_ms", 10.0))
    rng = coerce_frequency_range(params)
    audio = normalize_audio_array(input_audio)
    hop_length = hop_length_from_ms(sample_rate, hop_ms)
    common = {
        "sample_rate": sample_rate,
        "sr": sample_rate,
        "fmin": rng.low_hz,
        "fmax": rng.high_hz,
        "min_frequency": rng.low_hz,
        "max_frequency": rng.high_hz,
        "hop_ms": hop_ms,
        "hop_length": hop_length,
    }

    errors = []
    for args in ((audio, sample_rate), (audio,)):
        try:
            return _normalize_result(_call_with_supported_kwargs(fn, *args, **common), sample_rate, hop_ms, rng)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError(f"swiftf0 backend could not be called. Attempts: {' | '.join(errors)}")


def _import_first(module_names: tuple[str, ...]):
    errors = []
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
    raise ImportError("Could not import swiftf0. Tried: " + " | ".join(errors))


def _find_callable(module: Any, names: tuple[str, ...]):
    for name in names:
        obj = getattr(module, name, None)
        if callable(obj):
            return obj
    for name in dir(module):
        obj = getattr(module, name, None)
        lowered = name.lower()
        if callable(obj) and any(part in lowered for part in ("predict", "infer", "extract", "estimate", "pitch", "f0")):
            return obj
    raise AttributeError(f"No pitch callable found in {module.__name__}")


def _call_with_supported_kwargs(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        sig = inspect.signature(fn)
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        names = set(sig.parameters)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if accepts_kwargs:
        return fn(*args, **kwargs)
    return fn(*args, **{key: value for key, value in kwargs.items() if key in names})


def _normalize_result(result: Any, sample_rate: int, hop_ms: float, rng):
    if isinstance(result, pd.DataFrame):
        df = result.copy()
        freq_col = _find_column(df, ("frequency_hz", "frequency", "f0", "f0_hz", "pitch_hz", "pitch"))
        time_col = _find_column(df, ("time", "times", "timestamp", "t"))
        conf_col = _find_column(df, ("confidence", "periodicity", "probability", "voiced_prob", "score"))
        if freq_col is None:
            raise ValueError("DataFrame result has no recognizable frequency column")
        times = df[time_col].to_numpy(dtype=float) if time_col else np.arange(len(df)) * hop_ms / 1000.0
        conf = df[conf_col].to_numpy(dtype=float) if conf_col else None
        return f0_dataframe(times=times, frequency_hz=df[freq_col].to_numpy(dtype=float), confidence=conf, model=MODEL_NAME, freq_range=rng)

    if isinstance(result, Mapping):
        freq = _first_mapping(result, ("frequency_hz", "frequency", "f0", "f0_hz", "pitch_hz", "pitch"))
        times = _first_mapping(result, ("time", "times", "timestamp", "t"))
        conf = _first_mapping(result, ("confidence", "periodicity", "probability", "voiced_prob", "score"))
        if freq is None:
            raise ValueError("Mapping result has no recognizable frequency key")
        freq = np.asarray(freq, dtype=float).reshape(-1)
        if times is None:
            times = np.arange(len(freq)) * hop_ms / 1000.0
        return f0_dataframe(times=times, frequency_hz=freq, confidence=conf, model=MODEL_NAME, freq_range=rng)

    if isinstance(result, (tuple, list)):
        arrays = [np.asarray(item) for item in result if _is_numeric_array_like(item)]
        if not arrays:
            raise ValueError("Tuple/list result had no numeric arrays")
        if len(arrays) >= 2 and _looks_like_time_axis(arrays[0]):
            times = arrays[0].astype(float).reshape(-1)
            freq = arrays[1].astype(float).reshape(-1)
            conf = arrays[2].astype(float).reshape(-1) if len(arrays) >= 3 else None
        else:
            freq = arrays[0].astype(float).reshape(-1)
            times = np.arange(len(freq)) * hop_ms / 1000.0
            conf = arrays[1].astype(float).reshape(-1) if len(arrays) >= 2 and arrays[1].shape == arrays[0].shape else None
        return f0_dataframe(times=times, frequency_hz=freq, confidence=conf, model=MODEL_NAME, freq_range=rng)

    arr = np.asarray(result)
    if np.issubdtype(arr.dtype, np.number):
        freq = arr.astype(float).reshape(-1)
        times = np.arange(len(freq)) * hop_ms / 1000.0
        return f0_dataframe(times=times, frequency_hz=freq, confidence=None, model=MODEL_NAME, freq_range=rng)
    raise ValueError(f"Cannot normalize swiftf0 result of type {type(result).__name__}")


def _is_numeric_array_like(value: Any) -> bool:
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    return arr.size > 0 and np.issubdtype(arr.dtype, np.number)


def _looks_like_time_axis(arr: np.ndarray) -> bool:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    return len(arr) > 1 and np.nanmin(arr) >= 0 and np.all(np.diff(arr[: min(len(arr), 100)]) >= -1e-9)


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
