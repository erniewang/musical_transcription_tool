from __future__ import annotations

import importlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .base import FrequencyRange, call_with_supported_kwargs, filter_frequencies, f0_dataframe, hop_length_from_ms


def import_first(module_names: Sequence[str]):
    errors = []
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
    raise ImportError("Could not import any of: " + ", ".join(module_names) + ". Errors: " + " | ".join(errors))


def find_callable(module: Any, names: Sequence[str]):
    for name in names:
        obj = getattr(module, name, None)
        if callable(obj):
            return obj
    # Some libraries expose a class with a predict/infer method.
    for class_name in ("Model", "PitchModel", "Inferencer", "Estimator"):
        cls = getattr(module, class_name, None)
        if cls is None:
            continue
        try:
            instance = cls()
        except Exception:
            continue
        for method_name in names:
            method = getattr(instance, method_name, None)
            if callable(method):
                return method
    raise AttributeError(f"No callable found. Tried: {', '.join(names)}")


def generic_pitch_call(
    *,
    module_names: Sequence[str],
    callable_names: Sequence[str],
    audio: Any,
    sample_rate: int,
    model_name: str,
    freq_range: FrequencyRange,
    hop_ms: float,
    kwargs: Mapping[str, Any],
) -> pd.DataFrame:
    module = import_first(module_names)
    fn = find_callable(module, callable_names)

    x = np.asarray(audio, dtype=np.float32)
    hop_length = hop_length_from_ms(sample_rate, hop_ms)
    common = dict(kwargs)
    common.update(
        {
            "sample_rate": int(sample_rate),
            "sr": int(sample_rate),
            "fmin": float(freq_range.low_hz),
            "fmax": float(freq_range.high_hz),
            "min_frequency": float(freq_range.low_hz),
            "max_frequency": float(freq_range.high_hz),
            "f0_min": float(freq_range.low_hz),
            "f0_max": float(freq_range.high_hz),
            "hop_ms": float(hop_ms),
            "hop_length": int(hop_length),
            "hopsize": float(hop_ms) / 1000.0,
        }
    )

    errors = []
    for call_args in ((x, int(sample_rate)), (x,), (audio, int(sample_rate)), (audio,)):
        try:
            result = call_with_supported_kwargs(fn, *call_args, **common)
            return normalize_external_result(result, model_name=model_name, freq_range=freq_range, hop_ms=hop_ms, sample_rate=sample_rate)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError(f"External adapter for {model_name} could not call backend. Attempts: {' | '.join(errors)}")


def normalize_external_result(
    result: Any,
    *,
    model_name: str,
    freq_range: FrequencyRange,
    hop_ms: float,
    sample_rate: int,
) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        df = result.copy()
        freq_col = _find_column(df, ("frequency_hz", "frequency", "f0", "f0_hz", "pitch_hz", "pitch"))
        time_col = _find_column(df, ("time", "times", "timestamp", "t"))
        conf_col = _find_column(df, ("confidence", "periodicity", "probability", "voiced_prob", "score"))
        if freq_col is None:
            raise ValueError("DataFrame result has no recognizable frequency column")
        times = df[time_col].to_numpy(dtype=float) if time_col else np.arange(len(df)) * hop_ms / 1000.0
        conf = df[conf_col].to_numpy(dtype=float) if conf_col else None
        return f0_dataframe(times=times, frequency_hz=df[freq_col].to_numpy(dtype=float), confidence=conf, model=model_name, freq_range=freq_range)

    if isinstance(result, Mapping):
        freq = _first_mapping(result, ("frequency_hz", "frequency", "f0", "f0_hz", "pitch_hz", "pitch"))
        times = _first_mapping(result, ("time", "times", "timestamp", "t"))
        conf = _first_mapping(result, ("confidence", "periodicity", "probability", "voiced_prob", "score"))
        if freq is None:
            raise ValueError("Mapping result has no recognizable frequency key")
        freq = np.asarray(freq, dtype=float)
        if times is None:
            times = np.arange(len(freq)) * hop_ms / 1000.0
        return f0_dataframe(times=times, frequency_hz=freq, confidence=conf, model=model_name, freq_range=freq_range)

    if isinstance(result, (tuple, list)):
        items = list(result)
        arrays = [np.asarray(item) for item in items if _is_numeric_array_like(item)]
        if not arrays:
            raise ValueError("Tuple/list result had no numeric arrays")
        # Common forms: (f0, periodicity), (time, f0, confidence), (pitch, voiced, periodicity)
        if len(arrays) >= 3 and _looks_like_time_axis(arrays[0]):
            times = arrays[0].astype(float)
            freq = arrays[1].astype(float)
            conf = arrays[2].astype(float)
        elif len(arrays) >= 2 and _looks_like_time_axis(arrays[0]) and _looks_like_frequency(arrays[1]):
            times = arrays[0].astype(float)
            freq = arrays[1].astype(float)
            conf = arrays[2].astype(float) if len(arrays) >= 3 else None
        else:
            freq = arrays[0].astype(float)
            conf = arrays[1].astype(float) if len(arrays) >= 2 and arrays[1].shape == freq.shape else None
            times = np.arange(len(freq)) * hop_ms / 1000.0
        return f0_dataframe(times=times, frequency_hz=freq, confidence=conf, model=model_name, freq_range=freq_range)

    arr = np.asarray(result)
    if np.issubdtype(arr.dtype, np.number):
        freq = arr.astype(float).reshape(-1)
        times = np.arange(len(freq)) * hop_ms / 1000.0
        return f0_dataframe(times=times, frequency_hz=freq, confidence=None, model=model_name, freq_range=freq_range)

    raise ValueError(f"Cannot normalize result of type {type(result).__name__}")


def _is_numeric_array_like(value: Any) -> bool:
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    return arr.size > 0 and np.issubdtype(arr.dtype, np.number)


def _looks_like_time_axis(arr: np.ndarray) -> bool:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if len(arr) < 2:
        return False
    return np.nanmin(arr) >= 0 and np.nanmax(arr) < 24 * 60 * 60 and np.all(np.diff(arr[: min(len(arr), 100)]) >= -1e-9)


def _looks_like_frequency(arr: np.ndarray) -> bool:
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size == 0:
        return False
    med = float(np.median(finite))
    return 20.0 <= med <= 5000.0


def _find_column(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lookup = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def _first_mapping(data: Mapping[str, Any], names: Iterable[str]):
    lower = {str(k).lower(): k for k in data.keys()}
    for name in names:
        key = lower.get(name.lower())
        if key is not None:
            return data[key]
    return None
