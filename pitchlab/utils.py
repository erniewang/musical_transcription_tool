from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


RANGE_PAIR_KEYS = (
    "range",
    "freq_range",
    "f0_range",
    "pitch_range",
    "hz_range",
    "frequency_range",
)

LOW_KEYS = (
    "fmin",
    "min_frequency",
    "minimum_frequency",
    "min_hz",
    "low_hz",
    "lower_hz",
    "f0_min",
    "f0_floor",
)

HIGH_KEYS = (
    "fmax",
    "max_frequency",
    "maximum_frequency",
    "max_hz",
    "high_hz",
    "upper_hz",
    "f0_max",
    "f0_ceil",
)

COMMON_PARAMETER_KEYS = {
    "sample_rate",
    "sr",
    *RANGE_PAIR_KEYS,
    *LOW_KEYS,
    *HIGH_KEYS,
}


@dataclass(frozen=True)
class FrequencyRange:
    low_hz: float
    high_hz: float

    def __post_init__(self) -> None:
        low = float(self.low_hz)
        high = float(self.high_hz)
        if low <= 0 or high <= 0:
            raise ValueError(f"Frequency bounds must be positive: {(low, high)!r}")
        if low >= high:
            raise ValueError(f"Lower frequency must be below upper frequency: {(low, high)!r}")
        object.__setattr__(self, "low_hz", low)
        object.__setattr__(self, "high_hz", high)

    @property
    def fmin(self) -> float:
        return self.low_hz

    @property
    def fmax(self) -> float:
        return self.high_hz

    def as_tuple(self) -> tuple[float, float]:
        return (self.low_hz, self.high_hz)

    def __iter__(self):
        yield self.low_hz
        yield self.high_hz


def require_dependency(module_name: str, install_hint: str | None = None):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        hint = f" Install with: {install_hint}" if install_hint else ""
        raise ImportError(f"Missing optional dependency '{module_name}'.{hint}") from exc


def normalize_audio_array(audio: Any) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError("Audio must be an array-like waveform, not a scalar.")
    if arr.ndim == 1:
        return np.ascontiguousarray(arr)
    if arr.ndim == 2:
        if arr.shape[0] <= 8:
            arr = arr.mean(axis=0)
        elif arr.shape[1] <= 8:
            arr = arr.mean(axis=1)
        else:
            arr = arr.reshape(-1)
        return np.ascontiguousarray(arr.astype(np.float32))
    return np.ascontiguousarray(arr.reshape(-1).astype(np.float32))


def normalize_parameters(parameters: Mapping[str, Any] | None, *, default_sample_rate: int = 44100) -> dict[str, Any]:
    params = dict(parameters or {})
    if "sample_rate" not in params and "sr" in params:
        params["sample_rate"] = params["sr"]
    if "sample_rate" not in params:
        warnings.warn(
            f"No sample_rate/sr supplied; using {default_sample_rate} Hz.",
            RuntimeWarning,
            stacklevel=2,
        )
        params["sample_rate"] = int(default_sample_rate)
    params["sample_rate"] = int(params["sample_rate"])
    return params


def clean_model_parameters(
    model_name: str,
    parameters: Mapping[str, Any] | None,
    accepted: Iterable[str],
    *,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    params = dict(parameters or {})
    aliases = dict(aliases or {})
    for old, new in aliases.items():
        if old in params and new not in params:
            params[new] = params.pop(old)

    accepted_keys = set(accepted) | COMMON_PARAMETER_KEYS
    unsupported = sorted(key for key in params if key not in accepted_keys)
    if unsupported:
        warnings.warn(
            f"{model_name} does not accept parameter(s): {', '.join(unsupported)}; ignoring them.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {key: value for key, value in params.items() if key in accepted_keys}


def _as_mapping(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    out: dict[str, Any] = {}
    for key in (*RANGE_PAIR_KEYS, *LOW_KEYS, *HIGH_KEYS):
        if hasattr(obj, key):
            out[key] = getattr(obj, key)
    return out


def _parse_pair(pair: Any) -> tuple[float, float]:
    if isinstance(pair, FrequencyRange):
        return pair.low_hz, pair.high_hz
    if isinstance(pair, str):
        parts = pair.replace(",", " ").split()
    else:
        try:
            parts = list(pair)
        except TypeError as exc:
            raise ValueError(f"Range must be a pair, not {pair!r}") from exc
    if len(parts) != 2:
        raise ValueError(f"Range must contain exactly two values: {pair!r}")
    return float(parts[0]), float(parts[1])


def coerce_frequency_range(
    value: Any = None,
    /,
    *,
    default: tuple[float, float] | FrequencyRange | None = (50.0, 2000.0),
    **kwargs: Any,
) -> FrequencyRange | None:
    if isinstance(value, FrequencyRange):
        return value

    if value is not None and not isinstance(value, Mapping):
        mapped = _as_mapping(value)
        if not mapped:
            low, high = _parse_pair(value)
            return FrequencyRange(low, high)
        data = mapped
    else:
        data = _as_mapping(value)

    data.update({key: val for key, val in kwargs.items() if val is not None})

    for key in RANGE_PAIR_KEYS:
        if data.get(key) is not None:
            low, high = _parse_pair(data[key])
            return FrequencyRange(low, high)

    lows = [data[key] for key in LOW_KEYS if data.get(key) is not None]
    highs = [data[key] for key in HIGH_KEYS if data.get(key) is not None]
    if lows or highs:
        if not lows or not highs:
            raise ValueError("Both lower and upper frequency bounds are required.")
        return FrequencyRange(float(lows[0]), float(highs[0]))

    if default is None:
        return None
    if isinstance(default, FrequencyRange):
        return default
    return FrequencyRange(float(default[0]), float(default[1]))


def remove_range_kwargs(parameters: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(parameters)
    for key in (*RANGE_PAIR_KEYS, *LOW_KEYS, *HIGH_KEYS):
        clean.pop(key, None)
    return clean


def hop_length_from_ms(sample_rate: int | float, hop_ms: float) -> int:
    hop = int(round(float(sample_rate) * float(hop_ms) / 1000.0))
    return max(1, hop)


def hz_to_midi(frequency_hz: float | np.ndarray) -> float | np.ndarray:
    freq = np.asarray(frequency_hz, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69.0 + 12.0 * np.log2(freq / 440.0)
    return midi


def midi_to_hz(midi_note: float | np.ndarray) -> float | np.ndarray:
    return 440.0 * np.power(2.0, (np.asarray(midi_note, dtype=float) - 69.0) / 12.0)


def filter_frequencies(
    frequencies: Any,
    freq_range: Optional[FrequencyRange],
    *,
    invalid_value: float = 0.0,
) -> np.ndarray:
    arr = np.asarray(frequencies, dtype=float).reshape(-1).copy()
    mask = ~np.isfinite(arr)
    if freq_range is not None:
        mask |= (arr < freq_range.low_hz) | (arr > freq_range.high_hz)
    arr[mask] = invalid_value
    return arr


def f0_dataframe(
    *,
    times: Any,
    frequency_hz: Any,
    confidence: Any = None,
    model: str,
    freq_range: Optional[FrequencyRange] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    f0 = filter_frequencies(frequency_hz, freq_range)
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    if times_arr.shape != f0.shape:
        times_arr = np.resize(times_arr, f0.shape)

    if confidence is None:
        conf_arr = np.where(f0 > 0.0, 1.0, 0.0)
    else:
        conf_arr = np.asarray(confidence, dtype=float).reshape(-1)
        if conf_arr.shape != f0.shape:
            conf_arr = np.resize(conf_arr, f0.shape)

    df = pd.DataFrame(
        {
            "time": times_arr,
            "frequency_hz": f0,
            "confidence": conf_arr,
            "voiced": f0 > 0.0,
            "model": model,
        }
    )
    if freq_range is not None:
        df["low_hz"] = freq_range.low_hz
        df["high_hz"] = freq_range.high_hz
    if extra:
        for key, value in extra.items():
            df[key] = value
    return df


def note_events_dataframe(
    events: Iterable[Any],
    *,
    model: str,
    freq_range: Optional[FrequencyRange] = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    midi_low = midi_high = None
    if freq_range is not None:
        midi_low = float(hz_to_midi(freq_range.low_hz))
        midi_high = float(hz_to_midi(freq_range.high_hz))

    for event in list(events):
        row = normalize_note_event(event)
        if row is None:
            continue
        midi = row.get("pitch_midi")
        if midi is not None and midi_low is not None and midi_high is not None:
            if not (midi_low <= float(midi) <= midi_high):
                continue
        if row.get("frequency_hz") is None and midi is not None:
            row["frequency_hz"] = float(midi_to_hz(float(midi)))
        row["time"] = row.get("start_time")
        row["model"] = model
        if freq_range is not None:
            row["low_hz"] = freq_range.low_hz
            row["high_hz"] = freq_range.high_hz
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_note_event(event: Any) -> dict[str, Any] | None:
    if event is None:
        return None
    if isinstance(event, Mapping):
        start = event.get("start_time", event.get("start", event.get("onset")))
        end = event.get("end_time", event.get("end", event.get("offset")))
        midi = event.get("pitch_midi", event.get("midi", event.get("pitch")))
        velocity = event.get("velocity")
        amplitude = event.get("amplitude", event.get("confidence", event.get("score")))
    elif hasattr(event, "start") or hasattr(event, "pitch"):
        start = getattr(event, "start_time", getattr(event, "start", getattr(event, "onset", None)))
        end = getattr(event, "end_time", getattr(event, "end", getattr(event, "offset", None)))
        midi = getattr(event, "pitch_midi", getattr(event, "midi", getattr(event, "pitch", None)))
        velocity = getattr(event, "velocity", None)
        amplitude = getattr(event, "amplitude", getattr(event, "confidence", None))
    else:
        try:
            seq = list(event)
        except TypeError:
            return None
        if len(seq) < 3:
            return None
        start, end, midi = seq[0], seq[1], seq[2]
        velocity = seq[3] if len(seq) > 3 else None
        amplitude = seq[4] if len(seq) > 4 else None

    try:
        start_f = float(start)
        end_f = float(end)
        midi_f = float(midi)
    except (TypeError, ValueError):
        return None

    return {
        "start_time": start_f,
        "end_time": end_f,
        "duration": max(0.0, end_f - start_f),
        "pitch_midi": midi_f,
        "velocity": velocity,
        "amplitude": amplitude,
    }
