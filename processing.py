"""Settings loading and name → function maps for pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from helpers import PROJECT_ROOT, log
from preprocessing import extract_harmonic_part, high_pass, load_audio, low_pass
from postprocessing import (
    constrain_pitch_range,
    filter_by_confidence,
    interpolate,
    quantize_pitch_run,
)

PITCH_SETTINGS_PATH = PROJECT_ROOT / "transcribe_settings.json"
RHYTHM_SETTINGS_PATH = PROJECT_ROOT / "rhythm_settings.json"

PREPROCESSING_OPS = {
    "extract_harmonic_part": extract_harmonic_part,
    "high_pass_filter": high_pass,
    "low_pass_filter": low_pass,
}

PITCH_POSTPROCESSING_OPS = {
    "confidence_filter": filter_by_confidence,
    "pitch_range_filter": constrain_pitch_range,
    "interpolate": interpolate,
    "quantize": quantize_pitch_run,
}

RHYTHM_POSTPROCESSING_OPS = {}


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _ns(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _ns(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_ns(item) for item in value]
    return value


def _unique_orders(steps: dict, location: str) -> None:
    seen = {}
    for name, step in steps.items():
        order = step["order"]
        if order in seen:
            raise ValueError(f"Duplicate order {order} in {location}: {seen[order]!r} and {name!r}")
        seen[order] = name


def load_settings(path: str | Path = PITCH_SETTINGS_PATH):
    """Load settings JSON as nested attribute objects."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")

    data = json.loads(path.read_text())
    audio_file = (data.get("input") or {}).get("audio_file")
    if not audio_file:
        raise ValueError("settings.input.audio_file is required")
    if not (data.get("extraction") or {}).get("models"):
        raise ValueError("settings.extraction.models must be a non-empty list")

    data["input"]["audio_file"] = resolve_path(audio_file)
    directory = (data.get("output") or {}).get("directory")
    if directory is not None:
        data["output"]["directory"] = resolve_path(directory)

    _unique_orders(data.get("preprocessing") or {}, "preprocessing")
    _unique_orders(data.get("postprocessing") or {}, "postprocessing")
    return _ns(data)


def load_input_audio(settings):
    """Load audio from ``settings.input``."""
    audio_in = settings.input
    return load_audio(
        audio_in.audio_file,
        start=audio_in.start_seconds,
        end=audio_in.end_seconds,
        sample_rate=audio_in.sample_rate,
    )


def preprocess_audio(audio, sample_rate, settings):
    """Run enabled preprocessing steps on loaded audio."""
    pre_steps = sorted(
        ((name, getattr(settings.preprocessing, name), fn) for name, fn in PREPROCESSING_OPS.items()),
        key=lambda step: step[1].order,
    )
    for name, cfg, fn in pre_steps:
        if not cfg.enabled:
            continue
        log(f"Preprocessing: {name}")
        audio = fn(audio, sample_rate, cfg)
    return audio
