"""Schema and loader for the transcription settings file.

``transcribe_settings.json`` is one complete set of instructions for one piece
of music. This module turns that JSON into typed settings, validating it at the
boundary so the rest of the program can trust the values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = PROJECT_ROOT / "transcribe_settings.json"


@dataclass(frozen=True)
class InputSettings:
    audio_file: Path
    start_seconds: float = 0.0
    end_seconds: float | None = None
    sample_rate: int = 44_100


@dataclass(frozen=True)
class HarmonicExtractionSettings:
    enabled: bool = True
    order: int = 1


@dataclass(frozen=True)
class HighPassFilterSettings:
    enabled: bool = True
    cutoff_hz: float = 131.0
    order: int = 2


@dataclass(frozen=True)
class LowPassFilterSettings:
    enabled: bool = True
    cutoff_hz: float = 2_000.0
    order: int = 3


@dataclass(frozen=True)
class PreprocessingSettings:
    extract_harmonic_part: HarmonicExtractionSettings = HarmonicExtractionSettings()
    high_pass_filter: HighPassFilterSettings = HighPassFilterSettings()
    low_pass_filter: LowPassFilterSettings = LowPassFilterSettings()


@dataclass(frozen=True)
class ExtractionSettings:
    models: tuple[str, ...]
    hop_ms: float = 10.0
    min_pitch_hz: float = 50.0
    max_pitch_hz: float = 2_000.0


@dataclass(frozen=True)
class ConfidenceFilterSettings:
    enabled: bool = True
    min_confidence: float = 0.25
    order: int = 1


@dataclass(frozen=True)
class PitchRangeFilterSettings:
    enabled: bool = True
    range_hz: tuple[float, float] = (165.0, 415.0)
    order: int = 2


@dataclass(frozen=True)
class InterpolationSettings:
    enabled: bool = True
    method: str = "linear"
    order: int = 3


@dataclass(frozen=True)
class QuantizeSettings:
    enabled: bool = True
    pitch_set: tuple[int | str, ...] | None = None
    order: int = 4


@dataclass(frozen=True)
class PostprocessingSettings:
    confidence_filter: ConfidenceFilterSettings = ConfidenceFilterSettings()
    pitch_range_filter: PitchRangeFilterSettings = PitchRangeFilterSettings()
    interpolate: InterpolationSettings = InterpolationSettings()
    quantize: QuantizeSettings = QuantizeSettings()


@dataclass(frozen=True)
class OutputSettings:
    directory: Path | None = None
    sonify_format: str = "mp3"


@dataclass(frozen=True)
class Settings:
    input: InputSettings
    extraction: ExtractionSettings
    preprocessing: PreprocessingSettings = PreprocessingSettings()
    postprocessing: PostprocessingSettings = PostprocessingSettings()
    output: OutputSettings = OutputSettings()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        """Build settings from a plain dict, as loaded from JSON or sent by a UI."""
        _reject_unknown_keys(data, {field.name for field in fields(cls)}, "settings")

        input_data = dict(data.get("input") or {})
        if "audio_file" not in input_data:
            raise ValueError("settings.input.audio_file is required")
        input_data["audio_file"] = resolve_path(input_data["audio_file"])

        output_data = dict(data.get("output") or {})
        if output_data.get("directory") is not None:
            output_data["directory"] = resolve_path(output_data["directory"])

        extraction_data = dict(data.get("extraction") or {})
        if not extraction_data.get("models"):
            raise ValueError("settings.extraction.models must be a non-empty list")
        extraction_data["models"] = tuple(extraction_data["models"])

        preprocessing_data = dict(data.get("preprocessing") or {})
        _reject_unknown_keys(
            preprocessing_data,
            {"extract_harmonic_part", "high_pass_filter", "low_pass_filter"},
            "preprocessing",
        )
        preprocessing = PreprocessingSettings(
            extract_harmonic_part=_section(
                HarmonicExtractionSettings,
                preprocessing_data.get("extract_harmonic_part"),
                "preprocessing.extract_harmonic_part",
            ),
            high_pass_filter=_section(
                HighPassFilterSettings,
                preprocessing_data.get("high_pass_filter"),
                "preprocessing.high_pass_filter",
            ),
            low_pass_filter=_section(
                LowPassFilterSettings,
                preprocessing_data.get("low_pass_filter"),
                "preprocessing.low_pass_filter",
            ),
        )
        _reject_duplicate_orders(
            {
                "extract_harmonic_part": preprocessing.extract_harmonic_part.order,
                "high_pass_filter": preprocessing.high_pass_filter.order,
                "low_pass_filter": preprocessing.low_pass_filter.order,
            },
            "preprocessing",
        )

        postprocessing_data = dict(data.get("postprocessing") or {})
        _reject_unknown_keys(
            postprocessing_data,
            {"confidence_filter", "pitch_range_filter", "interpolate", "quantize"},
            "postprocessing",
        )
        pitch_range_data = dict(postprocessing_data.get("pitch_range_filter") or {})
        if pitch_range_data.get("range_hz") is not None:
            pitch_range_data["range_hz"] = tuple(pitch_range_data["range_hz"])
        quantize_data = dict(postprocessing_data.get("quantize") or {})
        if quantize_data.get("pitch_set") is not None:
            quantize_data["pitch_set"] = tuple(quantize_data["pitch_set"])
        postprocessing = PostprocessingSettings(
            confidence_filter=_section(
                ConfidenceFilterSettings,
                postprocessing_data.get("confidence_filter"),
                "postprocessing.confidence_filter",
            ),
            pitch_range_filter=_section(
                PitchRangeFilterSettings, pitch_range_data, "postprocessing.pitch_range_filter"
            ),
            interpolate=_section(
                InterpolationSettings,
                postprocessing_data.get("interpolate"),
                "postprocessing.interpolate",
            ),
            quantize=_section(QuantizeSettings, quantize_data, "postprocessing.quantize"),
        )
        _reject_duplicate_orders(
            {
                "confidence_filter": postprocessing.confidence_filter.order,
                "pitch_range_filter": postprocessing.pitch_range_filter.order,
                "interpolate": postprocessing.interpolate.order,
                "quantize": postprocessing.quantize.order,
            },
            "postprocessing",
        )

        return cls(
            input=_section(InputSettings, input_data, "input"),
            preprocessing=preprocessing,
            extraction=_section(ExtractionSettings, extraction_data, "extraction"),
            postprocessing=postprocessing,
            output=_section(OutputSettings, output_data, "output"),
        )


def load_settings(path: str | Path = SETTINGS_PATH) -> Settings:
    """Read and validate the settings file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    return Settings.from_dict(json.loads(path.read_text()))


def resolve_path(value: str | Path) -> Path:
    """Resolve a settings path, treating relative paths as project-relative."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _section(section_class, data: dict[str, Any] | None, name: str):
    values = dict(data or {})
    _reject_unknown_keys(values, {field.name for field in fields(section_class)}, f"settings.{name}")
    return section_class(**values)


def _reject_unknown_keys(data: dict[str, Any], known: set[str], location: str) -> None:
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"Unknown key(s) in {location}: {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(known))}"
        )


def _reject_duplicate_orders(orders: dict[str, int], location: str) -> None:
    seen: dict[int, str] = {}
    for name, order in orders.items():
        if order in seen:
            raise ValueError(
                f"Duplicate order {order} in {location}: "
                f"{seen[order]!r} and {name!r}"
            )
        seen[order] = name
