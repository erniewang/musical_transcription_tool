from __future__ import annotations

from typing import Any, Iterable, Type

from .models.base import BasePitchModel, FrequencyRange, coerce_frequency_range, remove_range_kwargs
from .models.basic_pitch_model import BasicPitchModel
from .models.crepe_model import CrepeModel
from .models.essentia_melodia import EssentiaMelodiaModel
from .models.fcpe_model import FcpeModel
from .models.librosa_pyin import LibrosaPyinModel
from .models.librosa_yin import LibrosaYinModel
from .models.penn_model import PennModel
from .models.pyworld_dio import PyWorldDioModel
from .models.pyworld_harvest import PyWorldHarvestModel
from .models.swiftf0_model import SwiftF0Model
from .models.torchcrepe_model import TorchCrepeModel
from .models.yaapt_model import YaaptModel


_MODEL_CLASSES: dict[str, Type[BasePitchModel]] = {}


def _canonical(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def register_model(cls: Type[BasePitchModel], *aliases: str) -> Type[BasePitchModel]:
    names = [cls.name, *getattr(cls, "aliases", ()), *aliases]
    for name in names:
        _MODEL_CLASSES[_canonical(name)] = cls
    return cls


def _register_defaults() -> None:
    for cls in (
        TorchCrepeModel,
        CrepeModel,
        BasicPitchModel,
        EssentiaMelodiaModel,
        LibrosaPyinModel,
        LibrosaYinModel,
        PyWorldDioModel,
        PyWorldHarvestModel,
        YaaptModel,
        PennModel,
        FcpeModel,
        SwiftF0Model,
    ):
        register_model(cls)


_register_defaults()


def available_models() -> list[str]:
    canonical_names = sorted({cls.name for cls in _MODEL_CLASSES.values()})
    return canonical_names


def model_info() -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for cls in _MODEL_CLASSES.values():
        if cls.name in seen:
            continue
        seen.add(cls.name)
        rows.append(
            {
                "name": cls.name,
                "aliases": tuple(getattr(cls, "aliases", ())),
                "output_type": getattr(cls, "output_type", "f0"),
                "supports_native_range": bool(getattr(cls, "supports_native_range", False)),
            }
        )
    return sorted(rows, key=lambda row: row["name"])


def get_model(name: str, **constructor_kwargs: Any) -> BasePitchModel:
    key = _canonical(name)
    if key not in _MODEL_CLASSES:
        valid = ", ".join(available_models())
        raise KeyError(f"Unknown model {name!r}. Available models: {valid}")
    return _MODEL_CLASSES[key](**constructor_kwargs)


def normalize_model_kwargs(**kwargs: Any) -> dict[str, Any]:
    """
    Normalize common range kwargs and remove aliases.

    Useful when a UI sends something like {'range': [100, 500]} and an engine
    wants explicit fmin/fmax.
    """
    freq_range = coerce_frequency_range(kwargs, default=None)
    clean = remove_range_kwargs(kwargs)
    if freq_range is not None:
        clean["fmin"] = freq_range.low_hz
        clean["fmax"] = freq_range.high_hz
        clean["freq_range"] = freq_range
    return clean


def attach_default_range(kwargs: dict[str, Any], default: tuple[float, float] = (50.0, 2000.0)) -> dict[str, Any]:
    clean = dict(kwargs)
    freq_range = coerce_frequency_range(clean, default=default)
    clean["freq_range"] = freq_range
    clean["fmin"] = freq_range.low_hz
    clean["fmax"] = freq_range.high_hz
    return clean


# Backward-compatible names that are convenient to import.
coerce_range = coerce_frequency_range
