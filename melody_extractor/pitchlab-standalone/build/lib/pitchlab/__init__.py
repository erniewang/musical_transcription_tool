"""Standalone pitchlab package."""

from .registry import available_models, get_model, register_model
from .models.base import FrequencyRange, coerce_frequency_range

__all__ = [
    "FrequencyRange",
    "coerce_frequency_range",
    "available_models",
    "get_model",
    "register_model",
]

__version__ = "0.1.0"
