"""Drop pitch estimates whose confidence falls below a threshold."""

from __future__ import annotations


DEFAULT_MIN_CONFIDENCE = 0.25


def filter_by_confidence(data_frame, cfg):
    """Zero out frequencies whose confidence is below ``cfg.min_confidence``.

    ``min_confidence`` is a fraction in ``[0, 1]`` (default ``0.25`` = 25%).
    Frames are kept so the time grid stays intact; only the pitch estimate
    is cleared. Rows without a confidence column are returned unchanged.
    """
    confidence_column = "confidence"
    frequency_column = "frequency_hz"
    min_confidence = cfg.min_confidence

    if confidence_column not in data_frame.columns:
        return data_frame.copy()

    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence!r}")

    filtered = data_frame.copy()
    low_confidence = filtered[confidence_column].fillna(0.0) < min_confidence
    filtered.loc[low_confidence, frequency_column] = 0.0
    if "voiced" in filtered.columns:
        filtered.loc[low_confidence, "voiced"] = False
    return filtered
