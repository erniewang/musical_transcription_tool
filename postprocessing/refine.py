"""Clean up raw pitch runs: range filtering and gap interpolation."""

from __future__ import annotations

import numpy as np


def constrain_pitch_range(data_frame, cfg):
    """Set frequencies outside ``cfg.range_hz`` to NaN."""
    frequency_column = "frequency_hz"
    lowest_note, highest_note = cfg.range_hz
    refined = data_frame.copy()
    refined.loc[refined[frequency_column] < lowest_note, frequency_column] = np.nan
    refined.loc[refined[frequency_column] > highest_note, frequency_column] = np.nan
    return refined


def interpolate(data_frame, cfg):
    """Replace zeroes with NaN, then interpolate using ``cfg.method``."""
    frequency_column = "frequency_hz"
    refined = data_frame.copy()
    refined[frequency_column] = (
        refined[frequency_column]
        .replace(0, np.nan)
        .interpolate(method=cfg.method)
        .fillna(0.0)
    )
    return refined
