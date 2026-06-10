"""Pitch-run cleanup functions used by the refinement notebook."""

import numpy as np


def apply_to_runs(data_frames, function, **kwargs):
    """Apply one cleanup function to each DataFrame."""
    return [function(data_frame, **kwargs) for data_frame in data_frames]


def constrain_pitch_range(
    data_frame,
    frequency_column="frequency_hz",
    lowest_note=None,
    highest_note=None,
):
    """Set frequencies outside the selected range to NaN."""
    refined = data_frame.copy()

    if lowest_note is not None:
        refined.loc[refined[frequency_column] < lowest_note, frequency_column] = np.nan
    if highest_note is not None:
        refined.loc[refined[frequency_column] > highest_note, frequency_column] = np.nan

    return refined


def replace_zeroes_with_nan(data_frame, columns=("frequency_hz",)):
    """Replace zeroes in selected columns with NaN."""
    refined = data_frame.copy()
    for column in _selected_columns(refined, columns):
        refined[column] = refined[column].replace(0, np.nan)
    return refined


def interpolate_missing_values(
    data_frame,
    columns=("frequency_hz",),
    method="linear",
    fill_value=0,
):
    """Interpolate missing values in selected columns."""
    refined = data_frame.copy()
    for column in _selected_columns(refined, columns):
        refined[column] = refined[column].interpolate(method=method).fillna(fill_value)
    return refined


def _selected_columns(data_frame, columns):
    if columns is None:
        return data_frame.select_dtypes(include="number").columns

    if isinstance(columns, str):
        columns = (columns,)

    missing_columns = [column for column in columns if column not in data_frame.columns]
    if missing_columns:
        raise KeyError(f"Missing column(s): {', '.join(missing_columns)}")

    return columns
