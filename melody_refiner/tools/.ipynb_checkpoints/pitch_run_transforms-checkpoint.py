"""CSV loading and dataframe transformation helpers for pitch runs."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_pitch_runs(input_dir, suffix=".csv"):
    """Load pitch-run CSV files from a directory into a list of DataFrames."""
    input_path = Path(input_dir)
    return [
        pd.read_csv(file)
        for file in sorted(input_path.iterdir())
        if file.suffix == suffix
    ]


def save_pitch_runs(data_frames, output_dir, filename_column="model", suffix="_refined"):
    """Save transformed pitch-run DataFrames and return the written paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for index, df in enumerate(data_frames, start=1):
        if filename_column in df.columns and not df.empty:
            filename = f"{df[filename_column].iloc[0]}{suffix}.csv"
        else:
            filename = f"pitch_run_{index}{suffix}.csv"

        csv_path = output_path / filename
        df.to_csv(csv_path, index=False)
        saved_paths.append(csv_path)

    return saved_paths


def constrain_pitch_range(df, frequency_column="frequency_hz", lowest_note=None, highest_note=None):
    """Set frequencies outside the configured range to NaN."""
    transformed = df.copy()

    if lowest_note is not None:
        transformed.loc[transformed[frequency_column] < lowest_note, frequency_column] = np.nan
    if highest_note is not None:
        transformed.loc[transformed[frequency_column] > highest_note, frequency_column] = np.nan

    return transformed


def replace_zeroes_with_nan(df):
    """Replace zero values with NaN so interpolation treats them as gaps."""
    return df.replace(0, np.nan)


def interpolate_missing_values(df, method="linear", fill_value=0):
    """Interpolate missing numeric values and fill any remaining gaps."""
    return df.interpolate(method=method).fillna(fill_value)


TRANSFORMATIONS = {
    "constrain_pitch_range": constrain_pitch_range,
    "replace_zeroes_with_nan": replace_zeroes_with_nan,
    "interpolate_missing_values": interpolate_missing_values,
}


def apply_transformations(data_frames, transformations):
    """Apply selected transformations to each dataframe.

    Each transformation can be either:
    - a string key from ``TRANSFORMATIONS``
    - a tuple of ``(name, kwargs)`` where ``name`` is a key from ``TRANSFORMATIONS``
    - a callable that accepts and returns a dataframe
    """
    transformed_frames = []

    for df in data_frames:
        transformed = df.copy()
        for transformation in transformations:
            kwargs = {}
            if isinstance(transformation, tuple):
                name, kwargs = transformation
                transform = TRANSFORMATIONS[name]
            elif isinstance(transformation, str):
                transform = TRANSFORMATIONS[transformation]
            else:
                transform = transformation

            transformed = transform(transformed, **kwargs)
        transformed_frames.append(transformed)

    return transformed_frames
