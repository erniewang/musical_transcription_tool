"""CSV loading and dataframe transformation helpers for pitch runs."""

from pathlib import Path
import re

import numpy as np
import pandas as pd


def _section_number(path):
    match = re.search(r"_section_(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return 0


def _model_name_from_frame(data_frame, path):
    if "model" in data_frame.columns and not data_frame.empty:
        return str(data_frame["model"].iloc[0])
    return path.parent.name


def _time_step(data_frame, time_column):
    if len(data_frame) < 2 or time_column not in data_frame:
        return 0

    diffs = data_frame[time_column].diff().dropna()
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.empty:
        return 0

    return float(positive_diffs.median())


def _append_with_time_offset(sections, time_column):
    merged_sections = []
    current_offset = 0.0

    for section in sections:
        data_frame = section["data_frame"].copy()

        if time_column in data_frame.columns and not data_frame.empty:
            section_start = float(data_frame[time_column].iloc[0])
            data_frame[time_column] = data_frame[time_column] - section_start + current_offset
            current_offset = float(data_frame[time_column].iloc[-1]) + _time_step(data_frame, time_column)

        merged_sections.append(data_frame)

    return pd.concat(merged_sections, ignore_index=True)


def load_pitch_runs(input_dir, suffix=".csv", combine_sections=True, time_column="time"):
    """Load pitch-run CSV files.

    When ``combine_sections`` is true, section CSVs are grouped by model and
    concatenated into one DataFrame per model with section-local times offset
    into one continuous timeline.
    """
    input_path = Path(input_dir)
    pitch_run_files = [
        file
        for file in sorted(input_path.rglob(f"*{suffix}"))
        if file.suffix == suffix
    ]

    if not combine_sections:
        data_frames = []

        for file in pitch_run_files:
            data_frame = pd.read_csv(file)
            data_frame.attrs["source_path"] = file
            data_frame.attrs["source_stem"] = file.stem
            data_frames.append(data_frame)

        return data_frames

    grouped_sections = {}
    for file in pitch_run_files:
        data_frame = pd.read_csv(file)
        model_name = _model_name_from_frame(data_frame, file)
        grouped_sections.setdefault(model_name, []).append(
            {
                "path": file,
                "section_number": _section_number(file),
                "data_frame": data_frame,
            }
        )

    merged_data_frames = []
    for model_name, sections in sorted(grouped_sections.items()):
        sorted_sections = sorted(sections, key=lambda section: (section["section_number"], section["path"].name))
        merged = _append_with_time_offset(sorted_sections, time_column)
        merged.attrs["source_paths"] = [section["path"] for section in sorted_sections]
        merged.attrs["source_stem"] = model_name
        merged_data_frames.append(merged)

    return merged_data_frames


def save_pitch_runs(data_frames, output_dir, filename_column="model", suffix="_refined"):
    """Save transformed pitch-run DataFrames and return the written paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for index, df in enumerate(data_frames, start=1):
        if "source_stem" in df.attrs:
            filename = f"{df.attrs['source_stem']}{suffix}.csv"
        elif filename_column in df.columns and not df.empty:
            filename = f"{df[filename_column].iloc[0]}{suffix}.csv"
        else:
            filename = f"pitch_run_{index}{suffix}.csv"

        if filename_column in df.columns and not df.empty:
            model_output_path = output_path / str(df[filename_column].iloc[0])
        else:
            model_output_path = output_path

        model_output_path.mkdir(parents=True, exist_ok=True)
        csv_path = model_output_path / filename
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
