"""Notebook helpers for loading, saving, plotting, and sonifying pitch runs."""

from pathlib import Path
import re

import pandas as pd


def list_pitch_run_files(input_dir, suffix=".csv"):
    """Return pitch-run CSV files under ``input_dir``."""
    input_path = Path(input_dir)
    return [
        file
        for file in sorted(input_path.rglob(f"*{suffix}"))
        if file.is_file() and file.suffix == suffix
    ]


def load_pitch_runs(input_dir, combine_sections=True, time_column="time"):
    """Load pitch-run CSV files.

    When ``combine_sections`` is true, section CSVs are grouped by model and
    concatenated into one continuous timeline per model.
    """
    pitch_run_files = list_pitch_run_files(input_dir)
    if not combine_sections:
        return [_read_pitch_run(file) for file in pitch_run_files]

    grouped_sections = {}
    for file in pitch_run_files:
        data_frame = _read_pitch_run(file)
        model_name = _model_name(data_frame, file)
        grouped_sections.setdefault(model_name, []).append(
            {
                "path": file,
                "section_number": _section_number(file),
                "data_frame": data_frame,
            }
        )

    data_frames = []
    for model_name, sections in sorted(grouped_sections.items()):
        ordered_sections = sorted(
            sections,
            key=lambda section: (section["section_number"], section["path"].name),
        )
        merged = _combine_sections(ordered_sections, time_column)
        merged.attrs["source_paths"] = [
            section["path"] for section in ordered_sections
        ]
        merged.attrs["source_stem"] = model_name
        data_frames.append(merged)

    return data_frames


def save_pitch_runs(data_frames, output_dir, model_column="model", suffix="_refined"):
    """Save pitch-run DataFrames and return written CSV paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for index, data_frame in enumerate(data_frames, start=1):
        model_name = _model_name(data_frame, None, fallback=f"pitch_run_{index}")
        filename_stem = data_frame.attrs.get("source_stem", model_name)
        filename = f"{_safe_name(filename_stem)}{suffix}.csv"

        if model_column in data_frame.columns and not data_frame.empty:
            model_output_path = output_path / _safe_name(data_frame[model_column].iloc[0])
        else:
            model_output_path = output_path

        model_output_path.mkdir(parents=True, exist_ok=True)
        csv_path = model_output_path / filename
        data_frame.to_csv(csv_path, index=False)
        saved_paths.append(csv_path)

    return saved_paths


def print_pitch_run_files(input_dir):
    """Print CSV paths relative to ``input_dir``."""
    input_path = Path(input_dir)
    for file in list_pitch_run_files(input_path):
        print(file.relative_to(input_path))


def print_pitch_run_summary(data_frames, time_column="time", frequency_column="frequency_hz"):
    """Print a compact summary for loaded or refined pitch runs."""
    for index, data_frame in enumerate(data_frames, start=1):
        name = _model_name(data_frame, None, fallback=f"run {index}")
        row_count = len(data_frame)
        if time_column in data_frame and row_count:
            duration = float(data_frame[time_column].max())
            time_text = f", {duration:.2f}s"
        else:
            time_text = ""

        if frequency_column in data_frame:
            missing = int(data_frame[frequency_column].isna().sum())
            missing_text = f", {missing} missing pitches"
        else:
            missing_text = ""

        print(f"{name}: {row_count} rows{time_text}{missing_text}")


def plot_pitch_runs(data_frames, time_column="time", frequency_column="frequency_hz"):
    """Plot every pitch run on one shared axis."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 8), dpi=90)
    cmap = plt.get_cmap("tab20")

    for index, data_frame in enumerate(data_frames):
        name = _model_name(data_frame, None, fallback=f"run {index + 1}")
        data_frame.plot(
            x=time_column,
            y=frequency_column,
            ax=ax,
            color=cmap(index % 20),
            alpha=0.5,
            label=name,
        )

    ax.set_xlabel(time_column)
    ax.set_ylabel(frequency_column)
    ax.grid(True, alpha=0.2)
    return fig, ax


def display_sonified_pitch_runs(data_frames, sample_rate=44100):
    """Display a sonified audio player for each pitch run."""
    from IPython.display import Audio, Markdown, display
    from pitchlab.sonify import sonify_f0_dataframe

    for index, data_frame in enumerate(data_frames, start=1):
        name = _model_name(data_frame, None, fallback=f"run {index}")
        display(Markdown(f"### {name}"))
        display(Audio(sonify_f0_dataframe(data_frame), rate=sample_rate))


def _read_pitch_run(file):
    data_frame = pd.read_csv(file)
    data_frame.attrs["source_path"] = file
    data_frame.attrs["source_stem"] = file.stem
    return data_frame


def _section_number(path):
    match = re.search(r"_section_(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def _model_name(data_frame, path, fallback="pitch run"):
    if "model" in data_frame.columns and not data_frame.empty:
        return str(data_frame["model"].iloc[0])
    if path is not None:
        return path.parent.name
    return str(data_frame.attrs.get("source_stem", fallback))


def _combine_sections(sections, time_column):
    merged_sections = []
    current_offset = 0.0

    for section in sections:
        data_frame = section["data_frame"].copy()
        if time_column in data_frame.columns and not data_frame.empty:
            section_start = float(data_frame[time_column].iloc[0])
            data_frame[time_column] = data_frame[time_column] - section_start + current_offset
            current_offset = float(data_frame[time_column].iloc[-1]) + _time_step(
                data_frame,
                time_column,
            )

        merged_sections.append(data_frame)

    return pd.concat(merged_sections, ignore_index=True)


def _time_step(data_frame, time_column):
    if len(data_frame) < 2 or time_column not in data_frame:
        return 0

    diffs = data_frame[time_column].diff().dropna()
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.empty:
        return 0

    return float(positive_diffs.median())


def _safe_name(value):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return safe or "unnamed"
