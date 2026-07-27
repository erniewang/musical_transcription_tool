"""Report on and save refined pitch runs."""

from __future__ import annotations

from pathlib import Path


def print_pitch_run_summary(data_frames, time_column="time", frequency_column="frequency_hz"):
    """Print a compact summary for loaded or refined pitch runs."""
    for index, data_frame in enumerate(data_frames, start=1):
        name = _model_name(data_frame, fallback=f"run {index}")
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


def save_pitch_run_csv(data_frame, output_path: str | Path) -> Path:
    """Save one pitch run as CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(output_path, index=False)
    return output_path


def _model_name(data_frame, fallback="pitch run"):
    if "model" in data_frame.columns and not data_frame.empty:
        return str(data_frame["model"].iloc[0])
    return str(data_frame.attrs.get("source_stem", fallback))
