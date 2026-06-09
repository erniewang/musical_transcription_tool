"""CSV export helpers for extracted pitch runs."""

from pathlib import Path
import re


def _safe_filename_part(value):
    """Return a filesystem-friendly name fragment."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def export_pitch_runs(data_frames, output_dir, source_name, model_column="model"):
    """Write one CSV per model dataframe and return the saved paths.

    Parameters
    ----------
    data_frames:
        Iterable of pandas DataFrames produced by PitchLab models.
    output_dir:
        Directory where CSV files should be written. Each model gets a
        subdirectory inside this directory.
    source_name:
        Original audio filename or stem used as the CSV filename prefix.
    model_column:
        Column containing the model name. Defaults to ``"model"``.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_stem = _safe_filename_part(Path(source_name).stem)
    saved_paths = []

    for df in data_frames:
        if df.empty:
            continue
        if model_column not in df.columns:
            raise KeyError(f"Missing required model column: {model_column}")

        model_name = _safe_filename_part(df[model_column].iloc[0])
        model_output_path = output_path / model_name
        model_output_path.mkdir(parents=True, exist_ok=True)

        csv_path = model_output_path / f"{source_stem}_{model_name}.csv"
        df.to_csv(csv_path, index=False)
        saved_paths.append(csv_path)

    return saved_paths
