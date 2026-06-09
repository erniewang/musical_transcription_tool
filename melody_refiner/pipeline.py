"""Reusable melody refinement steps."""

from tools.pitch_run_transforms import (
    apply_transformations,
    load_pitch_runs,
    save_pitch_runs,
)


def selected_transformations(transformation_pipeline):
    """Return enabled transformations in the format expected by apply_transformations."""
    selected = []

    for transformation in transformation_pipeline:
        if not transformation.get("enabled", True):
            continue

        name = transformation["name"]
        params = transformation.get("params", {})
        selected.append((name, params) if params else name)

    return selected


def load_runs(input_dir, combine_sections=True, time_column="time"):
    """Load pitch-run CSVs."""
    return load_pitch_runs(
        input_dir,
        combine_sections=combine_sections,
        time_column=time_column,
    )

def refine_runs(data_frames, transformation_pipeline):
    """Apply enabled refinement transformations."""
    return apply_transformations(
        data_frames,
        selected_transformations(transformation_pipeline),
    )


def save_runs(data_frames, output_dir, model_column="model"):
    """Save refined pitch runs."""
    return save_pitch_runs(data_frames, output_dir, filename_column=model_column)


def plot_runs(data_frames, time_column="time", frequency_column="frequency_hz"):
    """Plot model estimates on one shared axis."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 8), dpi=90)
    cmap = plt.get_cmap("tab20")

    for index, data_frame in enumerate(data_frames):
        model_name = data_frame["model"].iloc[0] if "model" in data_frame else f"run {index + 1}"
        data_frame.plot(
            x=time_column,
            y=frequency_column,
            ax=ax,
            color=cmap(index % 20),
            alpha=0.5,
            label=model_name,
        )

    return fig, ax


def display_sonified_runs(data_frames, sample_rate=44100):
    """Display sonified pitch tracks for each model in a notebook."""
    from IPython.display import Audio, Markdown, display
    from pitchlab.sonify import sonify_f0_dataframe

    for data_frame in data_frames:
        model_name = data_frame["model"].iloc[0] if "model" in data_frame else "pitch run"
        resulting_audio = sonify_f0_dataframe(data_frame)

        display(Markdown(f"### {model_name}"))
        display(Audio(resulting_audio, rate=sample_rate))
