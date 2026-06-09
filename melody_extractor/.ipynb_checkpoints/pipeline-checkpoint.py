"""Reusable melody extraction steps."""

from pathlib import Path

from tools.audio_modifiers import high_pass, segment_audio
from tools.pitch_run_io import export_pitch_runs


FILTERS = {
    "high_pass": high_pass,
}


def enabled_names(items):
    """Return the names of enabled config entries."""
    return [item["name"] for item in items if item.get("enabled", True)]


def load_audio_mono(audio_path):
    """Load audio as mono while preserving its original sample rate."""
    import librosa

    return librosa.load(audio_path, sr=None, mono=True)


def apply_prefilters(audio, sr, prefilters, output_path):
    """Apply enabled prefilters in order and write the resulting audio."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_audio = audio
    enabled_filters = [item for item in prefilters if item.get("enabled", True)]

    if not enabled_filters:
        import soundfile as sf

        sf.write(output_path, processed_audio, sr)
        return processed_audio, output_path

    for filter_config in enabled_filters:
        filter_name = filter_config["name"]
        filter_params = filter_config.get("params", {})
        try:
            filter_fn = FILTERS[filter_name]
        except KeyError as exc:
            known_filters = ", ".join(sorted(FILTERS))
            raise KeyError(f"Unknown prefilter '{filter_name}'. Known filters: {known_filters}") from exc

        processed_audio = filter_fn(
            audio=processed_audio,
            sr=sr,
            output_path=output_path,
            **filter_params,
        )

    return processed_audio, output_path


def prepare_audio(input_audio_path, prefilters, filtered_audio_path):
    """Load input audio, run enabled prefilters, and return the filtered file path."""
    audio, sr = load_audio_mono(input_audio_path)
    processed_audio, processed_path = apply_prefilters(audio, sr, prefilters, filtered_audio_path)
    return processed_audio, sr, processed_path


def split_into_sections(audio_path, timestamps_seconds, output_dir, prefix="section"):
    """Split an audio file into configured sections and return section file paths."""
    return segment_audio(
        audio_path=audio_path,
        time_stamps=timestamps_seconds,
        output_dir=output_dir,
        prefix=prefix,
    )


def selected_section_path(section_paths, section_number):
    """Return the configured 1-based section path."""
    if section_number < 1 or section_number > len(section_paths):
        raise IndexError(f"section_number must be between 1 and {len(section_paths)}")
    return section_paths[section_number - 1]


def run_pitch_models(audio_path, model_configs, pitch_range_hz):
    """Run enabled PitchLab models and return their pitch-run DataFrames."""
    import pitchlab
    from pitchlab.audio_io import load_audio

    audio, sr = load_audio(audio_path)
    results = []
    failures = []

    for model_name in enabled_names(model_configs):
        try:
            print(f"attempting to run {model_name}")
            model = pitchlab.get_model(model_name)
            df = model.predict(audio, sr, range=pitch_range_hz)
            results.append(df)
        except Exception as exc:
            failures.append((model_name, exc))
            print(f"{model_name} failed: {exc}")

    return results, failures


def export_results(data_frames, output_dir, source_name):
    """Export pitch-run DataFrames to CSV."""
    return export_pitch_runs(data_frames, output_dir, source_name)
