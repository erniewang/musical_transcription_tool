"""Reusable melody extraction steps."""

from tools.audio_modifiers import high_pass, split_audio
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


def apply_prefilters(audio, sr, prefilters):
    """Apply enabled prefilters in order and keep the result in memory."""
    processed_audio = audio
    enabled_filters = [item for item in prefilters if item.get("enabled", True)]

    if not enabled_filters:
        return processed_audio

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
            **filter_params,
        )

    return processed_audio


def prepare_audio(input_audio_path, prefilters):
    """Load input audio and run enabled prefilters in memory."""
    audio, sr = load_audio_mono(input_audio_path)
    processed_audio = apply_prefilters(audio, sr, prefilters)
    return processed_audio, sr


def split_into_sections(audio, sr, timestamps_seconds):
    """Split an in-memory audio array into configured sections."""
    return split_audio(audio, sr, timestamps_seconds)


def sections_from_audio(audio, sr, split_enabled, timestamps_seconds):
    """Return section dictionaries for either split audio or one whole-file section."""
    if split_enabled:
        return split_into_sections(audio, sr, timestamps_seconds)

    return [
        {
            "section_index": 1,
            "start_seconds": 0,
            "end_seconds": len(audio) / sr,
            "audio": audio,
            "sr": sr,
        }
    ]


def run_pitch_models_on_audio(audio, sr, model_configs, pitch_range_hz):
    """Run enabled PitchLab models on an in-memory audio array."""
    import pitchlab

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
