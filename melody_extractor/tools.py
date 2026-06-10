"""Small helpers for the melody extraction notebook."""

from pathlib import Path
import re
import shutil


def load_audio(audio_path):
    """Load audio as mono and keep its original sample rate."""
    import librosa

    return librosa.load(audio_path, sr=None, mono=True)


def seconds_text(seconds):
    """Format time as simple whole seconds for notebook labels."""
    return f"{round(seconds)} seconds"


def show_audio(audio, sr, title):
    """Display a titled audio player in a notebook."""
    try:
        from IPython.display import Audio, Markdown, display
    except ModuleNotFoundError:
        print(f"{title}: {seconds_text(len(audio) / sr)}")
        return

    display(Markdown(f"### {title} ({seconds_text(len(audio) / sr)})"))
    display(Audio(data=audio, rate=sr))


def split_audio(audio, sr, timestamps_seconds):
    """Split audio into sections using timestamps in seconds.

    Use -1 as the final timestamp to mean "until the end".
    """
    if len(timestamps_seconds) < 2:
        raise ValueError("Need at least a start and end timestamp.")

    duration_seconds = len(audio) / sr
    section_times = list(timestamps_seconds)
    if section_times[-1] == -1:
        section_times[-1] = duration_seconds

    sections = []
    for number, (start_seconds, end_seconds) in enumerate(
        zip(section_times, section_times[1:]),
        start=1,
    ):
        if end_seconds <= start_seconds:
            raise ValueError(
                f"Section {number} ends before it starts: "
                f"{start_seconds} to {end_seconds}."
            )

        start_sample = max(0, int(start_seconds * sr))
        end_sample = min(len(audio), int(end_seconds * sr))
        sections.append(
            {
                "number": number,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "audio": audio[start_sample:end_sample],
                "sr": sr,
            }
        )

    return sections


def print_sections(sections):
    """Print a compact section summary."""
    for section in sections:
        start = seconds_text(section["start_seconds"])
        end = seconds_text(section["end_seconds"])
        print(f"section {section['number']}: {start} to {end}")


def show_sections(sections, title_prefix="Section"):
    """Display every section as a separate audio player."""
    for section in sections:
        title = (
            f"{title_prefix} {section['number']}: "
            f"{seconds_text(section['start_seconds'])} to "
            f"{seconds_text(section['end_seconds'])}"
        )
        show_audio(section["audio"], section["sr"], title)


def high_pass(audio, sr, cutoff_hz, order=5):
    """Apply a high-pass filter."""
    if cutoff_hz is None:
        return audio

    from scipy.signal import butter, sosfiltfilt

    _check_cutoff(cutoff_hz, sr, "high_pass")
    sos = butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, audio)


def low_pass(audio, sr, cutoff_hz, order=5):
    """Apply a low-pass filter."""
    if cutoff_hz is None:
        return audio

    from scipy.signal import butter, sosfiltfilt

    _check_cutoff(cutoff_hz, sr, "low_pass")
    sos = butter(order, cutoff_hz, btype="lowpass", fs=sr, output="sos")
    return sosfiltfilt(sos, audio)


def filter_audio(audio, sr, high_pass_hz=None, low_pass_hz=None, order=5):
    """Apply the filters that are set and return the filtered audio."""
    filtered = high_pass(audio, sr, high_pass_hz, order=order)
    filtered = low_pass(filtered, sr, low_pass_hz, order=order)
    return filtered


def filter_sections(sections, high_pass_hz=None, low_pass_hz=None, order=5):
    """Apply per-section filters and return new section dictionaries."""
    filtered_sections = []
    for index, section in enumerate(sections):
        filtered_audio = filter_audio(
            section["audio"],
            section["sr"],
            high_pass_hz=_value_for_section(high_pass_hz, index, sections, "high_pass_hz"),
            low_pass_hz=_value_for_section(low_pass_hz, index, sections, "low_pass_hz"),
            order=order,
        )
        filtered_section = dict(section)
        filtered_section["audio"] = filtered_audio
        filtered_sections.append(filtered_section)

    return filtered_sections


def run_pitch_model(section, model_name, pitch_range_hz):
    """Run one PitchLab model on one section."""
    import pitchlab

    model = pitchlab.get_model(model_name)
    return model.predict(section["audio"], section["sr"], range=pitch_range_hz)


def save_pitch_run(data_frame, output_dir, audio_name, section_number, model_name):
    """Save one model result as output/model/audio_section_model.csv."""
    model_folder = _safe_name(model_name)
    audio_stem = _safe_name(Path(audio_name).stem)
    file_model_name = _safe_name(model_name)

    csv_dir = Path(output_dir) / model_folder
    csv_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_dir / f"{audio_stem}_section_{section_number}_{file_model_name}.csv"
    data_frame.to_csv(csv_path, index=False)
    return csv_path


def empty_folder(folder_path):
    """Remove everything inside a folder without deleting the folder itself."""
    folder = Path(folder_path)
    if not folder.exists():
        return

    for path in folder.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _check_cutoff(cutoff_hz, sr, filter_name):
    nyquist_hz = sr / 2
    if cutoff_hz <= 0 or cutoff_hz >= nyquist_hz:
        raise ValueError(
            f"{filter_name} cutoff must be between 0 and {nyquist_hz:g} Hz. "
            f"Got {cutoff_hz}."
        )


def _value_for_section(value, index, sections, name):
    if value is None or isinstance(value, (int, float)):
        return value

    if len(value) != len(sections):
        raise ValueError(
            f"{name} needs {len(sections)} values, one for each section. "
            f"Got {len(value)}."
        )

    return value[index]


def _safe_name(value):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return safe or "unnamed"
