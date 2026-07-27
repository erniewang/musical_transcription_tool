"""Heuristic transcriber entry point.

Load and preprocess audio once, then for every model:
extract -> postprocess -> write results.

Every tunable value lives in ``transcribe_settings.json``; this file is only wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path
import time

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pitchlab import available_models, run_model
from preprocessing import extract_harmonic_part, high_pass, load_audio, low_pass
from postprocessing import (
    constrain_pitch_range,
    filter_by_confidence,
    interpolate,
    print_pitch_run_summary,
    quantize_pitch_run,
    save_pitch_run_csv,
    score_from_pitch_run,
    write_score_pdf,
    write_sonified_audio,
)

from settings import (
    ExtractionSettings,
    PostprocessingSettings,
    PreprocessingSettings,
    Settings,
    load_settings,
)


PREPROCESSING_OPS = {
    "extract_harmonic_part": extract_harmonic_part,
    "high_pass_filter": high_pass,
    "low_pass_filter": low_pass,
}

POSTPROCESSING_OPS = {
    "confidence_filter": filter_by_confidence,
    "pitch_range_filter": constrain_pitch_range,
    "interpolate": interpolate,
    "quantize": quantize_pitch_run,
}

RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RESET = "\033[0m"
log = lambda message: print(message)
log_error = lambda message: print(f"{RED}{message}{RESET}")
log_warning = lambda message: print(f"{YELLOW}{message}{RESET}")
log_success = lambda message: print(f"{GREEN}{message}{RESET}")

def run_preprocessing(audio, sample_rate: int, settings: PreprocessingSettings):
    """Run enabled preprocessing steps sorted by ``order``."""
    for name, apply in sorted(PREPROCESSING_OPS.items(), key=lambda item: getattr(settings, item[0]).order):
        cfg = getattr(settings, name)
        if not cfg.enabled:
            continue
        log(f"Preprocessing: {name}")
        audio = apply(audio, sample_rate, cfg)
    return audio


def run_postprocessing(pitch_run, settings: PostprocessingSettings):
    """Run enabled postprocessing steps on one pitch run, sorted by ``order``."""
    for name, apply in sorted(POSTPROCESSING_OPS.items(), key=lambda item: getattr(settings, item[0]).order):
        cfg = getattr(settings, name)
        if not cfg.enabled:
            if name == "interpolate":
                # Range filtering marks excluded frames as NaN; use zero when
                # interpolation will not fill them.
                pitch_run = pitch_run.assign(frequency_hz=pitch_run["frequency_hz"].fillna(0.0))
            continue
        log(f"  Postprocessing: {name}")
        pitch_run = apply(pitch_run, cfg)
    return pitch_run


def run_extraction(
    audio,
    sample_rate: int,
    model_name: str,
    extraction: ExtractionSettings,
    *,
    start_seconds: float,
):
    """Run one pitch model and return its pitch-run DataFrame."""
    parameters = {
        "sample_rate": sample_rate,
        "hop_ms": extraction.hop_ms,
        "fmin": extraction.min_pitch_hz,
        "fmax": extraction.max_pitch_hz,
    }
    log(f"  Extracting: {model_name}")
    pitch_run = run_model(model_name, audio, parameters)
    pitch_run["model"] = model_name
    pitch_run["time"] += start_seconds
    return pitch_run


def write_model_result(pitch_run, output_dir: Path, *, sample_rate: int, sonify_format: str):
    """Write sonified audio, refined CSV, and notation PDF for one model.

    Sonification and CSV run before notation so a MuseScore failure still
    leaves the audio and data on disk.
    """
    model_name = str(pitch_run["model"].iloc[0])
    model_dir = output_dir / model_name
    files: dict[str, Path] = {}

    try:
        files["audio"] = write_sonified_audio(
            pitch_run, model_dir / f"{model_name}_sonified.{sonify_format}", sample_rate=sample_rate
        )
        files["csv"] = save_pitch_run_csv(pitch_run, model_dir / f"{model_name}_refined.csv")
    except Exception as error:
        log_warning(f"  unable to sonify {model_name}")
        log_error(f"  reason {error}")

    try:
        score = score_from_pitch_run(pitch_run)
        files["pdf"] = write_score_pdf(score, model_dir / f"{model_name}.pdf")
    except Exception as error:
        log_warning(f"  unable to write notation PDF for {model_name}")
        log_error(f"  reason {error}")

    written = ", ".join(path.name for path in files.values())
    if written:
        log_success(f"  wrote {written} -> {model_dir}")
    else:
        log_warning(f"  wrote nothing -> {model_dir}")
    return files


def transcribe_model(
    settings: Settings,
    model_name: str,
    audio,
    sample_rate: int,
    output_dir: Path,
) -> dict[str, Path]:
    """Extract, postprocess, and write results for one model."""
    log(f"Model: {model_name}")
    pitch_run = run_extraction(
        audio,
        sample_rate,
        model_name,
        settings.extraction,
        start_seconds=settings.input.start_seconds,
    )
    pitch_run = run_postprocessing(pitch_run, settings.postprocessing)
    print_pitch_run_summary([pitch_run])
    return write_model_result(
        pitch_run,
        output_dir,
        sample_rate=sample_rate,
        sonify_format=settings.output.sonify_format,
    )


def transcribe(settings: Settings) -> dict[str, dict[str, Path]]:

    """Transcribe one audio file with each selected model, one model at a time."""
    known = set(available_models())
    unknown = [name for name in settings.extraction.models if name not in known]
    if unknown:
        raise ValueError(
            f"Unknown pitch model(s): {', '.join(unknown)}. "
            f"Available models: {', '.join(sorted(known))}"
        )
    audio_settings = settings.input
    output_dir = settings.output.directory or PACKAGE_ROOT / "output" / audio_settings.audio_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = load_audio(
        audio_settings.audio_file,
        start=audio_settings.start_seconds,
        end=audio_settings.end_seconds,
        sample_rate=audio_settings.sample_rate,
    )

    audio = run_preprocessing(audio, sample_rate, settings.preprocessing)

    results: dict[str, dict[str, Path]] = {}
    for model_name in settings.extraction.models:
        try:
            start_time = time.perf_counter()
            results[model_name] = transcribe_model(
                settings, model_name, audio, sample_rate, output_dir
            )
            elapsed = time.perf_counter() - start_time
            log_success(f"Model {model_name} took {elapsed:.6f} seconds to complete.")
        except Exception as error:
            log_error(f"unable to successfully run {model_name}")
            log_error(f"reason {error}")
    return results


if __name__ == "__main__":
    transcribe(load_settings())
