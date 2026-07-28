"""Pitch (f0) pipeline: load/preprocess once, then per model extract → postprocess → write."""

from __future__ import annotations

import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from helpers import PROJECT_ROOT, log, log_error, log_success
from pitchlab import available_models, run_model
from postprocessing import (
    print_pitch_run_summary,
    save_pitch_run_csv,
    score_from_pitch_run,
    write_score_pdf,
    write_sonified_audio,
)
from processing import PITCH_POSTPROCESSING_OPS, load_input_audio, load_settings, preprocess_audio


def run(settings):
    models = settings.extraction.models
    known = set(available_models())
    unknown = [name for name in models if name not in known]
    audio_in = settings.input
    extraction = settings.extraction
    piece_name = audio_in.audio_file.stem
    output_dir = settings.output.directory or PROJECT_ROOT / "output" / piece_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    try:
        audio, sample_rate = load_input_audio(settings)
    except Exception as error:
        log_error(f"load failed: {error}")
        return results

    try:
        audio = preprocess_audio(audio, sample_rate, settings)
    except Exception as error:
        log_error(f"preprocessing failed: {error}")
        return results

    for model_name in models:
        start = time.perf_counter()
        log(f"Model: {model_name}")

        try:
            log(f"  Extracting: {model_name}")
            pitch_run = run_model(
                model_name,
                audio,
                {
                    "sample_rate": sample_rate,
                    "hop_ms": extraction.hop_ms,
                    "fmin": extraction.min_pitch_hz,
                    "fmax": extraction.max_pitch_hz,
                },
            )
            pitch_run["model"] = model_name
            pitch_run["time"] += audio_in.start_seconds
        except Exception as error:
            log_error(f"  extraction failed for {model_name}: {error}")
            continue

        try:
            post_steps = sorted(
                (
                    (name, getattr(settings.postprocessing, name), fn)
                    for name, fn in PITCH_POSTPROCESSING_OPS.items()
                ),
                key=lambda step: step[1].order,
            )
            for name, cfg, fn in post_steps:
                if not cfg.enabled:
                    continue
                log(f"  Postprocessing: {name}")
                pitch_run = fn(pitch_run, cfg)
            pitch_run = pitch_run.assign(frequency_hz=pitch_run["frequency_hz"].fillna(0.0))
            print_pitch_run_summary([pitch_run])
        except Exception as error:
            log_error(f"  postprocessing failed for {model_name}: {error}")
            continue

        try:
            model_dir = output_dir / model_name
            files = {
                "audio": write_sonified_audio(
                    pitch_run,
                    model_dir / f"{model_name}_sonified.{settings.output.sonify_format}",
                    sample_rate=sample_rate,
                ),
                "csv": save_pitch_run_csv(pitch_run, model_dir / f"{model_name}_refined.csv"),
                "pdf": write_score_pdf(
                    score_from_pitch_run(pitch_run, piece_name),
                    model_dir / f"{model_name}.pdf",
                ),
            }
            results[model_name] = files
            log_success(f"  wrote {', '.join(path.name for path in files.values())} -> {model_dir}")
            log_success(f"Model {model_name} took {time.perf_counter() - start:.6f} seconds to complete.")
        except Exception as error:
            log_error(f"  output failed for {model_name}: {error}")

    return results


if __name__ == "__main__":
    from processing import PITCH_SETTINGS_PATH

    settings_path = sys.argv[1] if len(sys.argv) > 1 else PITCH_SETTINGS_PATH
    run(load_settings(settings_path))
