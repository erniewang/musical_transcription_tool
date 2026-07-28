"""Rhythm pipeline: load/preprocess once, then per model extract → postprocess → write."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from helpers import PROJECT_ROOT, log_error
from processing import RHYTHM_POSTPROCESSING_OPS, RHYTHM_SETTINGS_PATH, load_input_audio, load_settings, preprocess_audio


def run(settings):
    models = settings.extraction.models
    piece_name = settings.input.audio_file.stem
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

    if not RHYTHM_POSTPROCESSING_OPS:
        raise ValueError(
            "rhythm postprocessing ops are not implemented yet "
            "(add functions under rhythm/ and wire them in RHYTHM_POSTPROCESSING_OPS)"
        )

    # Same shape as pitch_main: for each model extract → postprocess → write.
    # Fill this in when rhythm extractors exist.
    raise ValueError("rhythm extraction is not implemented yet")


if __name__ == "__main__":
    settings_path = sys.argv[1] if len(sys.argv) > 1 else RHYTHM_SETTINGS_PATH
    run(load_settings(settings_path))
