"""Configuration for the melody refinement notebook.

Toggle transformations here instead of editing the notebook workflow.
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

PITCH_RUN_INPUT_DIR = PROJECT_ROOT / "experiments" / "pitch_runs"
REFINED_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "pitch_runs_refined"

TIME_COLUMN = "time"
FREQUENCY_COLUMN = "frequency_hz"
MODEL_COLUMN = "model"

LOWEST_NOTE_HZ = 137
HIGHEST_NOTE_HZ = 515

TRANSFORMATION_PIPELINE = [
    {
        "name": "constrain_pitch_range",
        "enabled": True,
        "params": {
            "frequency_column": FREQUENCY_COLUMN,
            "lowest_note": LOWEST_NOTE_HZ,
            "highest_note": HIGHEST_NOTE_HZ,
        },
    },
    {
        "name": "replace_zeroes_with_nan",
        "enabled": True,
        "params": {},
    },
    {
        "name": "interpolate_missing_values",
        "enabled": True,
        "params": {
            "method": "linear",
            "fill_value": 0,
        },
    },
]
