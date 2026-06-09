"""Configuration for the melody extraction notebook.

Change this file when you want to turn extraction steps on or off.
The notebook should mostly call the pipeline helpers.
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

AUDIO_FILE_NAME = "uzbek_dari.mp3"
INPUT_AUDIO_PATH = PROJECT_ROOT / "experiments" / "audio_samples" / AUDIO_FILE_NAME

SECTION_PREFIX = "section"
PITCH_RUN_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "pitch_runs"

# Each entry can be enabled or disabled independently.
PREFILTERS = [
    {
        "name": "high_pass",
        "enabled": True,
        "params": {
            "cutoff_hz": 147,
            "order": 10,
        },
    },
]

# Use -1 for the final timestamp to mean "until the end of the file".
SPLIT_AUDIO = True
SECTION_TIMESTAMPS_SECONDS = [0, 27, 43, 56, 78, 89, -1]

PITCH_MODELS = [
    {"name": "fcpe", "enabled": True},
    {"name": "essentia-melodia", "enabled": True},
    {"name": "pyworld-dio", "enabled": True},
    {"name": "librosa-yin", "enabled": True},
    {"name": "yaapt", "enabled": True},
    {"name": "librosa-pyin", "enabled": True},
    {"name": "torchcrepe", "enabled": True},
    {"name": "crepe", "enabled": False},
]

PITCH_RANGE_HZ = (200, 1007)
