"""Entry point. Defaults to pitch; pass ``rhythm`` for rhythm_main."""

from __future__ import annotations

import sys

from processing import PITCH_SETTINGS_PATH, RHYTHM_SETTINGS_PATH, load_settings

if __name__ == "__main__":
    if sys.argv[1:2] == ["rhythm"]:
        from rhythm_main import run

        path = sys.argv[2] if len(sys.argv) > 2 else RHYTHM_SETTINGS_PATH
        run(load_settings(path))
    else:
        from pitch_main import run

        path = sys.argv[1] if len(sys.argv) > 1 else PITCH_SETTINGS_PATH
        run(load_settings(path))
