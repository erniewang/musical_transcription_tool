"""Run a set of pitch models over prepared audio, tolerating per-model failures."""

from __future__ import annotations

from typing import Any, Mapping

from .registry import available_models, model_specifications, run_model


def f0_model_names() -> list[str]:
    """Names of every registered model that outputs a frame-level f0 track."""
    return [spec["name"] for spec in model_specifications() if spec["output_type"] == "f0"]


def run_pitch_models(model_names, audio, parameters: Mapping[str, Any], start_seconds: float = 0.0):
    """Run each named model, returning one DataFrame per model that succeeded."""
    known = set(available_models())
    unknown = [name for name in model_names if name not in known]
    if unknown:
        raise ValueError(
            f"Unknown pitch model(s): {', '.join(unknown)}. "
            f"Available models: {', '.join(sorted(known))}"
        )

    pitch_runs = []
    for name in model_names:
        try:
            print(f"Running {name}...")
            pitch_run = run_model(name, audio, parameters)
            pitch_run["model"] = name
            pitch_run["time"] += start_seconds
            pitch_runs.append(pitch_run)
        except Exception as error:
            print(f"unable to successfully run {name}")
            print("reason", error)
    return pitch_runs
