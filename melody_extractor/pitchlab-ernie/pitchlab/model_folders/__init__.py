"""Backend-specific pitchlab runner functions."""

from __future__ import annotations


def run_model_smoke_test(run_model, model_name: str, *, sample_rate: int = 16000, duration: float = 0.5) -> None:
    """Run a tiny default-parameter smoke test for a model runner."""
    import numpy as np

    t = np.arange(int(sample_rate * duration), dtype=np.float32) / float(sample_rate)
    audio = 0.2 * np.sin(2.0 * np.pi * 440.0 * t)
    result = run_model(audio, {"sample_rate": sample_rate})
    print(result.head())
    print(f"{model_name}: rows={len(result)} columns={list(result.columns)}")
