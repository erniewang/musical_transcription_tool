from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "pitchlab.model_folders"

import warnings
from typing import Any

import numpy as np

from ..utils import clean_model_parameters, coerce_frequency_range, f0_dataframe, normalize_audio_array


MODEL_NAME = "fcpe"
ACCEPTED_PARAMETERS = {
    "hop_ms",
    "device",
    "decoder_mode",
    "threshold",
    "interp_uv",
    "target_sample_rate",
    "backend_min_frequency",
}


def run_fcpe(input_audio, parameters=None):
    params = clean_model_parameters(MODEL_NAME, parameters, ACCEPTED_PARAMETERS)

    sample_rate = int(params["sample_rate"])
    hop_ms = float(params.get("hop_ms", 10.0))
    target_sample_rate = int(params.get("target_sample_rate", 16000))
    device = str(params.get("device", "cpu"))
    decoder_mode = str(params.get("decoder_mode", "local_argmax"))
    threshold = float(params.get("threshold", 0.006))
    interp_uv = bool(params.get("interp_uv", False))
    backend_min = float(params.get("backend_min_frequency", 80.0))
    rng = coerce_frequency_range(params, default=(80.0, 880.0))

    audio = normalize_audio_array(input_audio)
    estimated_length = len(audio)
    if sample_rate != target_sample_rate:
        estimated_length = int(np.ceil(len(audio) * target_sample_rate / float(sample_rate)))
    hop_size = max(1, int(round(target_sample_rate * hop_ms / 1000.0)))
    target_length = max(1, int(np.ceil(estimated_length / hop_size)))
    times = np.arange(target_length, dtype=float) * hop_size / float(target_sample_rate)

    f0_min = max(float(rng.low_hz), backend_min)
    f0_max = float(rng.high_hz)
    if f0_max <= f0_min:
        warnings.warn(
            (
                f"fcpe cannot run the requested effective range {rng.as_tuple()} Hz because "
                f"the bundled backend minimum is {backend_min:g} Hz; returning unvoiced output."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return f0_dataframe(
            times=times,
            frequency_hz=np.zeros(target_length, dtype=float),
            confidence=np.zeros(target_length, dtype=float),
            model=MODEL_NAME,
            freq_range=rng,
            extra={"backend_min_hz": backend_min},
        )

    import torch
    import librosa
    from torchfcpe import spawn_bundled_infer_model

    if sample_rate != target_sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
        target_length = max(1, int(np.ceil(len(audio) / hop_size)))
        times = np.arange(target_length, dtype=float) * hop_size / float(target_sample_rate)

    audio_tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32)).float().unsqueeze(0).unsqueeze(-1).to(device)
    model = spawn_bundled_infer_model(device=device)

    with torch.inference_mode():
        result = model.infer(
            audio_tensor,
            sr=target_sample_rate,
            decoder_mode=decoder_mode,
            threshold=threshold,
            f0_min=f0_min,
            f0_max=f0_max,
            interp_uv=interp_uv,
            output_interp_target_length=target_length,
        )

    f0 = _extract_f0_array(result)
    if f0.shape[0] != target_length:
        f0 = np.resize(f0, target_length)
    return f0_dataframe(
        times=times,
        frequency_hz=f0,
        confidence=np.where(f0 > 0.0, 1.0, 0.0),
        model=MODEL_NAME,
        freq_range=rng,
        extra={"backend_min_hz": backend_min},
    )


def _extract_f0_array(result: Any) -> np.ndarray:
    if hasattr(result, "detach"):
        return result.detach().cpu().numpy().squeeze().astype(float).reshape(-1)
    if isinstance(result, dict):
        for key in ("f0", "frequency", "frequency_hz", "pitch", "pitch_hz"):
            if key in result:
                return _extract_f0_array(result[key])
    if isinstance(result, (tuple, list)):
        for item in result:
            try:
                arr = _extract_f0_array(item)
            except ValueError:
                continue
            if arr.size:
                return arr
    arr = np.asarray(result)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float).squeeze().reshape(-1)
    raise ValueError(f"FCPE returned an unsupported result type: {type(result).__name__}")


if __name__ == "__main__":
    from pitchlab.model_folders import run_model_smoke_test

    run_model_smoke_test(run_fcpe, MODEL_NAME)
