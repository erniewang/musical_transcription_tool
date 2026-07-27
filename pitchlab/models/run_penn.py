from __future__ import annotations

import numpy as np

from ..utils import clean_model_parameters, coerce_frequency_range, f0_dataframe, normalize_audio_array


MODEL_NAME = "penn"
ACCEPTED_PARAMETERS = {
    "hop_ms",
    "checkpoint",
    "batch_size",
    "center",
    "decoder",
    "interp_unvoiced_at",
    "gpu",
}


def run_penn(input_audio, parameters=None):
    params = clean_model_parameters(MODEL_NAME, parameters, ACCEPTED_PARAMETERS)
    try:
        import torch
        import penn
    except FileNotFoundError as exc:
        raise RuntimeError(
            "PENN failed while loading torbi binaries for the installed torch/CUDA build. "
            "This is an environment mismatch, not an audio input problem. Reinstall a torbi/PENN "
            "build matching torch, or exclude 'penn' in run_all_models."
        ) from exc

    sample_rate = int(params["sample_rate"])
    hop_ms = float(params.get("hop_ms", 10.0))
    checkpoint = params.get("checkpoint")
    batch_size = int(params.get("batch_size", 2048))
    center = str(params.get("center", "half-hop"))
    decoder = str(params.get("decoder", "viterbi"))
    interp_unvoiced_at = params.get("interp_unvoiced_at", 0.065)
    gpu = params.get("gpu")
    rng = coerce_frequency_range(params, default=(30.0, 1000.0))

    audio = normalize_audio_array(input_audio).astype(np.float32)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    pitch, periodicity = penn.from_audio(
        audio_tensor,
        sample_rate,
        hopsize=hop_ms / 1000.0,
        fmin=float(rng.low_hz),
        fmax=float(rng.high_hz),
        checkpoint=checkpoint,
        batch_size=batch_size,
        center=center,
        decoder=decoder,
        interp_unvoiced_at=interp_unvoiced_at,
        gpu=gpu,
    )
    pitch_np = pitch.detach().cpu().numpy().squeeze().astype(float).reshape(-1)
    periodicity_np = periodicity.detach().cpu().numpy().squeeze().astype(float).reshape(-1)
    times = np.arange(len(pitch_np), dtype=float) * hop_ms / 1000.0
    return f0_dataframe(times=times, frequency_hz=pitch_np, confidence=periodicity_np, model=MODEL_NAME, freq_range=rng)
