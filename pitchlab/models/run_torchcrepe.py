from __future__ import annotations

import numpy as np

from ..utils import clean_model_parameters, coerce_frequency_range, f0_dataframe, hop_length_from_ms, normalize_audio_array, require_dependency


MODEL_NAME = "torchcrepe"
ACCEPTED_PARAMETERS = {"hop_ms", "model_size", "model", "batch_size", "device", "min_periodicity", "median_filter"}


def run_torchcrepe(input_audio, parameters=None):
    params = clean_model_parameters(MODEL_NAME, parameters, ACCEPTED_PARAMETERS)
    torch = require_dependency("torch", "pip install torch")
    torchcrepe = require_dependency("torchcrepe", "pip install torchcrepe")

    sample_rate = int(params["sample_rate"])
    hop_ms = float(params.get("hop_ms", 10.0))
    model_size = params.get("model", params.get("model_size", "full"))
    batch_size = int(params.get("batch_size", 2048))
    device = params.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    min_periodicity = params.get("min_periodicity", 0.21)
    median_filter = int(params.get("median_filter", 3))
    rng = coerce_frequency_range(params)

    audio = normalize_audio_array(input_audio)
    hop_length = hop_length_from_ms(sample_rate, hop_ms)
    tensor = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)

    pitch, periodicity = torchcrepe.predict(
        tensor,
        sample_rate,
        hop_length,
        rng.low_hz,
        rng.high_hz,
        model_size,
        batch_size=batch_size,
        device=device,
        return_periodicity=True,
    )
    if median_filter and median_filter > 1:
        periodicity = torchcrepe.filter.median(periodicity, median_filter)
    if min_periodicity is not None:
        pitch = torchcrepe.threshold.At(float(min_periodicity))(pitch, periodicity)

    pitch_np = pitch.squeeze().detach().cpu().numpy()
    periodicity_np = periodicity.squeeze().detach().cpu().numpy()
    times = np.arange(len(pitch_np), dtype=float) * hop_length / float(sample_rate)
    return f0_dataframe(
        times=times,
        frequency_hz=pitch_np,
        confidence=periodicity_np,
        model=MODEL_NAME,
        freq_range=rng,
        extra={"backend_model": model_size},
    )
