from __future__ import annotations

import numpy as np

from .base import BasePitchModel, f0_dataframe, hop_length_from_ms, require_dependency


class TorchCrepeModel(BasePitchModel):
    name = "torchcrepe"
    aliases = ("torch-crepe", "torch_crepe")
    supports_native_range = True
    output_type = "f0"

    def predict(
        self,
        audio,
        sample_rate: int,
        *,
        hop_ms: float = 10.0,
        model_size: str = "full",
        model: str | None = None,
        batch_size: int = 2048,
        device: str | None = None,
        min_periodicity: float | None = 0.21,
        median_filter: int = 3,
        freq_range=None,
        fmin=None,
        fmax=None,
        **kwargs,
    ):
        torch = require_dependency("torch", "pip install torch")
        torchcrepe = require_dependency("torchcrepe", "pip install torchcrepe")

        x = self.clean_audio(audio)
        rng = self.resolve_range(freq_range=freq_range, fmin=fmin, fmax=fmax, **kwargs)
        hop_length = hop_length_from_ms(sample_rate, hop_ms)
        if model is not None:
            model_size = model
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

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
            periodicity = torchcrepe.filter.median(periodicity, int(median_filter))

        if min_periodicity is not None:
            pitch = torchcrepe.threshold.At(float(min_periodicity))(pitch, periodicity)

        pitch_np = pitch.squeeze().detach().cpu().numpy()
        periodicity_np = periodicity.squeeze().detach().cpu().numpy()
        times = np.arange(len(pitch_np), dtype=float) * hop_length / float(sample_rate)

        return f0_dataframe(
            times=times,
            frequency_hz=pitch_np,
            confidence=periodicity_np,
            model=self.name,
            freq_range=rng,
            extra={"backend_model": model_size},
        )
