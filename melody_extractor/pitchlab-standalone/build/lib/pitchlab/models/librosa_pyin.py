from __future__ import annotations

import numpy as np

from .base import BasePitchModel, f0_dataframe, hop_length_from_ms, require_dependency


class LibrosaPyinModel(BasePitchModel):
    name = "librosa-pyin"
    aliases = ("pyin", "librosa_pyin")
    supports_native_range = True
    output_type = "f0"

    def predict(
        self,
        audio,
        sample_rate: int,
        *,
        hop_ms: float = 10.0,
        frame_length: int = 2048,
        resolution: float = 0.1,
        freq_range=None,
        fmin=None,
        fmax=None,
        **kwargs,
    ):
        librosa = require_dependency("librosa", "pip install librosa")
        x = self.clean_audio(audio)
        rng = self.resolve_range(freq_range=freq_range, fmin=fmin, fmax=fmax, **kwargs)
        hop_length = hop_length_from_ms(sample_rate, hop_ms)

        f0, voiced_flag, voiced_prob = librosa.pyin(
            x,
            fmin=rng.low_hz,
            fmax=rng.high_hz,
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            resolution=resolution,
        )
        f0 = np.where(np.asarray(voiced_flag, dtype=bool), f0, 0.0)
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sample_rate, hop_length=hop_length)
        return f0_dataframe(
            times=times,
            frequency_hz=f0,
            confidence=voiced_prob,
            model=self.name,
            freq_range=rng,
        )
