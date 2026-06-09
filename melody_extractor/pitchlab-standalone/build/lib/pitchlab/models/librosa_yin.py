from __future__ import annotations

import numpy as np

from .base import BasePitchModel, f0_dataframe, hop_length_from_ms, require_dependency


class LibrosaYinModel(BasePitchModel):
    name = "librosa-yin"
    aliases = ("yin", "librosa_yin")
    supports_native_range = True
    output_type = "f0"

    def predict(
        self,
        audio,
        sample_rate: int,
        *,
        hop_ms: float = 10.0,
        frame_length: int = 2048,
        trough_threshold: float = 0.1,
        freq_range=None,
        fmin=None,
        fmax=None,
        **kwargs,
    ):
        librosa = require_dependency("librosa", "pip install librosa")
        x = self.clean_audio(audio)
        rng = self.resolve_range(freq_range=freq_range, fmin=fmin, fmax=fmax, **kwargs)
        hop_length = hop_length_from_ms(sample_rate, hop_ms)

        f0 = librosa.yin(
            x,
            fmin=rng.low_hz,
            fmax=rng.high_hz,
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            trough_threshold=trough_threshold,
        )
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sample_rate, hop_length=hop_length)
        confidence = np.where(np.isfinite(f0), 1.0, 0.0)
        return f0_dataframe(
            times=times,
            frequency_hz=f0,
            confidence=confidence,
            model=self.name,
            freq_range=rng,
        )
