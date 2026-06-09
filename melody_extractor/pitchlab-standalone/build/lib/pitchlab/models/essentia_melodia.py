from __future__ import annotations

import numpy as np

from .base import BasePitchModel, f0_dataframe, hop_length_from_ms, require_dependency, remove_range_kwargs


class EssentiaMelodiaModel(BasePitchModel):
    name = "essentia-melodia"
    aliases = ("melodia", "essentia", "predominant-pitch-melodia", "essentia_melodia")
    supports_native_range = True
    output_type = "f0"

    def predict(
        self,
        audio,
        sample_rate: int,
        *,
        hop_ms: float = 10.0,
        frame_size: int = 2048,
        guess_unvoiced: bool = False,
        min_confidence: float | None = None,
        freq_range=None,
        fmin=None,
        fmax=None,
        **kwargs,
    ):
        es = require_dependency("essentia.standard", "pip install essentia")
        x = self.clean_audio(audio).astype(np.float32)
        rng = self.resolve_range(freq_range=freq_range, fmin=fmin, fmax=fmax, **kwargs)
        hop_size = hop_length_from_ms(sample_rate, hop_ms)

        alg_kwargs = {
            "sampleRate": int(sample_rate),
            "frameSize": int(frame_size),
            "hopSize": int(hop_size),
            "minFrequency": float(rng.low_hz),
            "maxFrequency": float(rng.high_hz),
            "guessUnvoiced": bool(guess_unvoiced),
        }
        alg_kwargs.update(remove_range_kwargs(kwargs))

        pitch, confidence = es.PredominantPitchMelodia(**alg_kwargs)(x)
        pitch = np.asarray(pitch, dtype=float)
        confidence = np.asarray(confidence, dtype=float)
        if min_confidence is not None:
            pitch[confidence < float(min_confidence)] = 0.0
        times = np.arange(len(pitch), dtype=float) * hop_size / float(sample_rate)

        return f0_dataframe(
            times=times,
            frequency_hz=pitch,
            confidence=confidence,
            model=self.name,
            freq_range=rng,
        )
