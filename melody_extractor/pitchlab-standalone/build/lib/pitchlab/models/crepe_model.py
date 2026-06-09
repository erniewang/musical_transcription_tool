from __future__ import annotations

import numpy as np

from .base import BasePitchModel, f0_dataframe, require_dependency


def crepe_bin_frequencies() -> np.ndarray:
    # CREPE's 360 pitch bins: 20-cent spacing, C1-B7 range.
    cents = 20.0 * np.arange(360) + 1997.3794084376191
    return 10.0 * np.power(2.0, cents / 1200.0)


def cents_to_hz(cents: np.ndarray) -> np.ndarray:
    return 10.0 * np.power(2.0, np.asarray(cents, dtype=float) / 1200.0)


class CrepeModel(BasePitchModel):
    name = "crepe"
    aliases = ("marl-crepe", "original-crepe")
    supports_native_range = False
    output_type = "f0"

    def predict(
        self,
        audio,
        sample_rate: int,
        *,
        hop_ms: float = 10.0,
        model_capacity: str = "full",
        model: str | None = None,
        viterbi: bool = False,
        center: bool = True,
        verbose: int = 0,
        min_confidence: float | None = 0.0,
        freq_range=None,
        fmin=None,
        fmax=None,
        **kwargs,
    ):
        crepe = require_dependency("crepe", "pip install crepe tensorflow")
        x = self.clean_audio(audio)
        rng = self.resolve_range(freq_range=freq_range, fmin=fmin, fmax=fmax, **kwargs)
        if model is not None:
            model_capacity = model

        try:
            from crepe.core import get_activation, to_local_average_cents, to_viterbi_cents

            activation = get_activation(
                x,
                sample_rate,
                model_capacity=model_capacity,
                center=center,
                step_size=hop_ms,
                verbose=verbose,
            )

            bin_freqs = crepe_bin_frequencies()
            allowed = (bin_freqs >= rng.low_hz) & (bin_freqs <= rng.high_hz)
            if not np.any(allowed):
                raise ValueError(f"No CREPE pitch bins are inside range {rng.as_tuple()} Hz")

            constrained = np.array(activation, copy=True)
            constrained[:, ~allowed] = 0.0
            confidence = constrained.max(axis=1)

            if viterbi:
                cents = to_viterbi_cents(constrained)
            else:
                cents = to_local_average_cents(constrained)
            frequency = cents_to_hz(cents)
            times = np.arange(len(frequency), dtype=float) * float(hop_ms) / 1000.0

        except Exception:
            # Fallback to public API, then hard-filter. This is less strong than
            # activation masking, but keeps the adapter usable across CREPE versions.
            time, frequency, confidence, _activation = crepe.predict(
                x,
                sample_rate,
                step_size=hop_ms,
                viterbi=viterbi,
                model_capacity=model_capacity,
                center=center,
                verbose=verbose,
            )
            times = np.asarray(time, dtype=float)

        frequency = np.asarray(frequency, dtype=float)
        confidence = np.asarray(confidence, dtype=float)
        if min_confidence is not None:
            frequency[confidence < float(min_confidence)] = 0.0

        return f0_dataframe(
            times=times,
            frequency_hz=frequency,
            confidence=confidence,
            model=self.name,
            freq_range=rng,
            extra={"backend_model": model_capacity},
        )
