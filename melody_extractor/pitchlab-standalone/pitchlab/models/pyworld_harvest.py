from __future__ import annotations

import numpy as np

from .base import BasePitchModel, f0_dataframe, require_dependency


class PyWorldHarvestModel(BasePitchModel):
    name = "pyworld-harvest"
    aliases = ("harvest", "pyworld_harvest", "world-harvest")
    supports_native_range = True
    output_type = "f0"

    def predict(
        self,
        audio,
        sample_rate: int,
        *,
        hop_ms: float = 10.0,
        refine: bool = True,
        freq_range=None,
        fmin=None,
        fmax=None,
        **kwargs,
    ):
        pw = require_dependency("pyworld", "pip install pyworld")
        x = self.clean_audio(audio).astype(np.float64)
        rng = self.resolve_range(freq_range=freq_range, fmin=fmin, fmax=fmax, **kwargs)

        f0, time_axis = pw.harvest(
            x,
            int(sample_rate),
            f0_floor=float(rng.low_hz),
            f0_ceil=float(rng.high_hz),
            frame_period=float(hop_ms),
        )
        if refine:
            f0 = pw.stonemask(x, f0, time_axis, int(sample_rate))
        confidence = np.where(np.asarray(f0) > 0.0, 1.0, 0.0)
        return f0_dataframe(
            times=time_axis,
            frequency_hz=f0,
            confidence=confidence,
            model=self.name,
            freq_range=rng,
        )
