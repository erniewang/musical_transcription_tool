from __future__ import annotations

import numpy as np

from .base import BasePitchModel, f0_dataframe, require_dependency, remove_range_kwargs


class YaaptModel(BasePitchModel):
    name = "yaapt"
    aliases = ("amfm-yaapt", "amfm_decompy_yaapt")
    supports_native_range = True
    output_type = "f0"

    def predict(
        self,
        audio,
        sample_rate: int,
        *,
        hop_ms: float = 10.0,
        frame_length_ms: float = 35.0,
        freq_range=None,
        fmin=None,
        fmax=None,
        **kwargs,
    ):
        basic_tools = require_dependency("amfm_decompy.basic_tools", "pip install amfm-decompy")
        pyaapt = require_dependency("amfm_decompy.pYAAPT", "pip install amfm-decompy")
        x = self.clean_audio(audio).astype(np.float64)
        rng = self.resolve_range(freq_range=freq_range, fmin=fmin, fmax=fmax, **kwargs)

        signal = basic_tools.SignalObj(x, int(sample_rate))
        params = remove_range_kwargs(kwargs)
        params.update(
            {
                "frame_length": float(frame_length_ms),
                "frame_space": float(hop_ms),
                "f0_min": float(rng.low_hz),
                "f0_max": float(rng.high_hz),
            }
        )
        try:
            pitch = pyaapt.yaapt(signal, **params)
        except TypeError:
            params.pop("f0_min", None)
            params.pop("f0_max", None)
            pitch = pyaapt.yaapt(signal, **params)

        f0 = np.asarray(getattr(pitch, "samp_values", getattr(pitch, "values", [])), dtype=float)
        if f0.size == 0 and hasattr(pitch, "frames"):
            f0 = np.asarray(pitch.frames, dtype=float)
        times = np.arange(len(f0), dtype=float) * float(hop_ms) / 1000.0
        confidence = np.where(f0 > 0.0, 1.0, 0.0)
        return f0_dataframe(
            times=times,
            frequency_hz=f0,
            confidence=confidence,
            model=self.name,
            freq_range=rng,
        )
