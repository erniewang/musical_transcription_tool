from __future__ import annotations

from .base import BasePitchModel, remove_range_kwargs
from .external_common import generic_pitch_call


class SwiftF0Model(BasePitchModel):
    name = "swiftf0"
    aliases = ("swift-f0", "swift_f0")
    supports_native_range = True
    output_type = "f0"

    def predict(
        self,
        audio,
        sample_rate: int,
        *,
        hop_ms: float = 10.0,
        freq_range=None,
        fmin=None,
        fmax=None,
        **kwargs,
    ):
        rng = self.resolve_range(freq_range=freq_range, fmin=fmin, fmax=fmax, **kwargs)
        return generic_pitch_call(
            module_names=("swiftf0", "swift_f0"),
            callable_names=("predict", "infer", "extract", "estimate", "estimate_f0", "get_f0"),
            audio=self.clean_audio(audio),
            sample_rate=sample_rate,
            model_name=self.name,
            freq_range=rng,
            hop_ms=hop_ms,
            kwargs=remove_range_kwargs(kwargs),
        )
