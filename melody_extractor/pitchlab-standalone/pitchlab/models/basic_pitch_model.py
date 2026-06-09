from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from .base import BasePitchModel, note_events_dataframe, require_dependency, remove_range_kwargs
from ..audio_io import write_temp_wav


class BasicPitchModel(BasePitchModel):
    name = "basic-pitch"
    aliases = ("basic_pitch", "spotify-basic-pitch")
    supports_native_range = True
    output_type = "notes"

    def __init__(self) -> None:
        self.last_midi_data = None

    def predict(
        self,
        audio=None,
        sample_rate: int | None = None,
        *,
        audio_path: str | Path | None = None,
        save_midi_path: str | Path | None = None,
        freq_range=None,
        fmin=None,
        fmax=None,
        **kwargs: Any,
    ):
        inference = require_dependency("basic_pitch.inference", "pip install basic-pitch")
        predict = inference.predict
        rng = self.resolve_range(freq_range=freq_range, fmin=fmin, fmax=fmax, **kwargs)

        temp_path = None
        if audio_path is None:
            if audio is None or sample_rate is None:
                raise ValueError("Basic Pitch needs either audio_path or audio + sample_rate.")
            with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = Path(tmp.name)
            write_temp_wav(temp_path, np.asarray(audio, dtype=np.float32), int(sample_rate))
            audio_path = temp_path

        audio_path = Path(audio_path)

        predict_kwargs = remove_range_kwargs(kwargs)
        predict_kwargs.update(
            {
                "minimum_frequency": rng.low_hz,
                "maximum_frequency": rng.high_hz,
            }
        )

        try:
            model_output, midi_data, note_events = predict(str(audio_path), **predict_kwargs)
        except TypeError:
            # Older Basic Pitch builds may not accept min/max frequency args.
            clean = remove_range_kwargs(kwargs)
            model_output, midi_data, note_events = predict(str(audio_path), **clean)

        self.last_midi_data = midi_data
        df = note_events_dataframe(note_events, model=self.name, freq_range=rng)

        if save_midi_path is not None:
            self.write_midi(save_midi_path)

        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

        return df

    def write_midi(self, path: str | Path) -> None:
        if self.last_midi_data is None:
            raise RuntimeError("No Basic Pitch MIDI data available. Run predict() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.last_midi_data.write(str(path))
