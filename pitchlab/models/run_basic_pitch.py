from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from ..audio_io import write_wav
from ..utils import clean_model_parameters, coerce_frequency_range, note_events_dataframe, normalize_audio_array, remove_range_kwargs, require_dependency


MODEL_NAME = "basic-pitch"
ACCEPTED_PARAMETERS = {"audio_path", "save_midi_path"}


def run_basic_pitch(input_audio, parameters=None):
    params = clean_model_parameters(MODEL_NAME, parameters, ACCEPTED_PARAMETERS)
    inference = require_dependency("basic_pitch.inference", "pip install basic-pitch")
    predict = inference.predict

    sample_rate = int(params["sample_rate"])
    rng = coerce_frequency_range(params)
    audio_path = params.get("audio_path")
    temp_path = None

    try:
        if audio_path is None:
            with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = Path(tmp.name)
            write_wav(temp_path, normalize_audio_array(input_audio).astype(np.float32), sample_rate)
            audio_path = temp_path
        audio_path = Path(audio_path)

        predict_kwargs = remove_range_kwargs(params)
        for key in ("sample_rate", "audio_path", "save_midi_path"):
            predict_kwargs.pop(key, None)
        predict_kwargs.update(
            {
                "minimum_frequency": rng.low_hz,
                "maximum_frequency": rng.high_hz,
            }
        )
        try:
            _model_output, midi_data, note_events = predict(str(audio_path), **predict_kwargs)
        except TypeError:
            predict_kwargs.pop("minimum_frequency", None)
            predict_kwargs.pop("maximum_frequency", None)
            _model_output, midi_data, note_events = predict(str(audio_path), **predict_kwargs)

        save_midi_path = params.get("save_midi_path")
        if save_midi_path is not None:
            midi_path = Path(save_midi_path)
            midi_path.parent.mkdir(parents=True, exist_ok=True)
            midi_data.write(str(midi_path))

        return note_events_dataframe(note_events, model=MODEL_NAME, freq_range=rng)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
