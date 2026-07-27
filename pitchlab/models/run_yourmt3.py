from __future__ import annotations

import copy
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..audio_io import write_wav
from ..utils import clean_model_parameters, coerce_frequency_range, normalize_audio_array


MODEL_NAME = "yourmt3"
ACCEPTED_PARAMETERS = {
    "checkpoint_path",
    "model_variant",
    "hf_repo_id",
    "hf_filename",
    "batch_size",
    "resample_to",
    "device",
    "midi_path",
    "keep_intermediate",
}

DEFAULT_HF_REPO_ID = "shethjenil/Audio2Midi_Models"
DEFAULT_MODEL_VARIANT = "YMT3+"
DEFAULT_HF_FILENAME = "YMT3+.pt"


def run_yourmt3(input_audio, parameters=None):
    params = clean_model_parameters(MODEL_NAME, parameters, ACCEPTED_PARAMETERS)

    try:
        import pretty_midi
    except ImportError as exc:
        raise ImportError("YourMT3 conversion requires pretty_midi. Install with: pip install pretty_midi") from exc
    try:
        from yourmt3 import YMT3
    except ImportError as exc:
        raise ImportError("Could not import yourmt3. Install with: pip install yourmt3") from exc

    sample_rate = int(params["sample_rate"])
    rng = coerce_frequency_range(params, default=None)
    min_pitch, max_pitch = _range_to_midi_bounds(rng)
    batch_size = int(params.get("batch_size", 8))
    model_variant = str(params.get("model_variant", DEFAULT_MODEL_VARIANT))
    checkpoint = _resolve_checkpoint(params)
    model = YMT3(str(checkpoint), model_variant)
    device = params.get("device")
    if device:
        for candidate in (model, getattr(model, "model", None), getattr(model, "net", None)):
            if hasattr(candidate, "to"):
                try:
                    candidate.to(device)
                except Exception:
                    pass

    temp_paths: list[Path] = []
    input_audio_path = _write_temp_input(input_audio, sample_rate, params, temp_paths)
    raw_result = _call_predict(model, input_audio_path, batch_size)
    raw_midi_path = _extract_midi_path(raw_result)
    if raw_midi_path is None or not raw_midi_path.exists():
        raise RuntimeError(f"YourMT3 did not return a readable MIDI path. Raw result: {raw_result!r}")

    pm = pretty_midi.PrettyMIDI(str(raw_midi_path))
    if min_pitch is not None or max_pitch is not None:
        pm = _filter_pretty_midi_by_pitch(pm, min_pitch=min_pitch, max_pitch=max_pitch)

    midi_path = params.get("midi_path")
    if midi_path:
        midi_path = Path(midi_path)
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        pm.write(str(midi_path))

    df = _pretty_midi_to_notes_dataframe(pm, pretty_midi)
    df["model"] = MODEL_NAME
    if midi_path:
        df.attrs["midi_path"] = str(midi_path)

    if not bool(params.get("keep_intermediate", False)):
        for path in temp_paths:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
    return df


def _resolve_checkpoint(params: dict[str, Any]) -> Path:
    checkpoint_path = params.get("checkpoint_path")
    if checkpoint_path:
        checkpoint = Path(checkpoint_path).expanduser()
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("Downloading YourMT3 checkpoints requires huggingface_hub.") from exc
        checkpoint = Path(
            hf_hub_download(
                str(params.get("hf_repo_id", DEFAULT_HF_REPO_ID)),
                str(params.get("hf_filename", DEFAULT_HF_FILENAME)),
            )
        )
    if not checkpoint.exists():
        raise FileNotFoundError(f"YourMT3 checkpoint not found: {checkpoint}")
    return checkpoint


def _write_temp_input(input_audio, sample_rate: int, params: dict[str, Any], temp_paths: list[Path]) -> Path:
    if isinstance(input_audio, (str, os.PathLike)):
        path = Path(input_audio).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    audio = normalize_audio_array(input_audio)
    target_sample_rate = params.get("resample_to", 16000)
    sr = sample_rate
    if target_sample_rate is not None and int(target_sample_rate) != sr:
        try:
            import librosa
        except ImportError as exc:
            raise ImportError("YourMT3 resampling requires librosa. Install with: pip install librosa") from exc
        audio = librosa.resample(audio, orig_sr=sr, target_sr=int(target_sample_rate))
        sr = int(target_sample_rate)

    fd, name = tempfile.mkstemp(prefix="pitchlab_yourmt3_", suffix=".wav")
    os.close(fd)
    path = Path(name)
    write_wav(path, audio, sr)
    temp_paths.append(path)
    return path


def _call_predict(model: Any, audio_path: Path, batch_size: int) -> Any:
    path = str(audio_path)
    attempts = (
        lambda: model.predict(path, batch_size, lambda _i, _total: None),
        lambda: model.predict(path, batch_size=batch_size),
        lambda: model.predict(path),
    )
    last_exc = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_exc = exc
    raise last_exc


def _extract_midi_path(raw_result: Any) -> Path | None:
    if raw_result is None:
        return None
    if isinstance(raw_result, (str, os.PathLike)):
        path = Path(raw_result).expanduser()
        return path if path.suffix.lower() in {".mid", ".midi"} else None
    if isinstance(raw_result, dict):
        for key in ("midi", "midi_path", "path", "file", "output", "result"):
            if key in raw_result:
                path = _extract_midi_path(raw_result[key])
                if path is not None:
                    return path
    if isinstance(raw_result, (list, tuple)):
        for item in raw_result:
            path = _extract_midi_path(item)
            if path is not None:
                return path
    return None


def _range_to_midi_bounds(rng) -> tuple[int | None, int | None]:
    if rng is None:
        return None, None
    min_pitch = int(max(0, min(127, math.ceil(69.0 + 12.0 * math.log2(rng.low_hz / 440.0)))))
    max_pitch = int(max(0, min(127, math.floor(69.0 + 12.0 * math.log2(rng.high_hz / 440.0)))))
    return min_pitch, max_pitch


def _filter_pretty_midi_by_pitch(pm, *, min_pitch: int | None, max_pitch: int | None):
    filtered = copy.deepcopy(pm)
    for inst in filtered.instruments:
        inst.notes = [
            note
            for note in inst.notes
            if (min_pitch is None or note.pitch >= min_pitch) and (max_pitch is None or note.pitch <= max_pitch)
        ]
    return filtered


def _pretty_midi_to_notes_dataframe(pm, pretty_midi) -> pd.DataFrame:
    rows = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            rows.append(
                {
                    "start_time": float(note.start),
                    "end_time": float(note.end),
                    "duration": float(note.end - note.start),
                    "pitch_midi": int(note.pitch),
                    "frequency_hz": float(440.0 * (2.0 ** ((int(note.pitch) - 69) / 12.0))),
                    "velocity": int(note.velocity),
                    "program": int(instrument.program),
                    "instrument_name": pretty_midi.program_to_instrument_name(instrument.program)
                    if not instrument.is_drum
                    else "Drums",
                    "is_drum": bool(instrument.is_drum),
                    "time": float(note.start),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["start_time", "program", "pitch_midi", "end_time"], kind="mergesort").reset_index(drop=True)
    return df
