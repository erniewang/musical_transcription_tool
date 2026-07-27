"""Canonical registry of every pitch model wrapper.

Trimmed from the original pitchlab: ``run_model`` returns the wrapper's raw
DataFrame; output conversion is the postprocessing package's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .utils import normalize_audio_array, normalize_parameters
from .models.run_basic_pitch import run_basic_pitch
from .models.run_crepe import run_crepe
from .models.run_essentia_melodia import run_essentia_melodia
from .models.run_fcpe import run_fcpe
from .models.run_librosa_pyin import run_librosa_pyin
from .models.run_librosa_yin import run_librosa_yin
from .models.run_penn import run_penn
from .models.run_pyworld_dio import run_pyworld_dio
from .models.run_pyworld_harvest import run_pyworld_harvest
from .models.run_swiftf0 import run_swiftf0
from .models.run_torchcrepe import run_torchcrepe
from .models.run_yaapt import run_yaapt
from .models.run_yourmt3 import run_yourmt3


Runner = Callable[[Any, Mapping[str, Any] | None], Any]

DEFAULT_SAMPLE_RATE = 44_100


@dataclass(frozen=True)
class ModelSpecification:
    name: str
    runner: Runner
    aliases: tuple[str, ...] = ()
    output_type: str = "f0"
    parameters: tuple[str, ...] = ()
    description: str = ""


MODEL_SPECS: tuple[ModelSpecification, ...] = (
    ModelSpecification(
        "librosa-pyin",
        run_librosa_pyin,
        aliases=("pyin", "librosa_pyin"),
        parameters=("sample_rate", "hop_ms", "frame_length", "resolution", "fmin", "fmax", "freq_range"),
        description="librosa probabilistic YIN F0 tracker.",
    ),
    ModelSpecification(
        "librosa-yin",
        run_librosa_yin,
        aliases=("yin", "librosa_yin"),
        parameters=("sample_rate", "hop_ms", "frame_length", "trough_threshold", "fmin", "fmax", "freq_range"),
        description="librosa YIN F0 tracker.",
    ),
    ModelSpecification(
        "torchcrepe",
        run_torchcrepe,
        aliases=("torch-crepe", "torch_crepe"),
        parameters=("sample_rate", "hop_ms", "model_size", "batch_size", "device", "min_periodicity", "fmin", "fmax"),
        description="PyTorch CREPE F0 tracker.",
    ),
    ModelSpecification(
        "crepe",
        run_crepe,
        aliases=("marl-crepe", "original-crepe"),
        parameters=("sample_rate", "hop_ms", "model_capacity", "viterbi", "center", "min_confidence", "fmin", "fmax"),
        description="Original CREPE backend with optional activation masking.",
    ),
    ModelSpecification(
        "essentia-melodia",
        run_essentia_melodia,
        aliases=("melodia", "essentia", "essential-melodia", "essentia_melodia"),
        parameters=("sample_rate", "hop_ms", "frame_size", "guess_unvoiced", "min_confidence", "fmin", "fmax"),
        description="Essentia PredominantPitchMelodia.",
    ),
    ModelSpecification(
        "pyworld-dio",
        run_pyworld_dio,
        aliases=("dio", "pyworld_dio", "world-dio"),
        parameters=("sample_rate", "hop_ms", "channels_in_octave", "speed", "refine", "fmin", "fmax"),
        description="WORLD DIO F0 estimator with optional StoneMask refinement.",
    ),
    ModelSpecification(
        "pyworld-harvest",
        run_pyworld_harvest,
        aliases=("harvest", "pyworld_harvest", "world-harvest"),
        parameters=("sample_rate", "hop_ms", "refine", "fmin", "fmax"),
        description="WORLD Harvest F0 estimator with optional StoneMask refinement.",
    ),
    ModelSpecification(
        "yaapt",
        run_yaapt,
        aliases=("amfm-yaapt", "amfm_decompy_yaapt"),
        parameters=("sample_rate", "hop_ms", "frame_length_ms", "fmin", "fmax"),
        description="AMFM decompy YAAPT F0 estimator.",
    ),
    ModelSpecification(
        "basic-pitch",
        run_basic_pitch,
        aliases=("basic_pitch", "spotify-basic-pitch"),
        output_type="notes",
        parameters=("sample_rate", "audio_path", "save_midi_path", "fmin", "fmax"),
        description="Spotify Basic Pitch note transcription.",
    ),
    ModelSpecification(
        "penn",
        run_penn,
        aliases=("penn-model", "penn_pitch"),
        parameters=("sample_rate", "hop_ms", "checkpoint", "batch_size", "center", "decoder", "interp_unvoiced_at", "gpu", "fmin", "fmax"),
        description="PENN pitch tracker. Backend import can fail if torbi does not match torch/CUDA.",
    ),
    ModelSpecification(
        "fcpe",
        run_fcpe,
        aliases=("torchfcpe", "fcpe-model"),
        parameters=("sample_rate", "hop_ms", "device", "decoder_mode", "threshold", "fmin", "fmax"),
        description="torchfcpe bundled inference model. Ranges below backend support return unvoiced output instead of crashing.",
    ),
    ModelSpecification(
        "yourmt3",
        run_yourmt3,
        aliases=("ymt3", "yourmt3-plus", "ymt3-plus", "yourmpt"),
        output_type="notes",
        parameters=("sample_rate", "checkpoint_path", "model_variant", "hf_repo_id", "hf_filename", "batch_size", "midi_path", "fmin", "fmax"),
        description="YourMT3 audio-to-MIDI transcription with post transcription range filtering.",
    ),
    ModelSpecification(
        "swiftf0",
        run_swiftf0,
        aliases=("swift-f0", "swift_f0"),
        parameters=("sample_rate", "hop_ms", "fmin", "fmax"),
        description="Best-effort SwiftF0 adapter.",
    ),
)


def _canonical(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


_REGISTRY: dict[str, ModelSpecification] = {}
for _spec in MODEL_SPECS:
    _REGISTRY[_canonical(_spec.name)] = _spec
    for _alias in _spec.aliases:
        _REGISTRY[_canonical(_alias)] = _spec


def get_specification(model_name: str) -> ModelSpecification:
    key = _canonical(model_name)
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        valid = ", ".join(available_models())
        raise KeyError(f"Unknown model {model_name!r}. Available models: {valid}") from exc


def run_model(model_name: str, input_audio: Any, parameters: Mapping[str, Any] | None = None) -> Any:
    """Run one model and return its raw DataFrame output."""
    spec = get_specification(model_name)
    params = normalize_parameters(parameters, default_sample_rate=DEFAULT_SAMPLE_RATE)
    audio = normalize_audio_array(input_audio)
    return spec.runner(audio, params)


def available_models() -> list[str]:
    return [spec.name for spec in MODEL_SPECS]


def model_specifications() -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "aliases": spec.aliases,
            "output_type": spec.output_type,
            "parameters": spec.parameters,
            "description": spec.description,
        }
        for spec in MODEL_SPECS
    ]
