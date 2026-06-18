from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .convert import convert_data
from .utils import FrequencyRange, coerce_frequency_range, normalize_audio_array, normalize_parameters
from .model_folders.run_basic_pitch import run_basic_pitch
from .model_folders.run_essentia_melodia import run_essentia_melodia
from .model_folders.run_fcpe import run_fcpe
from .model_folders.run_librosa_pyin import run_librosa_pyin
from .model_folders.run_librosa_yin import run_librosa_yin
from .model_folders.run_penn import run_penn
from .model_folders.run_pyworld_dio import run_pyworld_dio
from .model_folders.run_pyworld_harvest import run_pyworld_harvest
from .model_folders.run_swiftf0 import run_swiftf0
from .model_folders.run_torchcrepe import run_torchcrepe
from .model_folders.run_yaapt import run_yaapt
from .model_folders.run_yourmt3 import run_yourmt3


Runner = Callable[[Any, Mapping[str, Any] | None], Any]


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
        parameters=("sample_rate", "confidence_threshold", "fmin", "fmax"),
        description="SwiftF0 ONNX pitch tracker with a fixed backend frame rate.",
    ),
)


def _canonical(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def _registry() -> dict[str, ModelSpecification]:
    rows: dict[str, ModelSpecification] = {}
    for spec in MODEL_SPECS:
        rows[_canonical(spec.name)] = spec
        for alias in spec.aliases:
            rows[_canonical(alias)] = spec
    return rows


class PitchLab:
    def __init__(self, *, default_sample_rate: int = 44100) -> None:
        self.default_sample_rate = int(default_sample_rate)
        self.models = [spec.name for spec in MODEL_SPECS]
        self._models = _registry()

    def run_model(
        self,
        model_name: str,
        input_audio: Any,
        parameters: Mapping[str, Any] | None = None,
        desired_output: str | None = None,
    ) -> Any:
        spec = self._get_spec(model_name)
        params = normalize_parameters(parameters, default_sample_rate=self.default_sample_rate)
        audio = normalize_audio_array(input_audio)
        return convert_data(spec.runner(audio, params), desired_output, params)

    def run_all_models(
        self,
        input_audio: Any,
        parameters: Mapping[str, Any] | None = None,
        except_models: list[str] | tuple[str, ...] | set[str] | None = None,
        desired_output: str | None = None,
        *,
        raise_errors: bool = False,
    ) -> dict[str, Any]:
        params = normalize_parameters(parameters, default_sample_rate=self.default_sample_rate)
        audio = normalize_audio_array(input_audio)
        skipped = {_canonical(name) for name in (except_models or [])}
        results: dict[str, Any] = {}
        for spec in MODEL_SPECS:
            names = {_canonical(spec.name), *(_canonical(alias) for alias in spec.aliases)}
            if names & skipped:
                continue
            try:
                results[spec.name] = convert_data(spec.runner(audio, params), desired_output, params)
            except Exception as exc:
                if raise_errors:
                    raise
                results[spec.name] = {
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                }
        return results

    def print_all_model_specifications(self) -> None:
        for spec in MODEL_SPECS:
            self.print_specifications(spec.name)

    def print_specifications(self, model: str) -> None:
        spec = self._get_spec(model)
        aliases = ", ".join(spec.aliases) if spec.aliases else "none"
        parameters = ", ".join(spec.parameters) if spec.parameters else "none"
        print(f"{spec.name}")
        print(f"  aliases: {aliases}")
        print(f"  output_type: {spec.output_type}")
        print(f"  parameters: {parameters}")

    def print_description(self, model: str) -> str:
        spec = self._get_spec(model)
        return spec.description

    def _convert_data(self, data: Any, desired_form: str | None = None, parameters: Mapping[str, Any] | None = None) -> Any:
        return convert_data(data, desired_form, parameters)

    def _get_spec(self, model_name: str) -> ModelSpecification:
        key = _canonical(model_name)
        try:
            return self._models[key]
        except KeyError as exc:
            valid = ", ".join(available_models())
            raise KeyError(f"Unknown model {model_name!r}. Available models: {valid}") from exc


pitch_lab = PitchLab

_DEFAULT_LAB = PitchLab()


def run_model(
    model_name: str,
    input_audio: Any,
    parameters: Mapping[str, Any] | None = None,
    desired_output: str | None = None,
) -> Any:
    return _DEFAULT_LAB.run_model(model_name, input_audio, parameters, desired_output)


def run_all_models(
    input_audio: Any,
    parameters: Mapping[str, Any] | None = None,
    except_models: list[str] | tuple[str, ...] | set[str] | None = None,
    desired_output: str | None = None,
    *,
    raise_errors: bool = False,
) -> dict[str, Any]:
    return _DEFAULT_LAB.run_all_models(input_audio, parameters, except_models, desired_output, raise_errors=raise_errors)


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


def print_all_model_specifications() -> None:
    _DEFAULT_LAB.print_all_model_specifications()


def print_specifications(model: str) -> None:
    _DEFAULT_LAB.print_specifications(model)


def print_description(model: str) -> str:
    return _DEFAULT_LAB.print_description(model)


def _convert_data(data: Any, desired_form: str | None = None, parameters: Mapping[str, Any] | None = None) -> Any:
    return convert_data(data, desired_form, parameters)


coerce_range = coerce_frequency_range
