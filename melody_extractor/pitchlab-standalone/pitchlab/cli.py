from __future__ import annotations

import argparse
from pathlib import Path

from .audio_io import load_audio, save_dataframe
from .plotting import plot_f0_csv
from .registry import available_models, get_model, model_info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pitchlab standalone F0 / note extraction CLI")

    parser.add_argument("--input", "-i", help="Input audio path")
    parser.add_argument("--output", "-o", help="Output CSV path")
    parser.add_argument("--model", "--engine", default="torchcrepe", help="Model/engine name")
    parser.add_argument("--list-models", action="store_true", help="List registered models and exit")

    parser.add_argument("--start", type=float, default=0.0, help="Section start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="Section end time in seconds")
    parser.add_argument("--sample-rate", type=int, default=None, help="Optional resampling rate")
    parser.add_argument("--hop-ms", type=float, default=10.0, help="Analysis hop in milliseconds")
    parser.add_argument(
        "--range",
        dest="freq_range",
        nargs=2,
        type=float,
        metavar=("LOW_HZ", "HIGH_HZ"),
        default=None,
        help="F0 range in Hz, e.g. --range 100 500",
    )

    # Common engine-specific flags. Only relevant flags are passed to each backend.
    parser.add_argument("--model-size", default=None, help="Model size/capacity, e.g. tiny/full")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Confidence/periodicity threshold")
    parser.add_argument("--viterbi", action="store_true", help="Use Viterbi decoding when backend supports it")
    parser.add_argument("--frame-length", type=int, default=None)
    parser.add_argument("--no-refine", action="store_true", help="Disable pyworld StoneMask refinement")

    parser.add_argument("--midi-output", default=None, help="MIDI output path for Basic Pitch")
    parser.add_argument("--plot", action="store_true", help="Also write a PNG plot for F0 outputs")
    parser.add_argument("--plot-output", default=None, help="Explicit plot PNG output path")

    return parser


def _default_output_path(input_path: str | Path, model_name: str) -> Path:
    p = Path(input_path)
    safe_model = model_name.replace("/", "-").replace("_", "-")
    return Path("output") / f"{p.stem}_{safe_model}.csv"


def _build_model_kwargs(args: argparse.Namespace) -> dict:
    model_name = args.model.strip().lower().replace("_", "-")
    kwargs = {
        "hop_ms": args.hop_ms,
        "freq_range": tuple(args.freq_range) if args.freq_range else None,
    }

    if model_name in {"torchcrepe", "torch-crepe"}:
        if args.model_size:
            kwargs["model_size"] = args.model_size
        if args.batch_size:
            kwargs["batch_size"] = args.batch_size
        if args.device:
            kwargs["device"] = args.device
        if args.threshold is not None:
            kwargs["min_periodicity"] = args.threshold

    elif model_name in {"crepe", "marl-crepe", "original-crepe"}:
        if args.model_size:
            kwargs["model_capacity"] = args.model_size
        if args.viterbi:
            kwargs["viterbi"] = True
        if args.threshold is not None:
            kwargs["min_confidence"] = args.threshold

    elif model_name in {"librosa-pyin", "pyin", "librosa-yin", "yin"}:
        if args.frame_length:
            kwargs["frame_length"] = args.frame_length

    elif model_name in {"essentia-melodia", "melodia", "essentia", "predominant-pitch-melodia"}:
        if args.frame_length:
            kwargs["frame_size"] = args.frame_length
        if args.threshold is not None:
            kwargs["min_confidence"] = args.threshold

    elif model_name in {"pyworld-dio", "dio", "pyworld-harvest", "harvest"}:
        kwargs["refine"] = not args.no_refine

    elif model_name in {"basic-pitch", "basic_pitch", "spotify-basic-pitch"}:
        if args.midi_output:
            kwargs["save_midi_path"] = args.midi_output

    else:
        # Best-effort external adapters.
        if args.model_size:
            kwargs["model_size"] = args.model_size
        if args.batch_size:
            kwargs["batch_size"] = args.batch_size
        if args.device:
            kwargs["device"] = args.device
        if args.threshold is not None:
            kwargs["threshold"] = args.threshold
        if args.viterbi:
            kwargs["viterbi"] = True

    return {k: v for k, v in kwargs.items() if v is not None}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        for row in model_info():
            aliases = ", ".join(row["aliases"])
            native = "native-range" if row["supports_native_range"] else "filtered-range"
            print(f"{row['name']:<20} {row['output_type']:<6} {native:<15} aliases: {aliases}")
        return 0

    if not args.input:
        parser.error("--input is required unless --list-models is used")

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else _default_output_path(input_path, args.model)

    model = get_model(args.model)
    kwargs = _build_model_kwargs(args)

    # Basic Pitch can use the source path directly only when no section slicing or resampling is requested.
    direct_path_ok = (
        model.name == "basic-pitch"
        and float(args.start) == 0.0
        and args.end is None
        and args.sample_rate is None
    )

    if direct_path_ok:
        df = model.predict(audio_path=input_path, **kwargs)
    else:
        audio, sr = load_audio(input_path, start=args.start, end=args.end, sample_rate=args.sample_rate)
        df = model.predict(audio, sr, **kwargs)
        if args.start:
            # Keep section-local extraction but expose song-global time for integration.
            if "time" in df.columns:
                df["time"] = df["time"] + float(args.start)
            if "start_time" in df.columns:
                df["start_time"] = df["start_time"] + float(args.start)
            if "end_time" in df.columns:
                df["end_time"] = df["end_time"] + float(args.start)

    save_dataframe(df, output_path)
    print(f"Saved CSV: {output_path}")

    if args.plot:
        plot_output = Path(args.plot_output) if args.plot_output else output_path.with_suffix(".png")
        try:
            plot_f0_csv(output_path, plot_output)
            print(f"Saved plot: {plot_output}")
        except Exception as exc:
            print(f"Could not create plot: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
