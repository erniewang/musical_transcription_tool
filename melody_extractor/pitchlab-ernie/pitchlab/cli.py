from __future__ import annotations

import argparse
from pathlib import Path

from .audio_io import load_audio, save_dataframe
from .main import PitchLab, model_specifications


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pitchlab raw-audio model runner")
    parser.add_argument("--input", "-i", help="Input audio path")
    parser.add_argument("--output", "-o", help="Output CSV path")
    parser.add_argument("--model", "-m", default="librosa-pyin", help="Model name")
    parser.add_argument("--list-models", action="store_true", help="List model specifications and exit")
    parser.add_argument("--sample-rate", type=int, default=None, help="Optional resampling rate for file input")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--range", dest="freq_range", nargs=2, type=float, metavar=("FMIN", "FMAX"))
    parser.add_argument("--desired-output", default="pandas_df", help="pandas_df, csv, midi, sonified_audio, numpy")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    lab = PitchLab()
    if args.list_models:
        for spec in model_specifications():
            aliases = ", ".join(spec["aliases"]) if spec["aliases"] else "none"
            params = ", ".join(spec["parameters"]) if spec["parameters"] else "none"
            print(f"{spec['name']} ({spec['output_type']})")
            print(f"  aliases: {aliases}")
            print(f"  parameters: {params}")
        return 0

    if not args.input:
        parser.error("--input is required unless --list-models is used")

    audio, sr = load_audio(args.input, start=args.start, end=args.end, sample_rate=args.sample_rate)
    parameters = {
        "sample_rate": sr,
        "hop_ms": args.hop_ms,
    }
    if args.freq_range:
        parameters["freq_range"] = tuple(args.freq_range)

    result = lab.run_model(args.model, audio, parameters, desired_output="pandas_df")
    if args.start:
        if "time" in result:
            result["time"] = result["time"] + float(args.start)
        if "start_time" in result:
            result["start_time"] = result["start_time"] + float(args.start)
        if "end_time" in result:
            result["end_time"] = result["end_time"] + float(args.start)

    output_path = Path(args.output) if args.output else _default_output_path(args.input, args.model)
    save_dataframe(result, output_path)
    print(f"Saved CSV: {output_path}")
    return 0


def _default_output_path(input_path: str, model_name: str) -> Path:
    path = Path(input_path)
    safe_model = model_name.replace("/", "-").replace("_", "-")
    return Path("output") / f"{path.stem}_{safe_model}.csv"


if __name__ == "__main__":
    raise SystemExit(main())
