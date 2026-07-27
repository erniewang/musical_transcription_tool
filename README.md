# Musical Transcription Tool

Self-contained audio-to-notation pipeline, organized by pipeline stage.

## Layout

```
main.py                   # transcribe() orchestrator (run this) - wiring only
transcribe_settings.json  # every tunable value for one piece of music
settings.py               # settings schema, validation, and loader
input/                    # sample audio / test inputs
output/                   # per-run results: output/<audio-stem>/<model>/
preprocessing/            # audio prep before models: load, filters, harmonic split
pitchlab/                 # model wrappers only (f0 / note extractors) + registry + runner
rhythm/                   # rhythmic branch (placeholder, not implemented yet)
postprocessing/           # after models: confidence, refine, quantize, notation (PDF), sonify (MP3)
```

## Usage

Edit `transcribe_settings.json`, then run:

```bash
conda activate pitchlab
python main.py
```

There are no command-line flags. `transcribe_settings.json` is one complete set of
instructions for one piece of music, grouped by pipeline stage:

- `input` - which audio file, what slice of it, at what sample rate
- `preprocessing` - harmonic extraction, high-pass filter, low-pass filter
- `extraction` - explicit `models` list, hop size, and the pitch search range
- `postprocessing` - confidence filtering, pitch-range filtering, interpolation, quantization
- `output` - where results go and the sonified audio format (`mp3` or `wav`)

Every step in `preprocessing` and `postprocessing` is its own object with
`enabled`, `order`, and its own parameters. `order` controls execution sequence
within that stage; duplicate orders in the same stage are rejected. Example:

```json
"high_pass_filter": { "enabled": true, "cutoff_hz": 131.0, "order": 2 }
```

`extraction.models` is a required explicit list of model names to run. A `null`
value means "use the default" elsewhere: chromatic for
`postprocessing.quantize.pitch_set`, `output/<audio-stem>` for
`output.directory`, and end-of-file for `input.end_seconds`.

Relative paths in the settings file resolve against the project root, so the program
works no matter your current working directory. Unknown keys are rejected with an error
naming the offending key rather than being silently ignored.


Per model, results land in `output/<audio-stem>/<model>/`:

- `<model>.pdf` (and the intermediate `.musicxml`) - transcribed notation
- `<model>_sonified.mp3` - sine-wave rendering of the predicted pitch track
- `<model>_refined.csv` - the refined pitch run

## Rules to keep this organized

- `pitchlab/` = how to call a model. No refinement or scoring logic here.
- `preprocessing/` / `postprocessing/` = singular ops on audio / pitch tracks.
- `main.py` = wiring only: load/preprocess once, then for each model run
  extract -> postprocess -> write start to finish.
