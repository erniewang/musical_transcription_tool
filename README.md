# Musical Transcription Tool

Self-contained audio-to-notation pipeline, organized by pipeline stage.

## Layout

```
main.py                   # thin entry: pitch by default, or `python main.py rhythm`
pitch_main.py             # f0 pipeline throughput
rhythm_main.py            # rhythm pipeline throughput (stub until models exist)
processing.py             # settings loader + name → function ops maps + load/preprocess audio
helpers.py                # colored logging + PROJECT_ROOT
transcribe_settings.json  # pitch (f0) run
rhythm_settings.json      # rhythm run (stub until models exist)
input/                    # sample audio / test inputs
output/                   # per-run results: output/<audio-stem>/<model>/
preprocessing/            # audio prep before models: load, filters, harmonic split
pitchlab/                 # model wrappers only (f0 / note extractors) + registry + runner
rhythm/                   # rhythmic branch (placeholder, not implemented yet)
postprocessing/           # after models: confidence, refine, quantize, notation (PDF), sonify (MP3)
```

## Usage

Edit a settings file, then run:

```bash
conda activate pitchlab
python pitch_main.py                      # pitch
python rhythm_main.py                     # rhythm (stub)
python main.py                            # same as pitch_main.py
python main.py rhythm                     # same as rhythm_main.py
python main.py path/to/custom_settings.json
```

There are no other command-line flags. Each settings file is one complete set of
instructions for one piece of music:

- `pitch_main.py` / `transcribe_settings.json` - f0 estimation → pitch postprocessing → CSV / sonify / PDF
- `rhythm_main.py` / `rhythm_settings.json` - same shell; extractors and post ops still to be filled in

Grouped by pipeline stage:

- `input` - which audio file, what slice of it, at what sample rate
- `preprocessing` - harmonic extraction, high-pass filter, low-pass filter
- `extraction` - explicit `models` list plus shared hop / search-range fields
- `postprocessing` - task-specific steps (`enabled` + `order` + params)
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
works no matter your current working directory. Duplicate `order` values in the same
stage are rejected.


Per model, results land in `output/<audio-stem>/<model>/`:

- `<model>.pdf` (and the intermediate `.musicxml`) - transcribed notation
- `<model>_sonified.mp3` - sine-wave rendering of the predicted pitch track
- `<model>_refined.csv` - the refined pitch run

## Rules to keep this organized

- `pitchlab/` = how to call a model. No refinement or scoring logic here.
- `preprocessing/` / `postprocessing/` = singular ops on audio / pitch tracks.
- `processing.py` = settings + ops registry (name → function) + `load_input_audio` / `preprocess_audio`.
- `main.py` = entry only; `pitch_main.py` / `rhythm_main.py` = per-task throughput.
