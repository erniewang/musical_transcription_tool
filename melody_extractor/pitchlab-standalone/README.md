# pitchlab-standalone

Standalone `pitchlab` package for running pitch / F0 / note extraction models with a shared Hz range argument.

The main feature is the same everywhere:

```bash
pitchlab --input input.wav --model torchcrepe --range 100 500 --output out.csv
```

`--range 100 500` means:

```text
lower F0 bound = 100 Hz
upper F0 bound = 500 Hz
```

This is designed for timestamped transcription sections where the user can constrain the expected pitch range before inference.

## Install

```bash
unzip pitchlab-standalone.zip
cd pitchlab-standalone
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Core install gives you `librosa-yin` and `librosa-pyin`:

```bash
pip install -r requirements-core.txt
```

Install only the engines you are testing:

```bash
pip install -r requirements-torchcrepe.txt
pip install -r requirements-crepe.txt
pip install -r requirements-basic-pitch.txt
pip install -r requirements-essentia.txt
pip install -r requirements-pyworld.txt
pip install -r requirements-yaapt.txt
```

## CLI examples

List registered models:

```bash
pitchlab --list-models
```

Run one section:

```bash
pitchlab \
  --input input/test_audio.wav \
  --start 4.2 \
  --end 12.8 \
  --model torchcrepe \
  --range 100 500 \
  --output output/section_001_torchcrepe.csv
```

Run librosa pYIN:

```bash
pitchlab --input input/test_audio.wav --model librosa-pyin --range 100 500 --output output/pyin.csv
```

Run CREPE with range-masked activations:

```bash
pitchlab --input input/test_audio.wav --model crepe --range 100 500 --viterbi --output output/crepe.csv
```

Run Basic Pitch and save MIDI:

```bash
pitchlab \
  --input input/test_audio.wav \
  --model basic-pitch \
  --range 100 500 \
  --output output/basic_pitch_notes.csv \
  --midi-output output/basic_pitch.mid
```

## Python API

```python
from pitchlab.audio_io import load_audio
from pitchlab.registry import get_model

x, sr = load_audio("input/test_audio.wav", start=4.2, end=12.8)
model = get_model("torchcrepe")

df = model.predict(x, sr, freq_range=(100, 500), hop_ms=10)
print(df.head())
```

Also accepted:

```python
model.predict(x, sr, range=(100, 500))
model.predict(x, sr, fmin=100, fmax=500)
model.predict(x, sr, min_frequency=100, max_frequency=500)
```

## Registered model names

Native / strong range support:

- `torchcrepe`
- `crepe`
- `basic-pitch`
- `essentia-melodia`
- `librosa-pyin`
- `librosa-yin`
- `pyworld-dio`
- `pyworld-harvest`
- `yaapt`

Best-effort external adapters:

- `penn`
- `fcpe`
- `swiftf0`

The best-effort adapters attempt common import/function signatures and then range-filter outputs as a safety net. If those external packages use a different API, edit only the one corresponding model file.

## Output format

Framewise F0 models return CSV columns like:

```text
time,frequency_hz,confidence,voiced,model,low_hz,high_hz
```

Basic Pitch returns note-event rows like:

```text
time,start_time,end_time,pitch_midi,frequency_hz,velocity,amplitude,model,low_hz,high_hz
```

Unvoiced or rejected frequencies are written as `0.0`.

## Why range filtering exists in two places

When the backend exposes native frequency bounds, this package passes bounds directly into inference. Then it applies a final safety mask to the output. That is intentional.

Native bounds reduce the model's search space. Final masking prevents a bad backend output from leaking outside the user-specified section constraint.
