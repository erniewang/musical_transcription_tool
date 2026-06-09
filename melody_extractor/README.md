# Melody Extractor

This folder owns the first half of the melody workflow: prepare audio, split it into sections, run pitch models, and export pitch-run CSVs.

## Where To Make Changes

- `config.py`: turn prefilters and pitch models on/off, choose the input file, edit section timestamps, and set the pitch range.
- `pipeline.py`: reusable workflow code used by the notebook.
- `tools/`: low-level helpers for audio changes and CSV export.
- `melody_extraction.ipynb`: notebook interface for running the workflow. The first code cell exposes the main toggles so you can experiment without digging through later cells.

Raw audio, filtered audio, and split sections are kept in notebook variables. The notebook does not save temporary sound files.

Pitch-run exports are grouped by model:

```text
experiments/pitch_runs/
  torchcrepe/
    uzbek_dari_section_1_torchcrepe.csv
    uzbek_dari_section_2_torchcrepe.csv
  librosa-yin/
    uzbek_dari_section_1_librosa-yin.csv
```

## Common Toggles

Disable a prefilter:

```python
{"name": "high_pass", "enabled": False, "params": {...}}
```

Disable a model:

```python
{"name": "torchcrepe", "enabled": False}
```

Split audio is processed section by section in order. To skip splitting and run models on the whole filtered file:

```python
SPLIT_AUDIO = False
```
