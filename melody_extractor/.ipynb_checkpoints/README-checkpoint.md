# Melody Extractor

This folder owns the first half of the melody workflow: prepare audio, split it into sections, run pitch models, and export pitch-run CSVs.

## Where To Make Changes

- `config.py`: turn prefilters and pitch models on/off, choose the input file, edit section timestamps, and set the pitch range.
- `pipeline.py`: reusable workflow code used by the notebook.
- `tools/`: low-level helpers for audio changes and CSV export.
- `melody_extraction.ipynb`: notebook interface for running the configured workflow.

## Common Toggles

Disable a prefilter:

```python
{"name": "high_pass", "enabled": False, "params": {...}}
```

Disable a model:

```python
{"name": "torchcrepe", "enabled": False}
```

Change the section that gets analyzed:

```python
SECTION_TO_ANALYZE = 2
```
