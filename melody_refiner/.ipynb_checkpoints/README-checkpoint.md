# Melody Refiner

This folder owns the second half of the melody workflow: load pitch-run CSVs, apply cleanup transformations, compare estimates, sonify results, and optionally save refined CSVs.

## Where To Make Changes

- `config.py`: turn refinement transformations on/off and edit transformation parameters.
- `pipeline.py`: reusable workflow code used by the notebook.
- `tools/`: low-level DataFrame transformation helpers.
- `refinement.ipynb`: notebook interface for running the configured workflow. The first code cell exposes the transformation toggles.

## Common Toggles

Disable a transformation:

```python
{
    "name": "interpolate_missing_values",
    "enabled": False,
    "params": {...},
}
```

Change the usable pitch range:

```python
LOWEST_NOTE_HZ = 137
HIGHEST_NOTE_HZ = 515
```

Refined exports preserve section filenames and are grouped by model folder under `experiments/pitch_runs_refined/`.
