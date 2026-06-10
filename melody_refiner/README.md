# Melody Refiner

This folder owns the second half of the melody workflow: load pitch-run CSVs, apply cleanup, compare estimates, sonify results, and optionally save refined CSVs.

## Where To Make Changes

- `refinement.ipynb`: the notebook workflow. Edit folder paths, columns, pitch range, and cleanup lines here.
- `tools.py`: notebook helpers for loading CSVs, saving CSVs, plotting, and sonifying.
- `post_processing.py`: DataFrame cleanup functions that modify pitch-run values.

By default, section CSVs are combined into one continuous DataFrame per model before refinement. Set `COMBINE_SECTIONS_BY_MODEL = False` in the notebook to inspect or refine sections independently.

## Cleanup Lines

The cleanup cell is intentionally direct:

```python
pitch_runs = post_processing.apply_to_runs(...)
```

Comment out one line to skip that cleanup step. Edit `post_processing.py` only when the behavior of a cleanup function needs to change.

Refined exports are grouped by model under:

```text
experiments/pitch_runs_refined/
```
