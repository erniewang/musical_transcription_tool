# Melody Extractor

This folder keeps the melody extraction workflow intentionally small:

- `melody_extraction.ipynb`: the readable notebook workflow.
- `tools.py`: helper functions for loading audio, displaying audio, splitting sections, filtering, running PitchLab models, cleaning output folders, and saving CSV files.
- `extractor_model.txt`: the notebook outline used to shape the workflow.

Edit the notebook when you want to change the audio file, section timestamps, filters, model pitch ranges, or model list. Edit `tools.py` only when the helper behavior itself needs to change.

Pitch-run exports are grouped by model:

```text
experiments/pitch_runs/
  torchcrepe/
    uzbek_dari_section_1_torchcrepe.csv
    uzbek_dari_section_2_torchcrepe.csv
  librosa-yin/
    uzbek_dari_section_1_librosa-yin.csv
```
