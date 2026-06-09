from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_f0_csv(csv_path: str | Path, output_path: str | Path | None = None, *, title: str | None = None):
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError("CSV must contain a 'time' column")
    freq_col = None
    for candidate in ("frequency_hz", "frequency", "f0", "f0_hz", "pitch_hz"):
        if candidate in df.columns:
            freq_col = candidate
            break
    if freq_col is None:
        raise ValueError("CSV has no recognizable frequency column")

    plot_df = df.copy()
    plot_df.loc[plot_df[freq_col] <= 0, freq_col] = None

    fig = plt.figure(figsize=(12, 4))
    plt.plot(plot_df["time"], plot_df[freq_col], marker=".", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title or csv_path.name)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
    return fig
