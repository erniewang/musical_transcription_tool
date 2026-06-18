"""Pitch-run cleanup functions used by the refinement notebook."""

import numpy as np
import pandas as pd
from music21 import scale, pitch

def apply_to_runs(data_frames, function, **kwargs):
    """Apply one cleanup function to each DataFrame."""
    return [function(data_frame, **kwargs) for data_frame in data_frames]


def constrain_pitch_range(
    data_frame,
    frequency_column="frequency_hz",
    lowest_note=None,
    highest_note=None,
):
    """Set frequencies outside the selected range to NaN."""
    refined = data_frame.copy()

    if lowest_note is not None:
        refined.loc[refined[frequency_column] < lowest_note, frequency_column] = np.nan
    if highest_note is not None:
        refined.loc[refined[frequency_column] > highest_note, frequency_column] = np.nan

    return refined


def replace_zeroes_with_nan(data_frame, columns=("frequency_hz",)):
    """Replace zeroes in selected columns with NaN."""
    refined = data_frame.copy()
    for column in _selected_columns(refined, columns):
        refined[column] = refined[column].replace(0, np.nan)
    return refined


def interpolate_missing_values(
    data_frame,
    columns=("frequency_hz",),
    method="linear",
    fill_value=0,
):
    """Interpolate missing values in selected columns."""
    refined = data_frame.copy()
    for column in _selected_columns(refined, columns):
        refined[column] = refined[column].interpolate(method=method).fillna(fill_value)
    return refined


def _selected_columns(data_frame, columns):
    if columns is None:
        return data_frame.select_dtypes(include="number").columns

    if isinstance(columns, str):
        columns = (columns,)

    missing_columns = [column for column in columns if column not in data_frame.columns]
    if missing_columns:
        raise KeyError(f"Missing column(s): {', '.join(missing_columns)}")

    return columns

#also problematic with super low frequencies, there might be collisions
#trying to directly index is gonna get ugly. because if i want to get the music21 pitches exactly, then each note is gonna have a very specific hz. 
#find the closest object in a in distance that is closer to the thing
def auto_tune(df, scale: scale, lower_bound="a0", upper_bound="c7", other_pitches: list[str] | None = None):
    pitch_set = [p.frequency for p in scale.getPitches(lower_bound, upper_bound)]

    if other_pitches:
        for i in range(10):
            for p in other_pitches:
                new_pitch = pitch.Pitch(p+str(i))
                if new_pitch.frequency > pitch.Pitch(lower_bound).frequency and new_pitch.frequency < pitch.Pitch(upper_bound).frequency:
                    pitch_set.append(new_pitch.frequency)
    #double check that this shit works
    
    pitch_data = np.zeros(3000)
    for _pitch in pitch_set:
        pitch_data[int(_pitch)] = _pitch

    def find_nearest_good_note(freq):
        if not freq or pd.isna(freq):
            return 0
        down = int(freq) - 1
        up = int(freq) + 1
        while up < 3000 and down > 0:
            if pitch_data[up] != 0:
                return pitch_data[up]
            if pitch_data[down] != 0:
                return pitch_data[down]
            up+=1
            down-=1
        return 0
        
    df["frequency_hz"] = df["frequency_hz"].apply(find_nearest_good_note)
    #print(df["frequency_hz"][:20])
    return df


if __name__ == "__main__":
    df = pd.read_csv('test.csv')
    auto_tune(df,scale.MinorScale("d"))