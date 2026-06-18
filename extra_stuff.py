import numpy as np
import ipywidgets as widgets
from IPython.display import display, Audio, clear_output

def sound_slider():
    sample_rate = 44100
    duration = 1.0

    NOTE_NAMES = ["C", "C#", "D", "E-", "E", "F", "F#", "G", "A-", "A", "B-", "B"]

    def hz_to_note(freq_hz):
        if freq_hz <= 0:
            return "None"

        midi = round(69 + 12 * np.log2(freq_hz / 440))
        note = NOTE_NAMES[midi % 12]
        octave = midi // 12 - 1
        return f"{note}{octave}"

    def label_text(freq_hz):
        return f"<b>Current frequency:</b> {freq_hz} Hz &nbsp; <b>Closest note:</b> {hz_to_note(freq_hz)}"

    freq_slider = widgets.IntSlider(
        value=440,
        min=0,
        max=2000,
        step=1,
        description="Hz:",
        continuous_update=True,
        layout=widgets.Layout(width="600px")
    )

    freq_label = widgets.HTML(value=label_text(freq_slider.value))

    play_button = widgets.Button(
        description="Play sine wave",
        icon="play"
    )

    audio_output = widgets.Output()

    def make_sine_wave(freq_hz):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        if freq_hz == 0:
            wave = np.zeros_like(t)
        else:
            wave = 0.25 * np.sin(2 * np.pi * freq_hz * t)

        fade_len = int(0.01 * sample_rate)
        fade = np.linspace(0, 1, fade_len)
        wave[:fade_len] *= fade
        wave[-fade_len:] *= fade[::-1]

        return wave

    def on_slider_change(change):
        freq_label.value = label_text(change["new"])

    freq_slider.observe(on_slider_change, names="value")

    def on_play_clicked(button):
        freq = freq_slider.value
        wave = make_sine_wave(freq)

        with audio_output:
            clear_output(wait=True)
            display(Audio(wave, rate=sample_rate, autoplay=True))

    play_button.on_click(on_play_clicked)

    ui = widgets.VBox([
        freq_label,
        freq_slider,
        play_button,
        audio_output
    ])

    display(ui)