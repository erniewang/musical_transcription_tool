import soundfile as sf
from scipy.signal import butter, sosfiltfilt
from pydub import AudioSegment

def high_pass(audio, sr, cutoff_hz, order, output_path):
    sos = butter(
        N=order,
        Wn=cutoff_hz,
        btype="highpass",
        fs=sr,
        output="sos"
    )

    filtered_audio = sosfiltfilt(sos, audio)
    sf.write(output_path, filtered_audio, sr)
    return filtered_audio

def segment_audio(audio_path, time_stamps):
    sound = AudioSegment.from_file(audio_path)
    time_stamps = [a * 1000 for a in time_stamps]
    time_stamps[-1] = sound.duration_seconds
    for i in range(1,len(time_stamps)):
        section = sound[time_stamps[i-1]:time_stamps[i]]
        section.export(f"section_{i}.mp3", format="mp3")