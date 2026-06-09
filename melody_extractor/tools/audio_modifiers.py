def high_pass(audio, sr, cutoff_hz, order, output_path=None):
    from scipy.signal import butter, sosfiltfilt

    sos = butter(
        N=order,
        Wn=cutoff_hz,
        btype="highpass",
        fs=sr,
        output="sos"
    )

    filtered_audio = sosfiltfilt(sos, audio)
    if output_path is not None:
        import soundfile as sf

        sf.write(output_path, filtered_audio, sr)
    return filtered_audio


def split_audio(audio, sr, time_stamps):
    """Split an in-memory audio array using timestamps in seconds."""
    sections = []
    sample_count = len(audio)
    section_times = list(time_stamps)

    if section_times[-1] == -1:
        section_times[-1] = sample_count / sr

    for index in range(1, len(section_times)):
        start_seconds = section_times[index - 1]
        end_seconds = section_times[index]
        start_sample = max(0, int(start_seconds * sr))
        end_sample = min(sample_count, int(end_seconds * sr))

        sections.append(
            {
                "section_index": index,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "audio": audio[start_sample:end_sample],
                "sr": sr,
            }
        )

    return sections
