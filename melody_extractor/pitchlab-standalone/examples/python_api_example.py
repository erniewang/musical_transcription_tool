from pitchlab.audio_io import load_audio
from pitchlab.registry import get_model

x, sr = load_audio("input/test_audio.wav", start=0, end=20)
model = get_model("librosa-pyin")
df = model.predict(x, sr, range=(100, 500))
print(df.head())
df.to_csv("output/example_pyin_100_500.csv", index=False)
