from setuptools import find_packages, setup


setup(
    name="pitchlab-ernie",
    version="0.2.0",
    description="Simple raw-audio pitch and transcription model runners.",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "soundfile>=0.12",
    ],
    extras_require={
        "librosa": ["librosa>=0.10"],
        "torchcrepe": ["torch", "torchaudio", "torchcrepe"],
        "fcpe": ["torch", "librosa>=0.10", "torchfcpe"],
        "penn": ["torch", "penn"],
        "basic-pitch": ["basic-pitch"],
        "essentia": ["essentia"],
        "pyworld": ["pyworld"],
        "yaapt": ["amfm-decompy"],
        "midi": ["pretty_midi>=0.2.10"],
        "yourmt3": ["yourmt3", "huggingface-hub", "pretty_midi>=0.2.10", "librosa>=0.10"],
        "all": [
            "librosa>=0.10",
            "torch",
            "torchaudio",
            "torchcrepe",
            "torchfcpe",
            "penn",
            "basic-pitch",
            "essentia",
            "pyworld",
            "amfm-decompy",
            "pretty_midi>=0.2.10",
            "yourmt3",
            "huggingface-hub",
        ],
    },
    entry_points={
        "console_scripts": [
            "pitchlab=pitchlab.cli:main",
            "pitchlab-ernie=pitchlab.cli:main",
        ],
    },
)
