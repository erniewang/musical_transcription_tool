from setuptools import setup, find_packages

setup(
    name="pitchlab-standalone",
    version="0.1.0",
    description="Standalone pitch/F0 extraction adapters with section-level Hz range constraints.",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "librosa>=0.10",
        "soundfile>=0.12",
    ],
    extras_require={
        "torchcrepe": ["torch", "torchaudio", "torchcrepe"],
        "crepe": ["crepe", "tensorflow"],
        "basic-pitch": ["basic-pitch"],
        "essentia": ["essentia"],
        "pyworld": ["pyworld"],
        "yaapt": ["amfm-decompy"],
        "plot": ["matplotlib"],
        "midi": ["pretty_midi"],
        "all": [
            "torch", "torchaudio", "torchcrepe",
            "crepe", "tensorflow",
            "basic-pitch",
            "essentia",
            "pyworld",
            "amfm-decompy",
            "matplotlib",
            "pretty_midi",
        ],
    },
    entry_points={
        "console_scripts": [
            "pitchlab=pitchlab.cli:main",
        ]
    },
)
