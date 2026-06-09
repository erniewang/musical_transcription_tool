#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:-input/test_audio.wav}
mkdir -p output

pitchlab --input "$INPUT" --model librosa-yin  --range 100 500 --output output/librosa_yin_100_500.csv
pitchlab --input "$INPUT" --model librosa-pyin --range 100 500 --output output/librosa_pyin_100_500.csv

# Install optional deps before these:
# pitchlab --input "$INPUT" --model torchcrepe --range 100 500 --output output/torchcrepe_100_500.csv
# pitchlab --input "$INPUT" --model crepe      --range 100 500 --output output/crepe_100_500.csv --viterbi
# pitchlab --input "$INPUT" --model pyworld-dio --range 100 500 --output output/pyworld_dio_100_500.csv
