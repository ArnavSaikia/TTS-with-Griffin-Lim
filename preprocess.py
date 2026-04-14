# audio > mel spectrogram > saved tensor

import os
import librosa
import numpy as np
import pandas as pd

DATA_PATH = "dataset/LJSpeech-1.1"
OUTPUT_PATH = "data/mels"

os.makedirs(OUTPUT_PATH, exist_ok=True)

metadata = pd.read_csv(
    f"{DATA_PATH}/metadata.csv",
    sep="|",
    header=None
)

metadata.columns = ["id", "text", "raw_text"]
metadata["text"] = metadata["text"].str.strip()

for i, row in metadata.iterrows():

    wav_path = f"{DATA_PATH}/wavs/{row['id']}.wav"

    audio, sr = librosa.load(wav_path, sr=22050)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    )

    mel = np.log(mel + 1e-9)

    np.save(f"{OUTPUT_PATH}/{row['id']}.npy", mel)

    if i % 500 == 0:
        print(f"processed {i}")