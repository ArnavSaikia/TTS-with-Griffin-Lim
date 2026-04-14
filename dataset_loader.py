import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class LJDataset(Dataset):

    def __init__(self, metadata_path, mel_dir):

        self.metadata = pd.read_csv(
            metadata_path,
            sep="|",
            header=None
        )

        self.metadata.columns = ["id","text","normalized"]
        self.metadata["text"] = self.metadata["normalized"]

        self.mel_dir = mel_dir

        chars = sorted(set("".join(self.metadata["text"])))
        self.char_to_id = {c:i+1 for i,c in enumerate(chars)}
        self.vocab_size = len(self.char_to_id) + 1

    def encode_text(self, text):

        return [self.char_to_id[c] for c in text]

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self, idx):

        row = self.metadata.iloc[idx]

        text = torch.tensor(
            self.encode_text(row["text"]),
            dtype=torch.long
        )

        mel = np.load(
            f"{self.mel_dir}/{row['id']}.npy"
        )

        mel = torch.tensor(mel, dtype=torch.float)

        return text, mel