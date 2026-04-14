import torch
from torch.utils.data import DataLoader

from dataset_loader import LJDataset
from simple_tts import SimpleTTS

dataset = LJDataset("data/processed")

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)