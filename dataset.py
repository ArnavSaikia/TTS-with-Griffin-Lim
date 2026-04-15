import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
import librosa
from tokenizer import Tokenizer
import numpy as np

def load_wav(path_to_audio , sr=22050):
    audio, orig_sr = torchaudio.load(path_to_audio)

    if sr != orig_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=sr)

    return audio.squeeze(0)


def amp_to_db (x , min_db=-100):
    clip_val = 10 ** (min_db / 20) 
    #minimum amplitude value allowed by the decibel clip value
    #i think the min value should be clip_val tbh
    return 20 * torch.log10(torch.clamp(x, min = min_db))

def db_to_amp(x):
    return 10**(x/20)

#here the x is in dB
def normalize(x, min_db=-100, max_abs_value =4):
    x = (x - min_db) / -min_db
    x = 2 * max_abs_value * x - max_abs_value
    x = torch.clip(x, min= -max_abs_value, max= max_abs_value)
    return x

def denormalize(x, min_db=-100, max_abs_val=4):
    x = torch.clip(x, min= -max_abs_val, max=max_abs_val)
    x = (x + max_abs_val) / (2 * max_abs_val)
    x = x * -min_db + min_db
    return x

class AudioMelConversions:
    def __init__(self,
                 num_mels=80,
                 sampling_rate=22050, 
                 n_fft=1024, 
                 window_size=1024, 
                 hop_size=256,
                 fmin=0, 
                 fmax=8000,
                 center=False,
                 min_db=-100, 
                 max_scaled_abs=4):
        
        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.window_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.min_db = min_db
        self.max_scaled_abs = max_scaled_abs

        self.spec2mel = self._get_spec2mel_proj()
        self.mel2spec = torch.linalg.pinv(self.spec2mel)

    def _get_spec2mel_proj(self):
        mel = librosa.filters.mel(sr=self.sampling_rate, 
                                  n_fft=self.n_fft, 
                                  n_mels=self.num_mels, 
                                  fmin=self.fmin, 
                                  fmax=self.fmax)
        return torch.from_numpy(mel)
    
    def audio2mel(self, audio, do_norm=False):

        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)

        spectrogram = torch.stft(input=audio, 
                                 n_fft=self.n_fft, 
                                 hop_length=self.hop_size, 
                                 win_length=self.window_size, 
                                 window=torch.hann_window(self.window_size).to(audio.device), 
                                 center=self.center, 
                                 pad_mode="reflect", 
                                 normalized=False, 
                                 onesided=True,
                                 return_complex=True)
        
        spectrogram = torch.abs(spectrogram)
        
        mel = torch.matmul(self.spec2mel.to(spectrogram.device), spectrogram)

        mel = amp_to_db(mel, self.min_db)
        
        if do_norm:
            mel = normalize(mel, min_db=self.min_db, max_abs_val=self.max_scaled_abs)

        return mel