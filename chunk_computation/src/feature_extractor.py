import torchaudio.compliance.kaldi as ta_kaldi
from typing import Tuple
import soundfile as sf
import numpy as np
import torch

def get_waveform(audio_path: str, normalization=True) -> Tuple[np.ndarray, int]:
    """Get the waveform and sample rate of a 16-bit mono-channel WAV or FLAC.
    adapted from https://github.com/pytorch/fairseq/blob/master/fairseq/data/audio/audio_utils.py
    Args:
        path_or_fp (str): the path or file-like object
        normalization (bool): Normalize values to [-1, 1] (Default: True)
    Returns:
        (numpy.ndarray): [n,] waveform array, (int): sample rate
   """   
    waveform, sample_rate = sf.read(audio_path)
    if normalization:
        waveform = normalize_rms(waveform, rms_level = 0.1)
    return waveform, sample_rate


def normalize_rms(signal: np.ndarray, rms_level: float = 0.1) -> np.ndarray:
    a = np.sqrt( (len(signal) * rms_level**2) / np.sum(signal**2) )
    return signal * a


def compute_mel_spectrograms(waveform: np.ndarray,sample_rate: int, options:dict) -> np.ndarray:
    """Compute mel spectrogram from waveform
    Args:
        waveform (np.ndarray): input waveform array
        sample_rate (float): sample rate
    """    
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    features = ta_kaldi.fbank(waveform, **options, sample_frequency=sample_rate)
    return features.numpy()

def compute_stft_spectrograms(waveform: np.ndarray,sample_rate: int, options:dict) -> np.ndarray:
    """Compute mel spectrogram from waveform
    Args:
        waveform (np.ndarray): input waveform array
        sample_rate (float): sample rate
    """    
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    features = ta_kaldi.spectrogram(waveform, **options, sample_frequency=sample_rate)
    return features.numpy()

def compute_mel_features(audio_file: str, options: dict) -> np.ndarray:
    """Compute features from audio file
    Args:
        audio_file (str): path of audio file
    Returns:
        features (np.ndarray): extracted features
    """
    waveform, sample_rate = get_waveform(audio_file)
    features = compute_mel_spectrograms(waveform, sample_rate, options)
    return features.T

def compute_stft_features(audio_file: str, options: dict) -> np.ndarray:
    """Compute features from audio file
    Args:
        audio_file (str): path of audio file
    Returns:
        features (np.ndarray): extracted features
    """
    waveform, sample_rate = get_waveform(audio_file)
    features = compute_stft_spectrograms(waveform, sample_rate, options)
    return features.T

