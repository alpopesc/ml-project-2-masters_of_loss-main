import os
import numpy as np
import csv
from feature_extractor import compute_mel_features, compute_stft_features

# Transform parameters
options_mel = {
    "frame_length" : 32, #ms
    "frame_shift" : 4, #ms
    "window_type" : "hamming",
    "num_mel_bins" : 129,
    "channel" : 0}

options_stft = {
    "frame_length" : 16, #ms
    "frame_shift" : 8, #ms
    "window_type" : "hamming",
    "channel" : 0}

# Paths to data
hc_path = "../../../data/HC/"
pd_path = "../../../data/PD/"
chunks_path = "../../../data/chunks/"

chunk_length = 500 #ms


def divide_audio(feats, options):
    """
    Divides spectrogram into 500 ms chunks

    :param feats: the spectrogram extraction
    :param options: the spectrogram options
    :return: a list of 500 ms chunks of the spectrogram
    """
    nb_frames_in_audio = feats.shape[1]
    nb_frames_per_chunk = int(chunk_length*nb_frames_in_audio / (options["frame_shift"]*(nb_frames_in_audio-1) + options["frame_length"]))
    
    i = 0
    chunks = []
    while ((i+1)*nb_frames_per_chunk < nb_frames_in_audio): # while there still exists another extra 500 ms chunk
        chunks.append(feats[:, i*nb_frames_per_chunk:(i+1)*nb_frames_per_chunk]) # append next chunk of 500 ms
        i += 1
        
    return chunks

def standardize(chunk):
    """
    Standardizes a chunk 

    :param chunk: a chunk of spectrogram
    :return: that chunk, standardized
    """ 
    return (chunk-chunk.mean())/(chunk.std())

def save_as_chunks(chunks_path, filename, feats, options, spec):
    """
    Saves all standardized chunks from spectrogram to filesystem as csv files

    :param speaker_path: path to speaker folder
    :param filename: name of original audio file
    :param feats: the spectrogram extraction
    :param options: the spectrogram options
    :param spec: the type spectrogram, 'mel' or 'stft'
    """ 
    chunks = divide_audio(feats, options) # Get spectrogram chunks of 500ms
    
    # Name each chunk, standardize and store in filesystem
    for i in range(len(chunks)):
        chunk_filename = f'{spec}_{filename[:-4]}_chunk_{i}.csv' # get rid of '.wav' in original filename
        stand = standardize(chunks[i])
        np.savetxt(f'{chunks_path}{spec}/{chunk_filename}', stand, delimiter=",")

def generate_csv(audio_path):
    """
    Generates all chunks as csv from audio files

    :param audio_path: path to audio files
    """ 
    for filename in os.listdir(audio_path):
        # Extract spectrograms
        mel_feats = compute_mel_features(f'{audio_path}{filename}', options_mel)
        stft_feats = compute_stft_features(f'{audio_path}{filename}', options_stft)
        
        # Save chunks to filesystem
        save_as_chunks(chunks_path, filename, mel_feats, options_mel, 'mel')
        save_as_chunks(chunks_path, filename, stft_feats, options_stft, 'stft')


# Run
generate_csv(hc_path)
generate_csv(pd_path)
