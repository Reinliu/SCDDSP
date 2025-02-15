import torch
import numpy as np
from IPython.display import Audio 
from ddsp import core
from preprocess import preprocess
import librosa
import matplotlib.pyplot as plt
import yaml
from effortless_config import Config
from os import path
from ddsp.model_phase_filter import DDSP
import soundfile as sf
import os

def process_directory(input_dir, output_dir, model, config, min_value, max_value):
    # Ensure the output directory structure mirrors the input
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Walk through the input directory
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.wav', '.mp3', '.flac', '.aiff')):
                file_path = path.join(root, filename)
                output_subdir = path.join(output_dir, path.relpath(root, input_dir))
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                
                # Process each file
                synthesized_audio = synthesize_audio(file_path, model, config, min_value, max_value)
                output_file_path = path.join(output_subdir, filename)
                sf.write(output_file_path, synthesized_audio, config["preprocess"]["sampling_rate"])
                print(f"Saved synthesized audio to {output_file_path}")

def synthesize_audio(audio_path, model, config, min_value, max_value):
    _, spec_centroid, loudness, mh_distances, _ = preprocess(
        audio_path,
        sampling_rate=config["preprocess"]["sampling_rate"],
        block_size=config["preprocess"]["block_size"],
        signal_length=config["preprocess"]["signal_length"],
        oneshot='True')

    spec_centroid = torch.tensor(spec_centroid, dtype=torch.float32).unsqueeze(-1)
    loudness = torch.tensor(loudness, dtype=torch.float32).unsqueeze(-1)
    mh_distances = torch.tensor(mh_distances, dtype=torch.float32).unsqueeze(0)

    normalized_mh_distances = (mh_distances - min_value) / (max_value - min_value)
    loudness_normalized = (loudness - config["data"]["mean_loudness"]) / config["data"]["std_loudness"]

    signal, noise, transient = model(spec_centroid, loudness_normalized, normalized_mh_distances)
    waveform = signal.detach().numpy().squeeze()
    return waveform

# Load model and configuration
model_path = "/home/rein/Documents/ICDDSP/runs/impacts-44100_13-11-2024_23"
with open(path.join(model_path, "config.yaml"), "r") as file:
    config = yaml.safe_load(file)

ddsp = DDSP(**config["model"])
state = ddsp.state_dict()
pretrained = torch.load(path.join(model_path, "state.pth"), map_location="cpu")
state.update(pretrained)
ddsp.load_state_dict(state)
ddsp.eval()

min_value = np.load('preprocessed/min_value.npy')
max_value = np.load('preprocessed/max_value.npy')

# Define input and output directories
input_directory = '/home/rein/Downloads/Sound_datasets/Impacts/Impact-set/test'
output_directory = '/home/rein/Documents/ICDDSP/ICDDSP-synth-Impact'

# Process all subdirectories
process_directory(input_directory, output_directory, ddsp, config, min_value, max_value)
