import yaml
import pathlib
import librosa as li
from ddsp.core import extract_features, MelSpec
from effortless_config import Config
import numpy as np
from tqdm import tqdm
from os import makedirs, path
import torch
import sys
sys.path.append("/home/rein/Documents/frechet-audio-distance/frechet_audio_distance")
from fad import FrechetAudioDistance
from scipy.spatial import distance
import os
from sklearn.preprocessing import MinMaxScaler

device = "cuda:0"

frechet = FrechetAudioDistance(
    model_name="clap",
    sample_rate=48000,
    submodel_name="630k-audioset",  # for CLAP only
    verbose=False,
    enable_fusion=False,            # for CLAP only
)

embedding_paths = '/home/rein/Documents/frechet-audio-distance/frechet_audio_distance/fad_embeddings'

def mahalanobis_distance(x, mean_group, covariance_group):
    if np.linalg.det(covariance_group) == 0:
    # Regularize the covariance matrix if it's singular
        covariance_group += np.eye(covariance_group.shape[0]) * 1e-10

    # Compute the Mahalanobis distance
    mahalanobis_dist = distance.mahalanobis(x, mean_group, np.linalg.inv(covariance_group))
    return mahalanobis_dist


def get_files(data_location, extension, **kwargs):
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))


def preprocess(f, sampling_rate, block_size, signal_length, oneshot, **kwargs):
    audio = li.load(f, sr=sampling_rate)
    x = audio[0]
    sr = audio[1]
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))
    # n_frames = int(signal_length / block_size)

    if oneshot:
        x = x[..., :signal_length]
    scale = 10
    offset_x = 0.8
    
    audio_embedding = frechet.get_embeddings(x=audio, sr=sr).squeeze(0)
    audio_embedding = np.array(audio_embedding)

    # Load group embeddings and calculate distances
    sorted_paths = sorted(os.listdir(embedding_paths))

    mh_distances = []
    for path in sorted_paths:
        folder_path = os.path.join(embedding_paths, path)
        mean_cov = np.load(folder_path)
        mean = mean_cov[0, :]
        cov = mean_cov[1:, :]
        dist = mahalanobis_distance(audio_embedding, mean, cov)
        mh_distances.append(dist)
    mh_distances = np.array(mh_distances)

    pitch, confidence, loudness, onsets = extract_features(x, sampling_rate, block_size, scale, offset_x)
    onsets = onsets * loudness
    # melspecs = li.feature.melspectrogram(y=x, n_fft = 799, hop_length = block_size, sr=sampling_rate, n_mels=128, fmin = 20, fmax = 8000)
    # melspecs = np.transpose(melspecs)
    # melspecs = melspecs.reshape(x.shape[0], n_frames, 128)
    x = x.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1)
    confidence = confidence.reshape(x.shape[0], -1)
    # mh_distances = mh_distances.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)
    onsets = onsets.reshape(x.shape[0], -1)

    return x, pitch, confidence, mh_distances, loudness, onsets


def main():
    class args(Config):
        CONFIG = "config.yaml"

    args.parse_args()
    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    files = get_files(**config["data"])
    pb = tqdm(files)

    signals = []
    pitchs = []
    confidence = []
    mh_distances = []
    loudness = []
    onsets = []

    for f in pb:
        pb.set_description(str(f))
        x, p, c, m, l, o = preprocess(f, **config["preprocess"])
        signals.append(x)
        pitchs.append(p)
        confidence.append(c)
        loudness.append(l)
        onsets.append(o)
        mh_distances.append(m)

    signals = np.concatenate(signals, 0).astype(np.float32)
    pitchs = np.expand_dims(np.concatenate(pitchs, 0).astype(np.float32), axis = -1)
    confidence = np.expand_dims(np.concatenate(confidence, 0).astype(np.float32), axis = -1)
    loudness = np.expand_dims(np.concatenate(loudness, 0).astype(np.float32), axis = -1)
    onsets = np.expand_dims(np.concatenate(onsets, 0).astype(np.float32), axis = -1)
    # Normalize the mh_distances to (0, 1)
    mh_distances = np.array(mh_distances, dtype=np.float32)
    scaler = MinMaxScaler()
    normalized_mh_distances = scaler.fit_transform(mh_distances)
    mh_distances = np.expand_dims(normalized_mh_distances, axis=-1)

    out_dir = config["preprocess"]["out_dir"]
    makedirs(out_dir, exist_ok=True)

    try:
        scaler.fit(mh_distances)
        np.save(path.join(out_dir, 'min_values.npy'), scaler.min_)
        np.save(path.join(out_dir, 'scale_values.npy'), scaler.scale_)
    except:
        pass

    np.save(path.join(out_dir, "signals.npy"), signals)
    np.save(path.join(out_dir, "pitchs.npy"), pitchs)
    np.save(path.join(out_dir, "confidence.npy"), confidence)
    np.save(path.join(out_dir, "mh_distances.npy"), normalized_mh_distances)
    np.save(path.join(out_dir, "loudness.npy"), loudness)
    np.save(path.join(out_dir, "onsets.npy"), onsets)


if __name__ == "__main__":
    main()