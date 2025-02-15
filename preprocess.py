import yaml
import pathlib
import librosa as li
from ddsp.core import extract_loudness, extract_onsets
from effortless_config import Config
import numpy as np
from tqdm import tqdm
import numpy as np
from os import makedirs, path
import torch
from scipy.io import wavfile
import sys
sys.path.append("/home/rein/Documents/frechet-audio-distance/frechet_audio_distance")
from fad import FrechetAudioDistance
from scipy.spatial import distance
import os

name = "footstep"

config_name = f"config_{name}.yaml"

frechet = FrechetAudioDistance(
    model_name="clap",
    sample_rate=48000,
    submodel_name="630k-audioset",  # for CLAP only
    verbose=False,
    enable_fusion=False,            # for CLAP only
)

embedding_paths = f'/home/rein/Documents/frechet-audio-distance/frechet_audio_distance/{name}_embeddings'

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
    n_frames = int(signal_length / block_size)

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

    # pitch, confidence, loudness, onsets = extract_features(x, sampling_rate, block_size, scale, offset_x)
    loudness = extract_loudness(x, sampling_rate, block_size)
    spec_cen = li.feature.spectral_centroid(y=x, sr=sr, hop_length=block_size)
    spec_cen = spec_cen[:, :-1]
    onsets = extract_onsets(x, sampling_rate, block_size)
    # onsets = onsets * loudness
    x = x.reshape(-1, signal_length)
    # pitch = pitch.reshape(x.shape[0], -1)
    # confidence = confidence.reshape(x.shape[0], -1)
    # mh_distances = mh_distances.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)
    onsets = onsets.reshape(x.shape[0], -1)

    return x, spec_cen, loudness, mh_distances, onsets


class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir):
        super().__init__()
        self.signals = np.load(path.join(out_dir, "signals.npy"))
        self.spec_centroid = np.load(path.join(out_dir, "spec_centroid.npy"))
        self.loudness = np.load(path.join(out_dir, "loudness.npy"))
        self.mh_distances = np.load(path.join(out_dir, "mh_distances.npy"))
        self.onsets = np.load(path.join(out_dir, "onsets.npy"))

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.signals[idx])
        sc = torch.from_numpy(self.spec_centroid[idx])
        l = torch.from_numpy(self.loudness[idx])
        mh = torch.from_numpy(self.mh_distances[idx])
        o = torch.from_numpy(self.onsets[idx])
        return s, sc, l, mh, o


def main():
    class args(Config):
        CONFIG = config_name

    args.parse_args()
    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    files = get_files(**config["data"])
    pb = tqdm(files)

    signals = []
    spec_centroid = []
    loudness = []
    mh_distances = []
    onsets = []

    for f in pb:
        pb.set_description(str(f))
        x, sc, l, mh, o = preprocess(f, **config["preprocess"])
        signals.append(x)
        spec_centroid.append(sc)
        loudness.append(l)
        mh_distances.append(mh)
        onsets.append(o)

    signals = np.concatenate(signals, 0).astype(np.float32)
    spec_centroid = np.expand_dims(np.concatenate(spec_centroid, 0).astype(np.float32), axis = -1)
    loudness = np.expand_dims(np.concatenate(loudness, 0).astype(np.float32), axis = -1)
    onsets = np.concatenate(onsets, 0).astype(np.float32)

    # Normalize the mh_distances to (0, 1)
    mh_distances = np.array(mh_distances, dtype=np.float32)
    # Normalize each feature (column) independently
    min_value = mh_distances.min(axis=0)  # Minimum values for each feature
    max_value = mh_distances.max(axis=0)  # Maximum values for each feature

    # Avoid division by zero if max and min are the same
    range_values = np.where(max_value - min_value == 0, 1, max_value - min_value)
    normalized_mh_distances = (mh_distances - min_value) / range_values
    print(normalized_mh_distances[:,1].max(), normalized_mh_distances.shape)

    out_dir = config["preprocess"]["out_dir"]
    config["model"]["n_classes"] = mh_distances.shape[1]
    with open(path.join("config_ft.yaml"), "w") as out_config:
        yaml.safe_dump(config, out_config)
        
    makedirs(out_dir, exist_ok=True)

    np.save(path.join(out_dir, 'min_value.npy'), min_value)
    np.save(path.join(out_dir, 'max_value.npy'), max_value)

    np.save(path.join(out_dir, "signals.npy"), signals)
    np.save(path.join(out_dir, "spec_centroid.npy"), spec_centroid)
    np.save(path.join(out_dir, "loudness.npy"), loudness)
    np.save(path.join(out_dir, "mh_distances.npy"), normalized_mh_distances)
    np.save(path.join(out_dir, "onsets.npy"), onsets)


if __name__ == "__main__":
    main()