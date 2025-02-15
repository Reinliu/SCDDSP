import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from ddsp.model_separate import DDSP
from effortless_config import Config
from os import path
from tqdm import tqdm
from ddsp.core import multiscale_fft, safe_log, mean_std_loudness
import soundfile as sf
from einops import rearrange
from ddsp.utils import get_scheduler
import numpy as np
import os
from ddsp.utils import create_date_folder
import sys
sys.path.append("/home/rein/Documents/frechet-audio-distance/frechet_audio_distance")
from fad import FrechetAudioDistance

def mahalanobis_distance(x, mean, covariance):

    x = torch.as_tensor(x)
    mean = torch.as_tensor(mean)
    covariance = torch.as_tensor(covariance)

    # Regularize the covariance matrix if it's singular
    if torch.det(covariance) == 0:
        covariance += torch.eye(covariance.shape[0], dtype=covariance.dtype, device=covariance.device) * 1e-10

    # Invert the covariance matrix
    inv_covariance = torch.linalg.inv(covariance)

    # Compute the Mahalanobis distance
    x_minus_mu = x - mean
    if x.dim() == 1:
        x_minus_mu = x_minus_mu.unsqueeze(0)  # Add batch dimension if necessary

    # Perform the dot product operations for the Mahalanobis distance
    left_term = torch.matmul(x_minus_mu, inv_covariance)
    mahalanobis = torch.sqrt(torch.sum(left_term * x_minus_mu, dim=1))

    if mahalanobis.numel() == 1:
        return mahalanobis.item()  # Return a Python float if a single value
    return mahalanobis

frechet = FrechetAudioDistance(
    model_name="clap",
    sample_rate=48000,
    submodel_name="630k-audioset",  # for CLAP only
    verbose=False,
    enable_fusion=False,            # for CLAP only
)
embedding_paths = '/home/rein/Documents/frechet-audio-distance/frechet_audio_distance/impact_embeddings'

class args(Config):
    CONFIG = "config_impact.yaml"
    STEPS = 500000
    DECAY_OVER = STEPS*0.8
    DEVICE = "cuda:0"
    SAVE_EPOCH = 20
    START_LR = 1e-4
    STOP_LR = 1e-5

class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir):
        super().__init__()
        self.signals = torch.from_numpy(np.load(path.join(out_dir, "signals.npy"))).to(device)
        self.spec_centroid = torch.from_numpy(np.load(path.join(out_dir, "spec_centroid.npy"))).to(device)
        self.loudness = torch.from_numpy(np.load(path.join(out_dir, "loudness.npy"))).to(device)
        self.mh_distances = torch.from_numpy(np.load(path.join(out_dir, "mh_distances.npy"))).to(device)
        self.onsets = torch.from_numpy(np.load(path.join(out_dir, "onsets.npy"))).to(device)

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        s = self.signals[idx]
        sc = self.spec_centroid[idx]
        l = self.loudness[idx]
        m = self.mh_distances[idx]
        o = self.onsets[idx]
        return s, sc, l, m, o

args.parse_args()
model_path = "/home/rein/Documents/ICDDSP/runs/impact-separate_23-11-2024_22"

with open(path.join(model_path, "config.yaml"), "r") as config:
    config = yaml.safe_load(config)

device = torch.device(args.DEVICE if torch.cuda.is_available() else "cpu")

min_value = torch.from_numpy(np.load('preprocessed-impact/min_value.npy')).to(device)
max_value = torch.from_numpy(np.load('preprocessed-impact/max_value.npy')).to(device)

model = DDSP(**config["model"]).to(device)
state = model.state_dict()
pretrained = torch.load(path.join(model_path, "state.pth"), map_location="cuda:0")
state.update(pretrained)
model.load_state_dict(state)
model.train()
# model.freeze_decoder(training=False)
# model.unfreeze_mh(training=True)

dataset = Dataset(config["preprocess"]["out_dir"])
dataloader = torch.utils.data.DataLoader(
    dataset,
    1,
    True,
    drop_last=True,
)

mean_loudness, std_loudness = mean_std_loudness(dataloader)
config["data"]["mean_loudness"] = mean_loudness
config["data"]["std_loudness"] = std_loudness

writer = SummaryWriter(model_path, flush_secs=20)

# Freeze the decoder
for param in model.decoder.parameters():
    param.requires_grad = False

# Unfreeze the mh_extractor parameters
for param in model.mh_extractor.parameters():
    param.requires_grad = True

opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.START_LR)
# opt = torch.optim.Adam(model.parameters(), lr=args.START_LR)


schedule = get_scheduler(
    len(dataloader),
    args.START_LR,
    args.STOP_LR,
    args.DECAY_OVER,
)

best_loss = float("inf")
mean_loss = 0
n_element = 0
step = 0
epochs = int(np.ceil(args.STEPS / len(dataloader)))

for e in tqdm(range(epochs)):
    for s, sc, l, m, o in dataloader:
        s = s.to(device)
        sc = sc.to(device)
        l = l.to(device)
        o = o.to(device)
        m = m.to(device)
        m = m.unsqueeze(-1)
        # m = torch.rand(1, config["model"]["n_classes"], 1).to(device)
        
        l = (l - mean_loudness) / std_loudness
        signal, noise, transient = model(sc, l, m)
        signal = signal.squeeze(-1)
        noise = noise.squeeze(-1)
        transient = transient.squeeze(-1)
        
        audio_embedding = frechet.get_embeddings(x=signal.detach().cpu().numpy(), sr=config["preprocess"]["sampling_rate"])
        audio_embedding = torch.from_numpy(audio_embedding).float().to(device)

        # Load group embeddings and calculate distances
        sorted_paths = sorted(os.listdir(embedding_paths))

        mh_distances = []
        for paths in sorted_paths:
            folder_path = os.path.join(embedding_paths, paths)
            mean_cov = torch.from_numpy(np.load(folder_path)).to(device)
            mean = mean_cov[0, :]
            cov = mean_cov[1:, :]
            dist = mahalanobis_distance(audio_embedding, mean, cov)
            mh_distances.append(dist)
        mh_distances = torch.tensor(mh_distances, requires_grad=True).to(device)
        normalized_mh_distances = (mh_distances - min_value) / (max_value - min_value)
        normalized_mh_distances = torch.reshape(normalized_mh_distances, (1,normalized_mh_distances.shape[0], 1))

        mse_loss = torch.nn.functional.mse_loss(m, normalized_mh_distances)
        mae_loss = torch.nn.functional.l1_loss(m, normalized_mh_distances)
        loss = (mse_loss+mae_loss).mean()
        # loss = mse_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar("loss", loss.item(), step)

        step += 1

        n_element += 1
        mean_loss += (loss.item() - mean_loss) / n_element

    if not e % args.SAVE_EPOCH: 
        writer.add_scalar("lr", schedule(e), e)
        writer.add_scalar("reverb_decay", model.decoder.reverb.decay.item(), e)
        writer.add_scalar("reverb_wet", model.decoder.reverb.wet.item(), e)
        # scheduler.step()
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                model.state_dict(),
                path.join(model_path, "finetuned_state.pth"),
            )

        mean_loss = 0
        n_element = 0

        audio = torch.cat([s, signal], -1).reshape(-1).detach().cpu().numpy()
        sf.write(path.join(model_path, f"signal_{e:06d}.wav"), audio, config["preprocess"]["sampling_rate"],)

        noises = torch.cat([s, noise], -1).reshape(-1).detach().cpu().numpy()
        sf.write(path.join(model_path, f"noise_{e:06d}.wav"), noises, config["preprocess"]["sampling_rate"],)

        transients = torch.cat([s, transient], -1).reshape(-1).detach().cpu().numpy()
        sf.write(path.join(model_path, f"transient_{e:06d}.wav"), transients, config["preprocess"]["sampling_rate"],)

    if not e % 1: 
        print('loss', loss)