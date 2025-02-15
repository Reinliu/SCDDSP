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
from ddsp.utils import create_date_folder

class args(Config):
    CONFIG = "config_impact.yaml"
    ROOT = "runs"
    STEPS = 500000
    DECAY_OVER = STEPS*0.8
    DEVICE = "cuda:0"
    SAVE_EPOCH = 400
    START_LR = 1e-4
    STOP_LR = 1e-5
    NAME = "impact-separate"

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
checkpoints_path = create_date_folder(args.ROOT, args.NAME)

with open(args.CONFIG, "r") as config:
    config = yaml.safe_load(config)

device = torch.device(args.DEVICE if torch.cuda.is_available() else "cpu")

model = DDSP(**config["model"]).to(device)
model.train()

dataset = Dataset(config["preprocess"]["out_dir"])
dataloader = torch.utils.data.DataLoader(
    dataset,
    config["train"]["batch"],
    True,
    drop_last=True,
)

mean_loudness, std_loudness = mean_std_loudness(dataloader)
config["data"]["mean_loudness"] = mean_loudness
config["data"]["std_loudness"] = std_loudness

writer = SummaryWriter(path.join(args.ROOT, args.NAME), flush_secs=20)

with open(path.join(checkpoints_path, "config.yaml"), "w") as out_config:
    yaml.safe_dump(config, out_config)

opt = torch.optim.Adam(model.parameters(), lr=args.START_LR)

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
        m = m.to(device)
        o = o.to(device)
        
        l = (l - mean_loudness) / std_loudness
        signal, noise, transient = model(sc, l, m)
        signal = signal.squeeze(-1)
        noise = noise.squeeze(-1)
        transient = transient.squeeze(-1)
        
        onset = o.repeat(1, config["preprocess"]["block_size"]).to(device)
        onset = onset*s
        t_loss = (transient-onset).abs().mean()

        ori_stft = multiscale_fft(
            s,
            config["train"]["scales"],
            config["train"]["overlap"],
        )
        rec_stft = multiscale_fft(
            signal,
            config["train"]["scales"],
            config["train"]["overlap"],
        )

        loss = 0
        # Linear, Log Loss
        for s_x, s_y in zip(ori_stft, rec_stft):
            # lin_loss = (s_x - s_y).abs().mean()
            log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
            loss = loss + log_loss 
        loss+=t_loss

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
                path.join(checkpoints_path, "state.pth"),
            )

        mean_loss = 0
        n_element = 0

        audio = torch.cat([s, signal], -1).reshape(-1).detach().cpu().numpy()
        sf.write(path.join(checkpoints_path, f"signal_{e:06d}.wav"), audio, config["preprocess"]["sampling_rate"],)

        noises = torch.cat([s, noise], -1).reshape(-1).detach().cpu().numpy()
        sf.write(path.join(checkpoints_path, f"noise_{e:06d}.wav"), noises, config["preprocess"]["sampling_rate"],)

        transients = torch.cat([s, transient], -1).reshape(-1).detach().cpu().numpy()
        sf.write(path.join(checkpoints_path, f"transient_{e:06d}.wav"), transients, config["preprocess"]["sampling_rate"],)

    if not e % 1: 
        print('loss', loss, 'noise mean', noise.mean(), 'transient loss', t_loss)