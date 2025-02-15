import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import librosa
import crepe
import math
from ddsp import torchdct
import torchaudio.transforms as T
from scipy.signal import find_peaks, find_peaks_cwt
# import gin
# from typing import Callable, Optional, Sequence, Union
# import cached_conv as cc

def safe_log(x):
    return torch.log(x + 1e-7)

@torch.no_grad()
def mean_std_loudness(dataset):
    mean = 0
    std = 0
    n = 0
    for _, _, l, _, _ in dataset:
        n += 1
        mean += (l.mean().item() - mean) / n
        std += (l.std().item() - std) / n
    return mean, std


def multiscale_fft(signal, scales, overlap):
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts

def multiscale_melfft(signal, scales, overlap):
    mel_spectrogram = []
    for s in scales:
        mel_transform = T.MelSpectrogram(
            sample_rate = 16000, 
            hop_length = int(s*(1-overlap)),
            win_length = s, 
            #window_fn = torch.hann_window(s).to(signal),
            center = True,
            normalized = True,
            n_mels = 128,
            n_fft=s).to(signal)
        S = mel_transform(signal)
        mel_spectrogram.append(S)
    return mel_spectrogram

def MFCC(signal, n_mfcc, block_size, sample_rate):
    transform = T.MFCC(sample_rate=sample_rate, 
                   n_mfcc=n_mfcc, norm='ortho', log_mels=True, dct_type=2,
                   melkwargs={"n_fft": block_size*2, "hop_length": block_size, "win_length": block_size*2, "n_mels": 128, 
                              "f_min": 20, "f_max": 8000, "normalized":True, "center": False},).to(signal)
    mfcc = transform(signal)
    return mfcc

def MelSpec(signal, block_size, sample_rate):
    transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, win_length=1024, hop_length=block_size, 
                                 f_min=20, f_max=8000, n_mels=128, center=True, normalized=True)
    mel_specgram = transform(signal)
    return mel_specgram

def spec_centroid(signal, sample_rate):
    transform = T.SpectralCentroid(sample_rate=sample_rate)
    spec_centroid = transform(signal)
    return spec_centroid

def extract_MFCC(signal, n_mfcc, block_size, sample_rate):
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, hop_length=block_size, fmin = 20, fmax = 8000, 
                         n_mfcc=50, dct_type=2, norm='ortho', lifter=0)
    return mfcc


def Normalize(x):
    z_norm = nn.functional.normalize(x)
    return z_norm

def resample(x, factor: int):
    batch, frame, channel = x.shape
    x = x.permute(0, 2, 1).reshape(batch * channel, 1, frame)

    window = torch.hann_window(
        factor * 2,
        dtype=x.dtype,
        device=x.device,
    ).reshape(1, 1, -1)
    y = torch.zeros(x.shape[0], x.shape[1], factor * x.shape[2]).to(x)
    y[..., ::factor] = x
    y[..., -1:] = x[..., -1:]
    y = torch.nn.functional.pad(y, [factor, factor])
    y = torch.nn.functional.conv1d(y, window)[..., :-1]

    y = y.reshape(batch, channel, factor * frame).permute(0, 2, 1)

    return y


def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
    return signal.permute(0, 2, 1)


def remove_above_nyquist(amplitudes, pitch, sampling_rate):
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
    aa = (pitches < sampling_rate / 2).float() + 1e-4
    return amplitudes * aa


def scale_function(x):
    return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7


def custom_sigmoid(x, scale=10, offset_x=0.8):
    y = 1 / (1 + np.exp(-scale * (x - offset_x)))
    return y

def extract_features(signal, sampling_rate, block_size, scale, offset_x):
    pitch, confidence = extract_pitch(signal, sampling_rate, block_size, scale, offset_x)
    # mfcc = extract_MFCC(signal, 50, block_size, sampling_rate)
    # mfcc = mfcc[:-1, :-1]
    # harmonics = extract_harmonics(signal, sampling_rate, block_size)
    # harmonics = convert_to_one_hot(harmonics, max_harmonics)
    loudness = extract_loudness(signal, sampling_rate, block_size)
    onset = extract_onsets(signal, sampling_rate, block_size)
    return pitch, confidence, loudness, onset

def extract_pitch(signal, sampling_rate, block_size, scale, offset_x):
    length = signal.shape[-1] // block_size
    _, f0, confidence, _ = crepe.predict(
        signal,
        sampling_rate,
        step_size=int(1000 * block_size / sampling_rate),
        verbose=1,
        center=True,
        viterbi=True,
    )
    f0 = f0[:-1]
    confidence = confidence[:-1]
    confidence = custom_sigmoid(confidence, scale, offset_x)

    if f0.shape[-1] != length:
        f0 = np.interp(
            np.linspace(0, 1, length, endpoint=False),
            np.linspace(0, 1, f0.shape[-1], endpoint=False),
            f0,
        )
    return f0, confidence

def convert_to_one_hot(n_harmonics, max_harmonics):
    one_hot_vectors = np.zeros((len(n_harmonics), max_harmonics))
    for i, n in enumerate(n_harmonics):
        one_hot_vectors[i, :n] = 1
    return one_hot_vectors

def extract_harmonics(signal, sampling_rate, block_size):
    cqt = np.abs(librosa.cqt(signal, sr=sampling_rate, hop_length=block_size, n_bins = 350,bins_per_octave=70))
    D_harmonic, _ = librosa.decompose.hpss(cqt, kernel_size=31, margin=(1.0, 3.0))
    peak_counts = []

    for frame in D_harmonic.T:
        peaks, _ = find_peaks(frame, height=0.1, prominence=None, threshold=1, width=None, distance=None)
        peak_counts.append(len(peaks))

    n_harmonics = np.asarray(peak_counts)[:-1]
    return n_harmonics

def extract_loudness(signal, sampling_rate, block_size, n_fft=2048):
    S = librosa.stft(
        signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
    )
    S = np.log(abs(S) + 1e-7)
    f = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    a_weight = librosa.A_weighting(f)

    S = S + a_weight.reshape(-1, 1)

    S = np.mean(S, 0)[..., :-1]

    return S

def extract_onsets(signal, sampling_rate, block_size):
    hop_length = block_size
    win_length = 2 * hop_length
    n_fft=win_length
    S = librosa.stft(
        signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
    )

    _, D_percussive = librosa.decompose.hpss(S, margin=8)
    Percussive_signal = librosa.istft(D_percussive, hop_length=hop_length, win_length=win_length,n_fft=win_length, center=True)

    frames_ps = int(sampling_rate / block_size) # Frames per second
    n_frames = int(signal.shape[0] / block_size) # Total number of frames
    onsets_one_hot = np.zeros(n_frames) # Initialize the one-hot vector
    onsets = librosa.onset.onset_detect(
            y=Percussive_signal, 
            sr=sampling_rate, 
            hop_length=block_size,
            units='time')

    onsets = np.array(onsets * frames_ps, dtype=int)
    onsets_one_hot[onsets-1] = 1
    return onsets_one_hot

def HPSS(y):
    D = librosa.stft(y)
    H, P = librosa.decompose.hpss(D, kernel_size=31, margin=(1.0, 3.0))
    R = D-(H+P)
    y_harm = np.expand_dims(librosa.istft(H), 0)
    y_perc = np.expand_dims(librosa.istft(P), 0)
    y_resi = np.expand_dims(librosa.istft(R), 0)
    return y_harm, y_perc, y_resi

def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)


def gru(n_input, hidden_size):
    return nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)

def TCN(in_channels, out_channels, kernel_size, dilation):
    padding = (kernel_size - 1) * dilation
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
        nn.ReLU()
    )

def ResTCN(in_channels, out_channels, kernel_size, dilation):
    tcn = TCN(in_channels, out_channels, kernel_size, dilation)
    residual = nn.Conv1d(in_channels, out_channels, 1)  # 1x1 conv for channel matching
    return nn.Sequential(
        lambda x: tcn(x) + residual(x),
        nn.ReLU()
    )


class cnn_upsampling(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(cnn_upsampling, self).__init__()
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=upscale_factor,
            stride=upscale_factor
        )
        self.batch_norm1 = nn.BatchNorm1d(out_channels)  # Batch norm after upsampling
        self.relu1 = nn.ReLU()  # ReLU after batch normalization
        
        # Smoothing convolution layer
        self.smooth = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3,
            padding=1  # Maintain the same size
        )
        self.batch_norm2 = nn.BatchNorm1d(out_channels)  # Batch norm after smoothing
        self.relu2 = nn.ReLU()  # ReLU after batch normalization for smoothing

    def forward(self, x):
        x = self.upsample(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.smooth(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        
        return x


def harmonic_synth(pitch, amplitudes, sampling_rate):
    n_harmonic = amplitudes.shape[-1]
    omega = torch.cumsum(2 * math.pi * pitch / sampling_rate, 1)
    omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal


def variable_harmonic_synth(signal_length,
                            amplitudes,
                            sample_rate,
                            fundamental_freqs,      # Fundamental frequencies for each frame
                            n_harmonics,            # Number of harmonics for each frame
                            max_harmonics,          # Max number of harmonics for each frame
):
    time_frames = torch.linspace(0, signal_length, 401, dtype=torch.long)
    batch_size = fundamental_freqs.size(0)
    synthesized_signal = torch.zeros(batch_size, signal_length, dtype=torch.float64).to(fundamental_freqs)
    prev_phases = torch.zeros(batch_size, max_harmonics, dtype=torch.float64).to(fundamental_freqs)

    for frame_idx in range(len(time_frames) - 1):
        frame_start = time_frames[frame_idx]
        frame_end = time_frames[frame_idx + 1]
        if frame_end > signal_length:
            frame_end = signal_length

        f0 = fundamental_freqs[:, frame_idx, :]
        harmonics = n_harmonics[:, frame_idx, :]  # Get the number of harmonics for each batch
        
        # Generate a matrix of ones up to the max number of harmonics
        harmonics_matrix = torch.ones((harmonics.shape[0], max_harmonics)).to(f0)

        # Generate a cumulative sum along the second dimension
        cumulative_harmonics = torch.cumsum(harmonics_matrix, dim=1)

        # Generate a boolean mask based on the number of harmonics for each row
        mask = cumulative_harmonics <= harmonics[:, None]

        # Apply the mask to get the matrix of harmonics
        harmonics = mask * cumulative_harmonics

        #harmonics = torch.arange(1, harmonics + 1, dtype=torch.float64).to(f0)
        frequencies = f0 * harmonics

        time = torch.linspace(0, (frame_end - frame_start) / sample_rate, frame_end - frame_start, dtype=torch.float64).to(frequencies)
        time = time[None, :, None]  # Add new axes for broadcasting with batch and frequencies
        time = time.repeat(batch_size, 1, 1)

        # Use a mask to zero out contributions from harmonics beyond n_harmonic_batch
        #mask = torch.arange(max_harmonics).to(frequencies).unsqueeze(0) < n_harmonic_batch.unsqueeze(1)
        #mask = mask.to(frequencies).unsqueeze(1)  # Add extra dimension for broadcasting
        current_frame = (amplitudes[:, frame_idx, :] * torch.sin(2 * np.pi * frequencies * time + prev_phases)).sum(dim=2)
        #current_frame *= mask

        synthesized_signal[:, frame_start:frame_end] += current_frame
        # Calculate the next phase for each harmonic
        prev_phases += (2 * np.pi * frequencies * (frame_end - frame_start) / sample_rate).squeeze()

    return synthesized_signal.unsqueeze(2)

def transient_synth(transient_amps, sampling_rate, block_size, n_transients=400):
    transient_amps = transient_amps.repeat_interleave(block_size, dim=1)
    f_interval = (0.0, sampling_rate/2)
    d_f = (f_interval[1] - f_interval[0]) / (n_transients+2)
    frequencies = torch.linspace(f_interval[0] + d_f, f_interval[1] - d_f, n_transients, dtype=torch.float32).to(transient_amps)
    low_frequency = f_interval[0] + d_f
    low_frequency = low_frequency.unsqueeze(0).unsqueeze(-1)
    low_frequency = low_frequency.repeat(transient_amps.shape[0], transient_amps.shape[1], 1)
    omega = torch.cumsum(2 * math.pi * low_frequency / sampling_rate, 1)
    omegas = omega * frequencies.to(omega)
    transient = (torch.sin(omegas) * transient_amps).sum(-1, keepdim=True) # Multiplying learned amplitudes in the dct domain.
    transient = torchdct.idct(transient.contiguous()) #* transient_amps
    return transient

# def transient_synth_frame(transient_amps, sampling_rate, block_size):
#     transient_amps = transient_amps.repeat_interleave(block_size, dim=1)
#     n_frames = int(transient_amps.shape[1] / block_size)
#     frequency = torch.tensor(1).to(transient_amps)
#     frequency = frequency.repeat(block_size)
#     omega = torch.cumsum(2 * math.pi * frequency / sampling_rate, 0)
#     #omegas = omega * frequencies.to(omega)
#     transient = torch.sin(omega) 
#     transient = torchdct.idct(transient.contiguous()) 
#     transients = transient.unsqueeze(0).unsqueeze(-1)
#     transients = transients.repeat(transient_amps.shape[0], n_frames, 1)
#     transients*= transient_amps
#     return transients

def transient_synth_frame(transient_amps, transient_frequency, frame_length):
    batch_size, num_frames, _ = transient_frequency.shape
    time_frame = torch.linspace(0, 1, frame_length)
    time_expanded = time_frame.view(1, 1, -1).to(transient_amps)  # Shape: [1, 1, frame_length] for broadcasting
    # Generate sinusoids for each frame based on frequencies
    sinusoids = torch.sin(2 * math.pi * transient_frequency * time_expanded)  # Shape: [batch, num_frames, frame_length]
    # Initialize LinearDCT layer for IDCT transformation
    linear_idct = torchdct.LinearDCT(frame_length, 'idct', norm='ortho').to(transient_amps)
    # Reshape sinusoids to apply IDCT on each frame independently
    sinusoids_reshaped = sinusoids.view(batch_size * num_frames, frame_length)
    transients = linear_idct(sinusoids_reshaped)  # Shape: [batch * num_frames, frame_length]
    # Reshape back to the original batched structure
    transients = transients.view(batch_size, num_frames, frame_length)  # Shape: [batch, num_frames, frame_length]
    # Concatenate frames along the time dimension for each batch element
    transients = transients.reshape(batch_size, transient_amps.shape[1], 1)
    transients*=transient_amps
    return transients


def amp_to_impulse_response(amp, target_size):
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal, kernel):
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output



# @gin.configurable
# def normalization(module: nn.Module, mode: str = 'identity'):
#     if mode == 'identity':
#         return module
#     elif mode == 'weight_norm':
#         return weight_norm(module)
#     else:
#         raise Exception(f'Normalization mode {mode} not supported')


# class SampleNorm(nn.Module):

#     def forward(self, x):
#         return x / torch.norm(x, 2, 1, keepdim=True)


# class Residual(nn.Module):

#     def __init__(self, module, cumulative_delay=0):
#         super().__init__()
#         additional_delay = module.cumulative_delay
#         self.aligned = cc.AlignBranches(
#             module,
#             nn.Identity(),
#             delays=[additional_delay, 0],
#         )
#         self.cumulative_delay = additional_delay + cumulative_delay

#     def forward(self, x):
#         x_net, x_res = self.aligned(x)
#         return x_net + x_res


# class ResidualLayer(nn.Module):

#     def __init__(
#         self,
#         dim,
#         kernel_size,
#         dilations,
#         cumulative_delay=0,
#         activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)):
#         super().__init__()
#         net = []
#         cd = 0
#         for d in dilations:
#             net.append(activation(dim))
#             net.append(
#                 normalization(
#                     cc.Conv1d(
#                         dim,
#                         dim,
#                         kernel_size,
#                         dilation=d,
#                         padding=cc.get_padding(kernel_size, dilation=d),
#                         cumulative_delay=cd,
#                     )))
#             cd = net[-1].cumulative_delay
#         self.net = Residual(
#             cc.CachedSequential(*net),
#             cumulative_delay=cumulative_delay,
#         )
#         self.cumulative_delay = self.net.cumulative_delay

#     def forward(self, x):
#         return self.net(x)


# class DilatedUnit(nn.Module):

#     def __init__(
#         self,
#         dim: int,
#         kernel_size: int,
#         dilation: int,
#         activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)
#     ) -> None:
#         super().__init__()
#         net = [
#             activation(dim),
#             normalization(
#                 cc.Conv1d(dim,
#                           dim,
#                           kernel_size=kernel_size,
#                           dilation=dilation,
#                           padding=cc.get_padding(
#                               kernel_size,
#                               dilation=dilation,
#                           ))),
#             activation(dim),
#             normalization(cc.Conv1d(dim, dim, kernel_size=1)),
#         ]

#         self.net = cc.CachedSequential(*net)
#         self.cumulative_delay = net[1].cumulative_delay

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)


# class ResidualBlock(nn.Module):

#     def __init__(self,
#                  dim,
#                  kernel_size,
#                  dilations_list,
#                  cumulative_delay=0) -> None:
#         super().__init__()
#         layers = []
#         cd = 0

#         for dilations in dilations_list:
#             layers.append(
#                 ResidualLayer(
#                     dim,
#                     kernel_size,
#                     dilations,
#                     cumulative_delay=cd,
#                 ))
#             cd = layers[-1].cumulative_delay

#         self.net = cc.CachedSequential(
#             *layers,
#             cumulative_delay=cumulative_delay,
#         )
#         self.cumulative_delay = self.net.cumulative_delay

#     def forward(self, x):
#         return self.net(x)


# @gin.configurable
# class ResidualStack(nn.Module):

#     def __init__(self,
#                  dim,
#                  kernel_sizes,
#                  dilations_list,
#                  cumulative_delay=0) -> None:
#         super().__init__()
#         blocks = []
#         for k in kernel_sizes:
#             blocks.append(ResidualBlock(dim, k, dilations_list))
#         self.net = cc.AlignBranches(*blocks, cumulative_delay=cumulative_delay)
#         self.cumulative_delay = self.net.cumulative_delay

#     def forward(self, x):
#         x = self.net(x)
#         x = torch.stack(x, 0).sum(0)
#         return x


# class UpsampleLayer(nn.Module):

#     def __init__(
#         self,
#         in_dim,
#         out_dim,
#         ratio,
#         cumulative_delay=0,
#         activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)):
#         super().__init__()
#         net = [activation(in_dim)]
#         if ratio > 1:
#             net.append(
#                 normalization(
#                     cc.ConvTranspose1d(in_dim,
#                                        out_dim,
#                                        2 * ratio,
#                                        stride=ratio,
#                                        padding=ratio // 2)))
#         else:
#             net.append(
#                 normalization(
#                     cc.Conv1d(in_dim, out_dim, 3, padding=cc.get_padding(3))))

#         self.net = cc.CachedSequential(*net)
#         self.cumulative_delay = self.net.cumulative_delay + cumulative_delay * ratio

#     def forward(self, x):
#         return self.net(x)