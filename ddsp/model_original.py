import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import mlp, gru, scale_function, remove_above_nyquist, upsample, MFCC, Normalize
from .core import harmonic_synth, variable_harmonic_synth, amp_to_impulse_response, fft_convolve, transient_synth, transient_synth_frame, TCN, ResTCN
from .core import resample
import math
import cached_conv as cc
# cc.use_cached_conv(True)

class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate
        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))
        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))
        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)
        return x

class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCN, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=self.padding, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        return self.relu(out)

class ResidualTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResidualTCNBlock, self).__init__()
        
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0, dilation=dilation),
            nn.ReLU()
        )
        self.residual = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)  # 1x1 convolution

    def forward(self, x):
        tcn_out = self.tcn(x)
        # Trim the original input (residual) to match the size of the TCN output
        residual_out = self.residual(x)[:, :, :tcn_out.size(2)]

        return tcn_out + residual_out


class Encoder(nn.Module):
    def __init__(self, hidden_size, sampling_rate, block_size):
        super().__init__()
        # Define your encoder architecture here
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        # CNN-based encoder:
        self.cnn1 = cc.Conv1d(128, 32, 3, stride=1, padding=(1,1))
        self.cnn2 = cc.Conv1d(32, 8, 3, stride=1, padding=(1,1))
        # #self.cnn3 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)
        # self.tcn_block1 = ResidualTCNBlock(128, 64, kernel_size=3, dilation=1)
        # self.tcn_block2 = ResidualTCNBlock(64, 32, kernel_size=3, dilation=2)
        # self.tcn_block3 = ResidualTCNBlock(32, 16, kernel_size=3, dilation=4)
        # self.tcn_block4 = ResidualTCNBlock(16, 8, kernel_size=3, dilation=8)
        # self.tcn_block5 = ResidualTCNBlock(8, 4, kernel_size=3, dilation=16)
        self.mu = nn.Linear(8, 1)
        self.logvar = nn.Linear(8, 1)

        # MLP+GRU based encoder:
        # self.z_in = mlp(128, hidden_size, 3)
        # self.z_gru = gru(1, hidden_size)
        # self.mu = nn.Linear(hidden_size, 1)   # Mean vector
        # self.logvar = nn.Linear(hidden_size, 1)   # Log variance vector
        
    def forward(self, melspecs):
        # TCN-based encoder
        # z = melspecs.permute(0, 2, 1)
        # z = self.tcn_block1(z)
        # z = self.tcn_block2(z)
        # z = self.tcn_block3(z)
        # z = self.tcn_block4(z)
        # z = self.tcn_block5(z)
        # z = z.permute(0, 2, 1)

        # CNN-based encoder:
        z = melspecs.permute(0, 2, 1)
        z = self.cnn1(z)
        z = F.relu(z)
        z = self.cnn2(z)
        z = F.relu(z)
        # z = self.cnn3(z)
        # z = F.relu(z)
        z = z.permute(0, 2, 1)

        # # MLP+GRU based encoder:
        # z = self.z_in(melspecs)
        # z = self.z_gru(z)[0]

        mu = self.mu(z)
        logvar = self.logvar(z)
        
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z_sampled = mu + eps * std
        return z_sampled, mu, logvar


class Decoder(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate, block_size):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.in_mlps = nn.ModuleList([
            mlp(1, hidden_size, 3),  # For pitch
            mlp(1, hidden_size, 3),  # For loudness
            mlp(1, hidden_size, 3),  # For onset
            mlp(1, hidden_size, 3)   # For z with dim 1
            #mlp(16, hidden_size, 3)  # For z with dim 16
        ])
        self.gru = gru(4, hidden_size)
        #self.out_mlp = mlp(hidden_size + 19, hidden_size, 3) # when z space is 16
        self.out_mlp = mlp(hidden_size + 4, hidden_size, 3) # when z space is 1

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
            nn.Linear(hidden_size, 2)
        ])

        self.reverb = Reverb(sampling_rate, sampling_rate)
        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def forward(self, pitch, confidence, loudness, onset, z):
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
            self.in_mlps[2](onset),
            self.in_mlps[3](z)
        ], -1)
        print(hidden.shape)
        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness, onset, z], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1] * confidence
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp
        #amplitudes *= harmonics

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        if self.training:
            # During training, we want to have randomness for better model robustness
            noise = torch.rand(
                impulse.shape[0],
                impulse.shape[1],
                self.block_size,
            ).to(impulse) * 2 - 1
        else:
            # During inference, we want deterministic outputs
            with torch.no_grad():
                torch.manual_seed(0)
                noise = torch.rand(
                    impulse.shape[0],
                    impulse.shape[1],
                    self.block_size,
                ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        # Modelling transients by frames
        transient_param = scale_function(self.proj_matrices[2](hidden))
        transient_amps = transient_param[..., :1]
        transient_frequency = transient_param[..., 1:]

        transient = transient_synth_frame(transient_amps, transient_frequency, self.sampling_rate, self.block_size)
        #transient = transient_synth(onset, self.sampling_rate, self.block_size)

        # add components together
        signal = harmonic + noise + transient
        #reverb part
        signal = self.reverb(signal)

        return signal, harmonic, noise, transient

    def realtime_forward(self, pitch, loudness):
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)

        gru_out, cache = self.gru(hidden, self.cache_gru)
        self.cache_gru.copy_(cache)

        hidden = torch.cat([gru_out, pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        n_harmonic = amplitudes.shape[-1]
        omega = torch.cumsum(2 * math.pi * pitch / self.sampling_rate, 1)

        omega = omega + self.phase
        self.phase.copy_(omega[0, -1, 0] % (2 * math.pi))

        omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)

        harmonic = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        return signal
    

class DDSP_VAE(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate, block_size):
        super(DDSP_VAE, self).__init__()
        self.encoder = Encoder(hidden_size, sampling_rate, block_size)
        self.decoder = Decoder(hidden_size, n_harmonic, n_bands, sampling_rate, block_size)
        
    def forward(self, pitch, confidence, melspecs, loudness, onset):
        z, mu, logvar = self.encoder(melspecs)
        recon_x, harmonic, noise, transient = self.decoder(pitch, confidence, loudness, onset, z)
        return recon_x, harmonic, noise, transient, mu, logvar

    def sample(self, mu=0, logvar=0):
        z = self.encoder.sample(mu, logvar)
        return self.decoder(z)