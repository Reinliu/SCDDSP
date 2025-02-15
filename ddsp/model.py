import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import mlp, gru, scale_function, remove_above_nyquist, upsample, MFCC, Normalize
from .core import harmonic_synth, variable_harmonic_synth, amp_to_impulse_response, fft_convolve, transient_synth_frame#, UpsampleLayer, ResidualBlock, ResidualStack, normalization
from .core import resample, upsample
import math
# import cached_conv as cc
# cc.use_cached_conv(True)

ratio = [4]

def mod_sigmoid(x):
    return 2 * torch.sigmoid(x)**2.3 + 1e-7

class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleLayer, self).__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=(kernel_size // 2) - 1, output_padding=stride-1)
        self.relu = nn.functional.normalize(nn.LeakyReLU(0.01))

    def forward(self, x):
        return self.relu(self.upsample(x))

class Decoder(nn.Module):
    def __init__(self, hidden_size, n_bands, sampling_rate, block_size, n_classes):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.in_mlps = nn.ModuleList([
            mlp(1, hidden_size, 3),  # For spectral centroid
            mlp(1, hidden_size, 3),  # For loudness
            mlp(7, hidden_size, 3)   # For mahalanobis distance
        ])
        self.gru = gru(2, hidden_size)
        self.out_mlp = mlp(hidden_size + 9, hidden_size, 3) # when z space is 1
        self.upsample = nn.Sequential(
            # First layer with dilation to expand the effective receptive field
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=5, padding=3, output_padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=4, padding=3, output_padding=1, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.ConvTranspose1d(16, 4, kernel_size=5, stride=4, padding=3, output_padding=1, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.ConvTranspose1d(4, 1, kernel_size=5, stride=2, padding=2, output_padding=1, dilation=1),
            nn.Tanh(),
        )
        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_bands), # Linear layer for noise synthesizer
            nn.Linear(hidden_size, 2),        # Linear layer for transient synthesizer
            nn.Linear(hidden_size, 65) # Linear layer for amplitudes of waveform synth
        ])

        self.reverb = Reverb(sampling_rate, sampling_rate)
        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

        self.gamma_layer = nn.Linear(n_classes, hidden_size*2)
        self.beta_layer = nn.Linear(n_classes, hidden_size*2)

        #         # Waveform synthesis part
        # capacity = 96
        # latent_size = 100
        # KERNEL_SIZE = [3]
        # DILATIONS = [
        #     [1, 3, 9],
        #     [1, 3, 9],
        #     [1, 3, 9],
        #     [1, 3],
        # ]
        # net = [
        #     normalization(
        #         cc.Conv1d(
        #             latent_size,
        #             2**len(ratio) * capacity,
        #             7,
        #             padding=cc.get_padding(7),
        #         ))
        # ]

        # recurrent_layer: Optional[Callable[[], nn.Module]] = None
        # if recurrent_layer is not None:
        #     net.append(
        #         recurrent_layer(
        #             dim=2**len(ratio) * capacity,
        #             cumulative_delay=net[0].cumulative_delay,
        #         ))

        # for i, r in enumerate(ratio):
        #     in_dim = 2**(len(ratio) - i) * capacity
        #     out_dim = 2**(len(ratio) - i - 1) * capacity

        #     net.append(
        #         UpsampleLayer(
        #             in_dim,
        #             out_dim,
        #             r,
        #             cumulative_delay=net[-1].cumulative_delay,
        #         ))
        #     net.append(
        #         ResidualStack(out_dim,kernel_sizes=KERNEL_SIZE, dilations_list=DILATIONS,
        #                       cumulative_delay=net[-1].cumulative_delay))

        # self.net = cc.CachedSequential(*net)

        # wave_gen = normalization(
        #     cc.Conv1d(out_dim, 64000, 7, padding=cc.get_padding(7)))

        # self.synth = cc.AlignBranches(
        #     wave_gen,
        #     cumulative_delay=self.net.cumulative_delay,
        # )


    def forward(self, spec_centroid, loudness, mh_distances):
        mh_distances = mh_distances.unsqueeze(1).repeat(1, loudness.shape[1], 1)

        hidden = torch.cat([
            self.in_mlps[0](spec_centroid),
            self.in_mlps[1](loudness),
            # self.in_mlps[2](mh_distances),
        ], -1)

        gamma = self.gamma_layer(mh_distances)  # shape: (batch, latent_dim)
        beta = self.beta_layer(mh_distances)    # shape: (batch, latent_dim)
        modulated_latent = gamma * hidden + beta
        hidden = torch.cat([self.gru(modulated_latent)[0], spec_centroid, loudness, mh_distances], -1)
        hidden = self.out_mlp(hidden)

        # noise part
        param = scale_function(self.proj_matrices[0](hidden) - 5)

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
        transient_param = scale_function(self.proj_matrices[1](hidden))
        transient_amps = transient_param[..., :1]
        transient_frequency = transient_param[..., 1:]

        transient = transient_synth_frame(transient_amps, transient_frequency, self.sampling_rate, self.block_size)
        #transient = transient_synth(onset, self.sampling_rate, self.block_size)

        # waveform synthesis
        param = scale_function(self.proj_matrices[2](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]
        total_amp = upsample(total_amp, self.block_size)

        x = amplitudes.permute(0, 2, 1)
        x = self.upsample(x)
        x = x.permute(0, 2, 1)
        waveform = x*total_amp
        # amplitudes = remove_above_nyquist(
        #     amplitudes,
        #     spec_centroid,
        #     self.sampling_rate,
        # )
        # amplitudes /= amplitudes.sum(-1, keepdim=True)
        # amplitudes *= total_amp
        #amplitudes *= harmonics

        # amplitudes = amplitudes.permute(0, 2, 1)
        # x = self.net(amplitudes)
        # waveform = self.synth(x)
        # print(waveform)
        # waveform = mod_sigmoid(total_amp) * torch.tanh(waveform)

        # add noisy components together
        signal = noise + transient
        #reverb part
        signal = self.reverb(signal)# + waveform

        return signal, noise, transient, waveform



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