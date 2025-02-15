import torch
import torch.nn as nn
from .core import mlp, gru, scale_function, remove_above_nyquist, upsample, cnn_upsampling
from .core import amp_to_impulse_response, fft_convolve, transient_synth_frame
from .core import resample
import math

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


class DDSP(nn.Module):
    def __init__(self, hidden_size, n_bands, n_classes, sampling_rate,
                 block_size):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.in_mlps = nn.ModuleList([
            mlp(1, hidden_size, 3),  # For pitch
            mlp(1, hidden_size, 3),  # For loudness
            mlp(1, hidden_size, 3),  # For onset
        ])
        self.gru = gru(2, hidden_size)  
        self.out_mlp = mlp(hidden_size + 2, hidden_size, 3)

        self.n_bands = n_bands
        self.proj_matrices = nn.ModuleList([
            # nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, self.n_bands),
            nn.Linear(hidden_size, 2)
        ])

        self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))
        self.gamma_layer = nn.Linear(n_classes, hidden_size*2)
        self.beta_layer = nn.Linear(n_classes, hidden_size*2)

    def forward(self, spec_centroid, loudness, mh_distances):
        hidden = torch.cat([
            self.in_mlps[0](spec_centroid),
            self.in_mlps[1](loudness),
        ], -1)
        mh_distances = mh_distances.reshape(mh_distances.shape[0], 1, mh_distances.shape[1])
        mh_distances = mh_distances.repeat(1, loudness.shape[1], 1)
        gamma = self.gamma_layer(mh_distances) 
        beta = self.beta_layer(mh_distances)
        hidden = gamma * hidden + beta
        hidden = torch.cat([self.gru(hidden)[0], spec_centroid, loudness], -1)
        hidden = self.out_mlp(hidden)
        
        # # harmonic part
        # param = scale_function(self.proj_matrices[0](hidden))

        # total_amp = param[..., :1]
        # amplitudes = param[..., 1:]

        # amplitudes = remove_above_nyquist(
        #     amplitudes,
        #     pitch,
        #     self.sampling_rate,
        # )
        # amplitudes /= amplitudes.sum(-1, keepdim=True)
        # amplitudes *= total_amp

        # amplitudes = upsample(amplitudes, self.block_size)
        # pitch = upsample(pitch, self.block_size)

        # harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # noise part
        param = scale_function(self.proj_matrices[0](hidden) - 5)
        # freq_amp = param[..., :self.n_bands]
        # phase = param[..., self.n_bands:]
        # real_part = freq_amp * torch.cos(phase)
        # imaginary_part = freq_amp * torch.sin(phase)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        # Modelling transients by frames
        transient_param = scale_function(self.proj_matrices[1](hidden))
        transient_frequency = transient_param[..., 1:]
        transient_amps = transient_param[..., :1]
        transient_amps = upsample(transient_amps, self.block_size)
        upsampling_layer = cnn_upsampling(transient_frequency.shape[1], transient_frequency.shape[1], upscale_factor=4).to(transient_frequency)
        transient_frequency = upsampling_layer(transient_frequency)
        transient_frequency = transient_frequency.view(transient_frequency.shape[0], -1, 1)
        transient = transient_synth_frame(transient_amps, transient_frequency, int(self.block_size/4))

        signal = noise + transient
        #reverb part
        signal = self.reverb(signal)
        return signal, noise, transient


    def realtime_forward(self, spec_centroid, loudness, mh_distances):
        hidden = torch.cat([
            self.in_mlps[0](spec_centroid),
            self.in_mlps[1](loudness),
        ], -1)

        mh_distances = mh_distances.reshape(mh_distances.shape[0], 1, mh_distances.shape[1])
        mh_distances = mh_distances.repeat(1, loudness.shape[1], 1)
        gamma = self.gamma_layer(mh_distances)
        beta = self.beta_layer(mh_distances)
        hidden = gamma * hidden + beta

        gru_out, cache = self.gru(hidden, self.cache_gru)
        self.cache_gru.copy_(cache)

        hidden = torch.cat([gru_out, spec_centroid, loudness], -1)
        hidden = self.out_mlp(hidden)

        # noise part
        param = scale_function(self.proj_matrices[0](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        # Step 3: Transient synthesis
        transient_param = scale_function(self.proj_matrices[1](hidden))
        transient_frequency = transient_param[..., 1:]
        transient_amps = transient_param[..., :1]
        transient_amps = upsample(transient_amps, self.block_size)

        # Upsampling transient frequencies
        if not hasattr(self, 'upsampling_layer'):
            self.upsampling_layer = cnn_upsampling(
                transient_frequency.shape[1], transient_frequency.shape[1], upscale_factor=4
            ).to(transient_frequency)

        transient_frequency = self.upsampling_layer(transient_frequency)
        transient_frequency = transient_frequency.view(transient_frequency.shape[0], -1, 1)
        transient = transient_synth_frame(transient_amps, transient_frequency, int(self.block_size / 4))

        signal = noise + transient

        return signal