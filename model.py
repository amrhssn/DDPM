import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # inv_freq[i] = 1 / 10000^(2i/dim)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer(name='inv_freq', tensor=inv_freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        B = t.shape[0]
        # [B,1]
        t = t.unsqueeze(1).float()
        # [B, dim/2]
        frequencies = t * self.inv_freq.unsqueeze(0)
        sin, cos = frequencies.sin(), frequencies.cos()            # each [B, dim/2]

        # interleave: even indices ← sin, odd indices ← cos
        emb = torch.zeros(B, self.dim, device=t.device, dtype=sin.dtype)
        emb[:, 0::2] = sin
        emb[:, 1::2] = cos
        return emb


class DDPM(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            T: int = 1_000,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.register_buffer(name='beta', tensor=torch.linspace(beta_start, beta_end, T))
        self.register_buffer(name='alpha', tensor=1 - self.beta)
        self.register_buffer(name='alpha_bar', tensor=torch.cumprod(self.alpha, dim=0))

        self.register_buffer(name='sqrt_alpha', tensor=torch.sqrt(self.alpha))
        self.register_buffer(name='sqrt_alpha_bar', tensor=torch.sqrt(self.alpha_bar))
        self.register_buffer(name='sqrt_one_minus_alpha_bar', tensor=torch.sqrt(1 - self.alpha_bar))

        self.time_encoder = TimeEmbedding(dim=hidden_dim)
        self.noise_decoder = nn.Sequential(
            nn.Linear(in_features=input_dim + hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=input_dim),
        )

    def diffusion_kernel(self, x_0: torch.Tensor, t: torch.Tensor):
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t]
        epsilon_0 = torch.randn_like(x_0)
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * epsilon_0
        return x_t

    def forward_encoder(self, x_0: torch.Tensor):
        batch_size = x_0.size(0)

        # sample t
        t = torch.randint(1, self.T, (batch_size,), device=x_0.device)

        # sample the noise
        noise = torch.randn_like(x_0)

        # run the diffusion process
        x_t = self.diffusion_kernel(x_0=x_0, t=t)
        return noise, x_t, t

    def reverse_decoder(self, x_t: torch.Tensor, t: torch.Tensor):
        time_embeddings = self.time_encoder(t)
        decoder_input = torch.cat([x_t, time_embeddings], dim=-1)
        noise_hat = self.noise_nn(decoder_input)
        return noise_hat

    def compute_loss(self, noise_hat: torch.Tensor, noise: torch.Tensor):
        loss = F.mse_loss(noise_hat, noise)
        return loss

    def sample(self):
        pass

    def forward(self, x_0: torch.Tensor):
        # forward process
        noise, x_t, t = self.forward_encoder(x_0=x_0)

        # backward process
        noise_hat = self.reverse_decoder(x_t=x_t, t=t)

        # compute the loss
        loss = self.compute_loss(noise_hat, noise)
        return loss

    def approximate_denoising_parameters(self, x, t):
        noise_hat = self.reverse_decoder(x, t)
        mean_first_term = (1.0 / self.sqrt_alpha[t]) * x
        mean_second_term = (1.0 - self.alpha[t]) / (self.sqrt_one_minus_alpha_bar[t] * self.sqrt_alpha[t]) * noise_hat
        mean = mean_first_term - mean_second_term
        variance = ((1.0 - self.alpha[t]) * (self.sqrt_one_minus_alpha_bar[t - 1])) / self.sqrt_one_minus_alpha_bar[t]
        std = torch.sqrt(variance)
        return mean, std

    @torch.no_grad()
    def sample(self, num_samples: int, device):
        x = torch.randn(num_samples, self.input_dim, device=device)
        for step in reversed(range(1, self.T)):
            t = torch.full((num_samples,), step, dtype=torch.long, device=device)
            mu, sigma = self.approximate_denoising_parameters(x, t)
            noise = torch.randn_like(x) if step > 1 else 0
            x = mu + sigma.unsqueeze(-1) * noise
        return x
