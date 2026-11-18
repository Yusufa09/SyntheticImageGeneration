"""Model construction for the rerun: Stable Diffusion 1.5 parts with a widened conv_in.

With --smoke, tiny randomly initialised stand-ins replace the real networks so the
pipeline can be tested on a laptop without downloading anything.
"""
import contextlib
import hashlib
from types import SimpleNamespace

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

LATENT_SCALING = 0.18215


class ClipText:
    """Frozen CLIP text encoder, called as text(prompt) -> [1, 77, 768]."""

    def __init__(self, base_model, device):
        from transformers import CLIPTextModel, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        self.model = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder").to(device).eval()
        self.model.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def __call__(self, prompt):
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return self.model(tokens.input_ids.to(self.device)).last_hidden_state


class FakeText:
    """Deterministic stand-in for CLIP in the smoke test."""

    dim = 16

    def __init__(self, device):
        self.device = device

    @torch.no_grad()
    def __call__(self, prompt):
        seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed(seed)
        return torch.randn(1, 77, self.dim, generator=g).to(self.device)


@contextlib.contextmanager
def _fixed_rng(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def _tiny_vae():
    with _fixed_rng(0):
        return AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            block_out_channels=(8, 8, 8, 8),
            layers_per_block=1,
            latent_channels=4,
            norm_num_groups=4,
            sample_size=64,
        )


def _tiny_unet():
    with _fixed_rng(1):
        return UNet2DConditionModel(
            sample_size=8,
            in_channels=4,
            out_channels=4,
            layers_per_block=1,
            block_out_channels=(8, 16, 16, 16),
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=FakeText.dim,
            attention_head_dim=2,
            norm_num_groups=8,
        )


def widen_conv_in(unet, in_channels, latent_channels=4):
    """Copies the pretrained latent weights and gives the new channels N(0, 1e-3), as in v1."""
    old = unet.conv_in
    new = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
    ).to(old.weight.device)
    with torch.no_grad():
        new.weight.zero_()
        new.bias.copy_(old.bias)
        new.weight[:, :latent_channels].copy_(old.weight)
        nn.init.normal_(new.weight[:, latent_channels:], mean=0.0, std=1e-3)
    unet.conv_in = new
    # Record the new width so save_pretrained / from_pretrained round-trip the model.
    unet.register_to_config(in_channels=in_channels)


def load_components(cfg, device, smoke=False, unet_dir=None):
    """VAE, text encoder, training scheduler and UNet.

    With unet_dir, the trained UNet is loaded from disk; otherwise the pretrained
    SD 1.5 UNet is loaded and its conv_in widened.
    """
    if smoke:
        vae = _tiny_vae()
        text = FakeText(device)
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
        )
    else:
        vae = AutoencoderKL.from_pretrained(cfg.base_model, subfolder="vae")
        text = ClipText(cfg.base_model, device)
        scheduler = DDPMScheduler.from_pretrained(cfg.base_model, subfolder="scheduler")
    vae = vae.to(device).eval()
    vae.requires_grad_(False)

    if unet_dir is not None:
        unet = UNet2DConditionModel.from_pretrained(unet_dir)
        if unet.config.in_channels != cfg.in_channels_unet:
            raise ValueError(
                f"{unet_dir} expects {unet.config.in_channels} input channels, config says {cfg.in_channels_unet}"
            )
    else:
        unet = _tiny_unet() if smoke else UNet2DConditionModel.from_pretrained(cfg.base_model, subfolder="unet")
        widen_conv_in(unet, cfg.in_channels_unet, cfg.latent_channels)
    return SimpleNamespace(vae=vae, text=text, scheduler=scheduler, unet=unet.to(device))


@torch.no_grad()
def encode(vae, pixels):
    return vae.encode(pixels).latent_dist.sample() * LATENT_SCALING


@torch.no_grad()
def decode(vae, latents):
    img = vae.decode(latents / LATENT_SCALING).sample
    return (img / 2 + 0.5).clamp(0, 1)


def min_snr_weights(scheduler, timesteps, gamma):
    alphas = scheduler.alphas_cumprod.to(timesteps.device)[timesteps]
    snr = alphas / (1 - alphas)
    return torch.minimum(snr, torch.full_like(snr, gamma)) / snr
