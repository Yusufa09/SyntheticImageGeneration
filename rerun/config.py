"""Configuration for the faithful rerun of the WG2Diff-ROI v3 model.

The values reproduce the Config and FineTuneConfig classes in
model_v5_saved/v5-7.ipynb, the notebook that produced arXiv:2609.13226 v1.
Two parts of that notebook are deliberately left out:

* EMA weights. They were saved but never used to generate the reported images,
  so the rerun trains and samples with the live weights, as v1 did.
* The ControlNet-style mask encoder. It was not in the executed forward pass.
"""
from dataclasses import asdict, dataclass, replace

# runwayml/stable-diffusion-v1-5 was removed from the Hugging Face Hub in 2024.
# This is the community mirror of the same weights.
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

BASE_PROMPT = "a realistic clinical photograph of a skin lesion"
FINETUNE_PROMPT = "a realistic clinical photograph of a leprosy ulcer"


@dataclass
class TrainConfig:
    stage: str = "base"
    base_model: str = BASE_MODEL

    # ROI and resolution
    image_size: int = 512
    latent_size: int = 64
    roi_margin_frac: float = 0.30
    min_roi_pixels: int = 24

    # Conditioning
    mask_channels: int = 3  # binary mask, inward distance, outward distance
    use_context_hint: bool = True
    context_blur_radius: float = 6.0
    ctx_drop_prob: float = 0.30
    latent_channels: int = 4
    prompt: str = BASE_PROMPT
    cfg_drop_prob: float = 0.10

    # Optimization (max_steps counts micro-batches, as in v1)
    lr: float = 1e-5
    weight_decay: float = 1e-2
    batch_size: int = 2
    grad_accum: int = 8
    num_workers: int = 4
    max_steps: int = 25000
    mixed_precision: bool = True
    use_min_snr: bool = True
    min_snr_gamma: float = 5.0
    freeze_down_blocks: bool = False

    # Sampling
    infer_steps: int = 30
    guidance_scale: float = 5.0
    seed: int = 42

    @property
    def in_channels_unet(self):
        ctx = self.latent_channels if self.use_context_hint else 0
        return self.latent_channels + self.mask_channels + ctx

    def to_dict(self):
        d = asdict(self)
        d["in_channels_unet"] = self.in_channels_unet
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def base_config():
    return TrainConfig()


def finetune_config():
    return TrainConfig(
        stage="finetune",
        prompt=FINETUNE_PROMPT,
        lr=5e-6,
        grad_accum=4,
        max_steps=5000,
        freeze_down_blocks=True,
    )


def smoke(cfg):
    """Shrinks a config so the whole pipeline runs on a laptop CPU in minutes."""
    return replace(
        cfg,
        image_size=64,
        latent_size=8,
        min_roi_pixels=4,
        grad_accum=2,
        num_workers=0,
        max_steps=8,
        infer_steps=3,
        mixed_precision=False,
    )
