#!/usr/bin/env bash
# One-time setup on a fresh EC2 GPU instance
# (AMI: "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)").
#
#   bash setup_ec2.sh
#
# Creates ~/venv-rerun, installs the pinned packages, and downloads every model
# file the run needs, so the long run never waits on a download.
set -euo pipefail
cd "$(dirname "$0")"

nvidia-smi
python3 -m venv ~/venv-rerun || { sudo apt-get update -y && sudo apt-get install -y python3-venv && python3 -m venv ~/venv-rerun; }
source ~/venv-rerun/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
print("GPU:", torch.cuda.get_device_name(0))

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from config import BASE_MODEL
for cls, sub in [(AutoencoderKL, "vae"), (UNet2DConditionModel, "unet"), (DDPMScheduler, "scheduler"),
                 (CLIPTextModel, "text_encoder"), (CLIPTokenizer, "tokenizer")]:
    cls.from_pretrained(BASE_MODEL, subfolder=sub)
    print("downloaded", sub)

import lpips
lpips.LPIPS(net="alex", verbose=False)
from torchmetrics.image.kid import KernelInceptionDistance
KernelInceptionDistance(subset_size=2)
print("Setup complete.")
EOF
