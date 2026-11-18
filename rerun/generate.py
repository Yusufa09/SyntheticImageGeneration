"""Generates one image per (mask, seed) in a manifest from one saved checkpoint.

    python generate.py --unet_dir RUN/checkpoints/finetune/final/unet \\
        --manifest RUN/manifests/heldout_gen.json --out_dir RUN/gen/heldout_ft

The model is always loaded from --unet_dir. The checkpoint's weight hash, stage
and prompt are written to OUT_DIR/manifest.json so evaluate.py can confirm which
model produced each set. Sampling follows generate_v3_image in v5-7.ipynb:
DPM-Solver++ with classifier-free guidance, a zero context latent, and fp32
live weights. Images that already exist are skipped, so the script can be rerun
after an interruption.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DPMSolverMultistepScheduler
from PIL import Image

import config as C
from data import generation_mask, mask_features
from model import decode, load_components
from utils import enable_tf32, load_json, pick_device, save_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unet_dir", required=True, help="a final/unet folder written by train.py")
    p.add_argument("--manifest", required=True, help="manifest from prepare.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--prompt_from", help="use the training prompt of this other final/unet folder")
    p.add_argument("--seeds", choices=["all", "first"], default="all", help="all seeds per mask, or only the first")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def checkpoint_meta(unet_dir):
    return load_json(os.path.join(os.path.dirname(os.path.normpath(unet_dir)), "train_config.json"))


def file_name(item_id, seed):
    return f"{item_id}_s{seed}.png"


@torch.no_grad()
def sample(m, cfg, jobs, cond, uncond, device):
    size = cfg.latent_size
    feats = np.stack([mask_features(generation_mask(mask_path, cfg)) for _, mask_path, _ in jobs])
    mask_lat = F.interpolate(torch.from_numpy(feats).to(device), size=(size, size), mode="nearest")
    b = len(jobs)

    # One generator per image, so each image depends only on its own seed.
    gen_device = device if device.type == "cuda" else torch.device("cpu")
    latents = torch.cat([
        torch.randn((1, cfg.latent_channels, size, size),
                    generator=torch.Generator(gen_device).manual_seed(seed), device=gen_device)
        for _, _, seed in jobs
    ]).to(device)
    extra = [mask_lat.repeat(2, 1, 1, 1)]
    if cfg.use_context_hint:
        extra.append(torch.zeros((2 * b, cfg.latent_channels, size, size), device=device))
    enc = torch.cat([uncond.repeat(b, 1, 1), cond.repeat(b, 1, 1)])

    sampler = DPMSolverMultistepScheduler.from_config(m.scheduler.config)
    sampler.set_timesteps(cfg.infer_steps, device=device)
    for t in sampler.timesteps:
        unet_in = torch.cat([torch.cat([latents, latents])] + extra, dim=1)
        noise_pred = m.unet(unet_in, t, encoder_hidden_states=enc).sample
        n_uncond, n_cond = noise_pred.chunk(2)
        latents = sampler.step(n_uncond + cfg.guidance_scale * (n_cond - n_uncond), t, latents).prev_sample

    images = decode(m.vae, latents)
    return [(img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for img in images]


def main():
    args = parse_args()
    meta = checkpoint_meta(args.unet_dir)
    cfg = C.TrainConfig.from_dict(meta["config"])
    prompt = checkpoint_meta(args.prompt_from)["config"]["prompt"] if args.prompt_from else cfg.prompt

    jobs = []
    for item in load_json(args.manifest)["items"]:
        seeds = item["seeds"] if args.seeds == "all" else item["seeds"][:1]
        jobs += [(item["id"], item["mask_path"], int(s)) for s in seeds]

    os.makedirs(args.out_dir, exist_ok=True)
    record_path = os.path.join(args.out_dir, "manifest.json")
    record = {
        "unet_dir": os.path.abspath(args.unet_dir),
        "weights_sha256": meta["weights_sha256"],
        "stage": meta["stage"],
        "prompt": prompt,
        "infer_steps": cfg.infer_steps,
        "guidance_scale": cfg.guidance_scale,
        "source_manifest": os.path.abspath(args.manifest),
        "items": [],
    }
    if os.path.exists(record_path):
        prev = load_json(record_path)
        if (prev["weights_sha256"], prev["prompt"]) != (record["weights_sha256"], prompt):
            sys.exit(f"{args.out_dir} holds images from a different checkpoint or prompt. Use a new --out_dir.")
    save_json(record, record_path)  # written first so a partial folder is still labelled

    todo = [j for j in jobs if not os.path.exists(os.path.join(args.out_dir, file_name(j[0], j[2])))]
    print(f"[generate] {meta['stage']} model, prompt '{prompt}': {len(jobs)} images, {len(todo)} to make")
    if todo:
        device = pick_device(args.device)
        enable_tf32()
        m = load_components(cfg, device, smoke=args.smoke, unet_dir=args.unet_dir)
        m.unet.eval()
        cond, uncond = m.text(prompt), m.text("")
        for i in range(0, len(todo), args.batch_size):
            chunk = todo[i:i + args.batch_size]
            for (item_id, _, seed), img in zip(chunk, sample(m, cfg, chunk, cond, uncond, device)):
                Image.fromarray(img).save(os.path.join(args.out_dir, file_name(item_id, seed)))
            print(f"[generate] {min(i + args.batch_size, len(todo))}/{len(todo)}", flush=True)

    record["items"] = [
        {"id": item_id, "seed": seed, "file": file_name(item_id, seed), "mask_path": mask_path}
        for item_id, mask_path, seed in jobs
    ]
    save_json(record, record_path)


if __name__ == "__main__":
    main()
