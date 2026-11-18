"""Trains the base (wound) or fine-tuned (leprosy) model for the rerun.

    python train.py --stage base --data_root DATA --out_dir RUN/checkpoints/base
    python train.py --stage finetune --data_root DATA --split RUN/split.json \\
        --init_unet RUN/checkpoints/base/final/unet --out_dir RUN/checkpoints/finetune

The loop follows train_v3 and finetune_v3_from_model in model_v5_saved/v5-7.ipynb.
Differences from v1:
* fine-tuning loads the base model from disk instead of reusing the in-memory
  object, so the base checkpoint is never modified;
* fine-tuning sees only the split's training images;
* progress is checkpointed every --ckpt_every micro-steps, and an interrupted run
  resumes from the last checkpoint when started again. The resumed run starts a
  new pass over the data, so it is not bit-identical to an uninterrupted one.

Finished output: OUT_DIR/final/{unet/, train_config.json, history.json, DONE}.
"""
import argparse
import os
import random
import shutil
import sys
import time
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader

import config as C
from data import ROIDataset, collate_skip, collect_pairs, leprosy_dirs, wound_pairs
from model import encode, load_components, min_snr_weights
from utils import enable_tf32, load_json, pick_device, save_json, set_seed, sha256_file

WEIGHTS_FILE = "diffusion_pytorch_model.safetensors"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["base", "finetune"], required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", help="split.json from prepare.py (finetune only)")
    p.add_argument("--init_unet", help="base model's final/unet folder (finetune only)")
    p.add_argument("--max_steps", type=int, help="override the configured number of micro-steps")
    p.add_argument("--ckpt_every", type=int, default=2000, help="micro-steps between resumable checkpoints")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="lower GPU memory at some speed cost; use only if training runs out of memory")
    p.add_argument("--device")
    p.add_argument("--smoke", action="store_true", help="tiny random models for the laptop smoke test")
    p.add_argument("--stop_at", type=int, help=argparse.SUPPRESS)  # smoke test: exit after this checkpoint
    return p.parse_args()


def training_pairs(args):
    if args.stage == "base":
        return wound_pairs(args.data_root)
    split = load_json(args.split)
    d = leprosy_dirs(args.data_root)
    return collect_pairs([d["cleaned"]], [d["masks"]], keep_stems=set(split["train"]))


def rng_state():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def set_rng_state(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def save_checkpoint(out_dir, unet, optimizer, scaler, step, history):
    """Writes OUT_DIR/last without ever leaving the run without a complete checkpoint."""
    tmp, last, old = (os.path.join(out_dir, n) for n in ("last_tmp", "last", "last_old"))
    shutil.rmtree(tmp, ignore_errors=True)
    unet.save_pretrained(os.path.join(tmp, "unet"), safe_serialization=True)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "step": step, "rng": rng_state()},
        os.path.join(tmp, "trainer_state.pt"),
    )
    save_json(history, os.path.join(tmp, "history.json"))
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(last):
        os.replace(last, old)
    os.replace(tmp, last)
    shutil.rmtree(old, ignore_errors=True)


def find_checkpoint(out_dir):
    for name in ("last", "last_old"):
        path = os.path.join(out_dir, name)
        if os.path.exists(os.path.join(path, "trainer_state.pt")) and os.path.exists(os.path.join(path, "unet", WEIGHTS_FILE)):
            return path
    return None


def save_final(out_dir, unet, cfg, args, history, n_pairs, resumed, hours):
    final, tmp = os.path.join(out_dir, "final"), os.path.join(out_dir, "final_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    unet.save_pretrained(os.path.join(tmp, "unet"), safe_serialization=True)
    sha = sha256_file(os.path.join(tmp, "unet", WEIGHTS_FILE))

    init_sha = None
    if args.init_unet:
        sha_file = os.path.join(os.path.dirname(os.path.normpath(args.init_unet)), "weights_sha256.txt")
        if os.path.exists(sha_file):
            init_sha = open(sha_file).read().strip()

    import diffusers
    import transformers

    meta = {
        "stage": args.stage,
        "config": cfg.to_dict(),
        "n_pairs": n_pairs,
        "data_root": os.path.abspath(args.data_root),
        "split": os.path.abspath(args.split) if args.split else None,
        "init_unet": os.path.abspath(args.init_unet) if args.init_unet else None,
        "init_unet_sha256": init_sha,
        "weights_sha256": sha,
        "resumed": resumed,
        "train_hours": round(hours, 3),
        "smoke": args.smoke,
        "versions": {"torch": torch.__version__, "diffusers": diffusers.__version__,
                     "transformers": transformers.__version__},
    }
    save_json(meta, os.path.join(tmp, "train_config.json"))
    save_json(history, os.path.join(tmp, "history.json"))
    with open(os.path.join(tmp, "weights_sha256.txt"), "w") as f:
        f.write(sha + "\n")
    with open(os.path.join(tmp, "DONE"), "w") as f:
        f.write(sha + "\n")
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    for name in ("last", "last_old"):
        shutil.rmtree(os.path.join(out_dir, name), ignore_errors=True)
    return sha


def main():
    args = parse_args()
    cfg = C.base_config() if args.stage == "base" else C.finetune_config()
    if args.smoke:
        cfg = C.smoke(cfg)
    if args.max_steps:
        cfg = replace(cfg, max_steps=args.max_steps)
    if args.stage == "finetune" and not (args.split and args.init_unet):
        sys.exit("--split and --init_unet are required for --stage finetune")
    if args.ckpt_every % cfg.grad_accum:
        sys.exit(f"--ckpt_every must be a multiple of grad_accum ({cfg.grad_accum})")

    if os.path.exists(os.path.join(args.out_dir, "final", "DONE")):
        print(f"[{args.stage}] already finished: {os.path.join(args.out_dir, 'final')}")
        return
    os.makedirs(args.out_dir, exist_ok=True)
    device = pick_device(args.device)
    enable_tf32()
    set_seed(cfg.seed)

    image_paths, mask_paths = training_pairs(args)
    if not image_paths:
        sys.exit(f"No image/mask pairs found under {args.data_root}")
    print(f"[{args.stage}] {len(image_paths)} image/mask pairs")
    loader = DataLoader(
        ROIDataset(image_paths, mask_paths, cfg),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_skip,
        drop_last=True,
    )

    m = load_components(cfg, device, smoke=args.smoke,
                        unet_dir=args.init_unet if args.stage == "finetune" else None)
    unet, vae, scheduler = m.unet, m.vae, m.scheduler
    unet.train()
    if cfg.freeze_down_blocks:
        for name, p in unet.named_parameters():
            if "down_blocks" in name:
                p.requires_grad_(False)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    params = [p for p in unet.parameters() if p.requires_grad]
    print(f"[{args.stage}] trainable UNet parameters: {sum(p.numel() for p in params):,} "
          f"/ {sum(p.numel() for p in unet.parameters()):,}")

    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = cfg.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    text_cond, text_uncond = m.text(cfg.prompt), m.text("")
    num_timesteps = scheduler.config.num_train_timesteps

    step, history = 0, {"step": [], "raw_loss": []}
    ckpt = find_checkpoint(args.out_dir)
    if ckpt:
        unet.load_state_dict(load_file(os.path.join(ckpt, "unet", WEIGHTS_FILE)))
        state = torch.load(os.path.join(ckpt, "trainer_state.pt"), map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        set_rng_state(state["rng"])
        step = state["step"]
        history = load_json(os.path.join(ckpt, "history.json"))
        print(f"[{args.stage}] resumed from {ckpt} at step {step}")
    optimizer.zero_grad(set_to_none=True)

    t0, start_step, window = time.time(), step, []
    while step < cfg.max_steps:
        for batch in loader:
            if batch is None:
                continue
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            ctx_pixels = batch["ctx_pixels"].to(device, non_blocking=True)
            mask_feats = batch["mask_feats"].to(device, non_blocking=True)
            bsz = pixel_values.shape[0]

            mask_lat = F.interpolate(mask_feats, size=(cfg.latent_size, cfg.latent_size), mode="nearest")
            latents = encode(vae, pixel_values)
            inputs_extra = [mask_lat]
            if cfg.use_context_hint:
                ctx = encode(vae, ctx_pixels)
                drop = (torch.rand(bsz, device=device) < cfg.ctx_drop_prob).float().view(bsz, 1, 1, 1)
                inputs_extra.append(ctx * (1.0 - drop))  # dropped samples see a zero context

            use_cond = torch.rand(bsz, device=device) > cfg.cfg_drop_prob
            enc = torch.where(use_cond[:, None, None], text_cond.repeat(bsz, 1, 1), text_uncond.repeat(bsz, 1, 1))
            timesteps = torch.randint(0, num_timesteps, (bsz,), device=device, dtype=torch.long)
            noise = torch.randn_like(latents)
            noisy = scheduler.add_noise(latents, noise, timesteps)
            unet_in = torch.cat([noisy] + inputs_extra, dim=1)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                noise_pred = unet(unet_in, timesteps, encoder_hidden_states=enc).sample
            mse = ((noise_pred.float() - noise) ** 2).mean(dim=(1, 2, 3))
            if cfg.use_min_snr:
                raw_loss = (min_snr_weights(scheduler, timesteps, cfg.min_snr_gamma) * mse).mean()
            else:
                raw_loss = mse.mean()
            scaler.scale(raw_loss / cfg.grad_accum).backward()

            loss_value = float(raw_loss.detach().cpu())
            history["step"].append(step + 1)
            history["raw_loss"].append(loss_value)
            window.append(loss_value)

            if (step + 1) % cfg.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == cfg.max_steps:
                rate = (step - start_step) / max(time.time() - t0, 1e-9)
                eta = (cfg.max_steps - step) / rate / 3600 if rate > 0 else float("nan")
                mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                print(f"[{args.stage}] step {step}/{cfg.max_steps}  loss {np.mean(window):.4f}  "
                      f"{rate:.2f} steps/s  ETA {eta:.1f} h  GPU mem {mem:.1f} GB", flush=True)
                window = []

            if step % args.ckpt_every == 0 and step < cfg.max_steps:
                save_checkpoint(args.out_dir, unet, optimizer, scaler, step, history)
                print(f"[{args.stage}] checkpoint saved at step {step}", flush=True)
                if args.stop_at and step >= args.stop_at:
                    sys.exit(3)

            if step >= cfg.max_steps:
                break

    sha = save_final(args.out_dir, unet, cfg, args, history, len(image_paths), bool(ckpt), (time.time() - t0) / 3600)
    print(f"[{args.stage}] finished. Weights sha256 {sha[:12]}…  saved to {os.path.join(args.out_dir, 'final')}")


if __name__ == "__main__":
    main()
