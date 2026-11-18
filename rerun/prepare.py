"""Builds the held-out split, the generation manifests and the real reference crops.

    python prepare.py --data_root ~/data/final_dataset --run_dir ~/runs/rerun1

Run once per run directory. An existing split is never overwritten, because the
fine-tuned model has to be evaluated on exactly the images it did not train on.

Outputs in run_dir:
  split.json                    train / held-out leprosy stems
  manifests/heldout_gen.json    held-out leprosy masks, two seeds each
  manifests/kaggle_gen.json     the v1 protocol: 250 Kaggle wound masks, seed 122 + i
  real/heldout/                 card-removed ROI crops of the held-out images
  real/v1_raw/                  the v1 real reference set (raw images, seed 122)
"""
import argparse
import os
import random

import config as C
from data import KAGGLE_MASKS, leprosy_dirs, list_images, load_mask01, real_roi, roi_box, stem
from utils import load_json, save_json

V1_SEED = 122  # GLOBAL_SEED in the v1 evaluation cell


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True, help="folder that contains Kaggle_images, AZH, resized_leprosy, ...")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--heldout_frac", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=2027)
    p.add_argument("--split_json", help="use this train/heldout split instead (e.g. a patient-level one)")
    p.add_argument("--n_kaggle", type=int, default=250)
    p.add_argument("--n_real_v1", type=int, default=250)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def make_split(args, cfg, cleaned, masks):
    stems = sorted(set(cleaned) & set(masks))
    if args.split_json:
        ext = load_json(args.split_json)
        train, held = sorted(ext["train"]), sorted(ext["heldout"])
        unknown = (set(train) | set(held)) - set(stems)
        if unknown or set(train) & set(held):
            raise SystemExit(f"{args.split_json}: overlapping or unknown stems, e.g. {sorted(unknown)[:5]}")
        kind = f"external split from {args.split_json}"
    else:
        order = stems[:]
        random.Random(args.split_seed).shuffle(order)
        n_held = round(args.heldout_frac * len(order))
        held, train = sorted(order[:n_held]), sorted(order[n_held:])
        kind = "random image-level split; images of the same patient can fall on both sides"
    valid = [s for s in held if roi_box(load_mask01(masks[s]), cfg) is not None]
    return {
        "kind": kind,
        "split_seed": args.split_seed,
        "heldout_frac": args.heldout_frac,
        "n_total": len(stems),
        "train": train,
        "heldout": held,
        "heldout_valid": valid,
    }


def main():
    args = parse_args()
    cfg = C.smoke(C.base_config()) if args.smoke else C.base_config()
    if args.smoke:
        args.n_kaggle, args.n_real_v1 = 6, 6
    for sub in ("manifests", "real/heldout", "real/v1_raw"):
        os.makedirs(os.path.join(args.run_dir, sub), exist_ok=True)

    d = leprosy_dirs(args.data_root)
    cleaned = {stem(p): p for p in list_images(d["cleaned"])}
    raw = {stem(p): p for p in list_images(d["raw"])}
    masks = {stem(p): p for p in list_images(d["masks"])}

    split_path = os.path.join(args.run_dir, "split.json")
    if os.path.exists(split_path):
        split = load_json(split_path)
        print(f"Using the existing split in {split_path}")
    else:
        split = make_split(args, cfg, cleaned, masks)
        save_json(split, split_path)
    print(f"Leprosy split: {len(split['train'])} train, {len(split['heldout'])} held out "
          f"({len(split['heldout_valid'])} usable for generation)")

    held_items = [
        {"id": s, "mask_path": os.path.abspath(masks[s]), "seeds": [10000 + i, 20000 + i]}
        for i, s in enumerate(split["heldout_valid"])
    ]
    save_json({"description": "held-out leprosy masks", "items": held_items},
              os.path.join(args.run_dir, "manifests", "heldout_gen.json"))

    # v1 protocol: random.seed(122) followed by random.sample over the sorted Kaggle masks.
    kaggle_all = list_images(os.path.join(args.data_root, KAGGLE_MASKS))
    kaggle_items = []
    for i, mp in enumerate(random.Random(V1_SEED).sample(kaggle_all, args.n_kaggle)):
        if roi_box(load_mask01(mp), cfg, check_size=False) is not None:
            kaggle_items.append({"id": f"{i:03d}_{stem(mp)}", "mask_path": os.path.abspath(mp), "seeds": [V1_SEED + i]})
    save_json({"description": "v1 protocol Kaggle wound masks", "items": kaggle_items},
              os.path.join(args.run_dir, "manifests", "kaggle_gen.json"))

    n_held = 0
    for s in split["heldout_valid"]:
        out = os.path.join(args.run_dir, "real", "heldout", f"{s}.png")
        if not os.path.exists(out):
            real_roi(cleaned[s], masks[s], cfg).save(out)
        n_held += 1

    # v1 real reference set: random.seed(122) followed by random.sample over the raw image names.
    common = sorted(set(raw) & set(masks))
    n_v1 = 0
    for i, name in enumerate(random.Random(V1_SEED).sample(common, args.n_real_v1)):
        out = os.path.join(args.run_dir, "real", "v1_raw", f"{i:03d}_{name}.png")
        if os.path.exists(out):
            n_v1 += 1
            continue
        crop = real_roi(raw[name], masks[name], cfg)
        if crop is not None:
            crop.save(out)
            n_v1 += 1

    summary = {
        "heldout_generation_masks": len(held_items),
        "kaggle_generation_masks": len(kaggle_items),
        "real_heldout_crops": n_held,
        "real_v1_crops": n_v1,
    }
    save_json(summary, os.path.join(args.run_dir, "prepare_summary.json"))
    print("Prepared:", summary)


if __name__ == "__main__":
    main()
