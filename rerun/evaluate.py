"""Computes every metric for the rerun and writes RUN/metrics/summary.md.

    python evaluate.py --run_dir RUN

Held-out block (masks and real images the fine-tuned model never trained on)
  set-level LPIPS : real-real, base-base, ft-ft, real-base, real-ft, ft-base,
                    each anchor paired with 5 random partners (the v1 protocol)
  paired LPIPS    : the same mask and seed through both models, set against the
                    distance between two seeds of the same model for scale
  KID / FID       : real held-out crops against each generated set
Replication block (the v1 protocol: Kaggle wound masks, raw leprosy crops)
  set-level LPIPS : compared with the values reported in v1

Confidence intervals are cluster bootstraps over anchor images (set-level) or
masks (paired), 5,000 replicates, as in the paper. Per-pair values are saved in
metrics/pairs/ so every number can be recomputed.
"""
import argparse
import csv
import glob
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
from scipy.stats import wilcoxon

import config as C
from data import generation_mask
from utils import load_json, pick_device, save_json

V1 = {"real_vs_real": 0.672, "ft_vs_ft": 0.662, "ft_vs_base": 0.663, "real_vs_ft": 0.716}
GEN_SETS = ["heldout_base", "heldout_ft", "heldout_base_ftprompt", "kaggle_base", "kaggle_ft"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--k", type=int, default=5, help="random partners per anchor for set-level LPIPS")
    p.add_argument("--n_boot", type=int, default=5000)
    p.add_argument("--partner_seed", type=int, default=122)
    p.add_argument("--device")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


class GenSet:
    """A folder written by generate.py."""

    def __init__(self, folder):
        rec = load_json(os.path.join(folder, "manifest.json"))
        self.sha, self.stage, self.prompt, self.unet_dir = rec["weights_sha256"], rec["stage"], rec["prompt"], rec["unet_dir"]
        self.by_key, self.first, self.seeds, self.mask_path = {}, {}, defaultdict(list), {}
        for it in rec["items"]:
            path = os.path.join(folder, it["file"])
            if not os.path.exists(path):
                continue
            self.by_key[(it["id"], it["seed"])] = path
            self.seeds[it["id"]].append(it["seed"])
            self.first.setdefault(it["id"], path)
            self.mask_path[it["id"]] = it["mask_path"]
        self.files = list(self.by_key.values())


class LpipsScorer:
    def __init__(self, device, smoke):
        import lpips

        if smoke:  # random weights, no download
            net = lpips.LPIPS(net="alex", pretrained=False, pnet_rand=True, verbose=False)
        else:
            net = lpips.LPIPS(net="alex", verbose=False)
        self.net = net.to(device).eval()
        self.device = device
        self.tf = T.Compose([T.Resize((256, 256)), T.ToTensor()])  # as in the v1 evaluation
        self.cache = {}

    def _load(self, path):
        if path not in self.cache:
            self.cache[path] = self.tf(Image.open(path).convert("RGB")) * 2 - 1
        return self.cache[path]

    @torch.no_grad()
    def score(self, pairs, batch=64):
        out = []
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            a = torch.stack([self._load(x) for x, _ in chunk]).to(self.device)
            b = torch.stack([self._load(y) for _, y in chunk]).to(self.device)
            out.append(self.net(a, b).flatten().cpu())
        return torch.cat(out).numpy().astype(float) if out else np.array([])


class TinyFeatures(nn.Module):
    """Random feature extractor so the smoke test exercises KID/FID without Inception weights."""

    num_features = 16

    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.net = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x.float() / 255.0)


def set_pairs(anchors, partners, k, seed):
    rng = random.Random(seed)
    pairs = []
    for a in anchors:
        pool = [b for b in partners if b != a]
        pairs += [(a, b) for b in rng.sample(pool, min(k, len(pool)))]
    return pairs


def cluster_ci(values, clusters, n_boot, seed=0):
    """95% CI of the mean, resampling whole clusters with replacement."""
    values = np.asarray(values, float)
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    sums, counts = np.bincount(inv, weights=values), np.bincount(inv)
    picks = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    boots = sums[picks].sum(1) / counts[picks].sum(1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def describe(name, block, description, values, clusters, n_boot):
    lo, hi = cluster_ci(values, clusters, n_boot) if len(values) else (float("nan"), float("nan"))
    return {
        "name": name, "block": block, "description": description,
        "n_pairs": int(len(values)), "n_clusters": int(len(set(clusters))),
        "mean": float(np.mean(values)) if len(values) else float("nan"),
        "sd": float(np.std(values)) if len(values) else float("nan"),
        "median": float(np.median(values)) if len(values) else float("nan"),
        "ci95": [lo, hi],
    }


def paired_test(name, description, x_by_mask, y_by_mask, n_boot):
    """Mean of x - y over masks, with a bootstrap CI and a Wilcoxon signed-rank test."""
    ids = sorted(set(x_by_mask) & set(y_by_mask))
    d = np.array([x_by_mask[i] - y_by_mask[i] for i in ids])
    if len(d) == 0:
        return {"name": name, "description": description, "n_masks": 0}
    lo, hi = cluster_ci(d, ids, n_boot)
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:
        p = float("nan")
    return {"name": name, "description": description, "n_masks": len(ids),
            "mean_x": float(np.mean([x_by_mask[i] for i in ids])), "mean_y": float(np.mean([y_by_mask[i] for i in ids])),
            "mean_diff": float(d.mean()), "ci95": [lo, hi], "wilcoxon_p": p}


def load_uint8(paths, batch=32):
    for i in range(0, len(paths), batch):
        arrs = [np.array(Image.open(p).convert("RGB")) for p in paths[i:i + batch]]
        yield torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()


def kid_fid(real, gen, device, smoke):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    subset = min(100, len(real), len(gen))
    kid = KernelInceptionDistance(feature=TinyFeatures() if smoke else 2048, subset_size=subset, subsets=100).to(device)
    fid = FrechetInceptionDistance(feature=TinyFeatures() if smoke else 2048).to(device)
    for paths, is_real in ((real, True), (gen, False)):
        for x in load_uint8(paths):
            kid.update(x.to(device), real=is_real)
            fid.update(x.to(device), real=is_real)
    kid_mean, kid_std = kid.compute()
    return {"n_real": len(real), "n_gen": len(gen), "kid_subset_size": subset,
            "kid_mean": float(kid_mean), "kid_std": float(kid_std), "fid": float(fid.compute())}


def check_provenance(sets, run_dir):
    base = {sets[n].sha for n in GEN_SETS if n.endswith("base") or n.endswith("ftprompt")}
    ft = {sets[n].sha for n in GEN_SETS if n.endswith("_ft")}
    stages_ok = all(sets[n].stage == ("finetune" if n.endswith("_ft") else "base") for n in GEN_SETS)
    if len(base) != 1 or len(ft) != 1 or not stages_ok:
        raise SystemExit("Generated folders disagree about which checkpoint produced them. Regenerate them.")
    if base == ft:
        raise SystemExit("The base and fine-tuned sets came from the same weights.")
    ft_meta = load_json(os.path.join(run_dir, "checkpoints", "finetune", "final", "train_config.json"))
    (base_sha,), (ft_sha,) = base, ft
    if ft_meta["init_unet_sha256"] not in (None, base_sha):
        raise SystemExit("The fine-tuned model was not started from this base checkpoint.")
    return {"base_sha256": base_sha, "finetune_sha256": ft_sha,
            "finetune_started_from_base": ft_meta["init_unet_sha256"] == base_sha}


def examples_figure(run_dir, hb, hf, ids, out_path, n=6, tile=192):
    """Mask, real held-out crop, base output, fine-tuned output and a second fine-tuned seed."""
    cfg = C.TrainConfig.from_dict(load_json(os.path.join(run_dir, "checkpoints", "finetune", "final", "train_config.json"))["config"])
    labels = ["mask", "real (held out)", "base", "fine-tuned", "fine-tuned, seed 2"]
    rows = [i for i in ids if len(hf.seeds[i]) > 1][:n]
    canvas = Image.new("RGB", (tile * len(labels), 24 + tile * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for c, label in enumerate(labels):
        draw.text((c * tile + 6, 6), label, fill=(0, 0, 0))
    for r, i in enumerate(rows):
        s0, s1 = hf.seeds[i][0], hf.seeds[i][1]
        mask = Image.fromarray((generation_mask(hf.mask_path[i], cfg) * 255).astype(np.uint8)).convert("RGB")
        tiles = [mask, Image.open(os.path.join(run_dir, "real", "heldout", f"{i}.png")),
                 Image.open(hb.by_key[(i, s0)]), Image.open(hf.by_key[(i, s0)]), Image.open(hf.by_key[(i, s1)])]
        for c, im in enumerate(tiles):
            canvas.paste(im.convert("RGB").resize((tile, tile)), (c * tile, 24 + r * tile))
    canvas.save(out_path)


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_markdown(path, summary):
    lines = ["# Rerun results", "",
             f"Base weights `{summary['provenance']['base_sha256'][:12]}`, fine-tuned weights "
             f"`{summary['provenance']['finetune_sha256'][:12]}`; fine-tuning started from this base: "
             f"{summary['provenance']['finetune_started_from_base']}.", ""]
    for block, title in (("heldout_set", "Held-out masks: set-level LPIPS"),
                         ("heldout_paired", "Held-out masks: paired LPIPS (same mask)"),
                         ("replication", "Replication of the v1 protocol")):
        lines += [f"## {title}", "", "| comparison | mean | SD | 95% CI | pairs | clusters | v1 |",
                  "|---|---|---|---|---|---|---|"]
        for r in summary["lpips"]:
            if r["block"] == block:
                v1 = V1.get(r["name"].replace("rep_", "")) if block == "replication" else None
                lines.append(f"| {r['description']} | {fmt(r['mean'])} | {fmt(r['sd'])} | "
                             f"[{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}] | {r['n_pairs']} | {r['n_clusters']} | {fmt(v1)} |")
        lines.append("")
    lines += ["## Does fine-tuning change the output more than a new seed does?", "",
              "| test | mean x | mean y | mean x − y | 95% CI | Wilcoxon p | masks |", "|---|---|---|---|---|---|---|"]
    for t in summary["tests"]:
        if t.get("n_masks"):
            lines.append(f"| {t['description']} | {fmt(t['mean_x'])} | {fmt(t['mean_y'])} | {fmt(t['mean_diff'])} | "
                         f"[{fmt(t['ci95'][0])}, {fmt(t['ci95'][1])}] | {fmt(t['wilcoxon_p'], 4)} | {t['n_masks']} |")
    lines += ["", "## KID and FID against real held-out crops", "",
              "KID is the primary number; FID is biased at this sample size.", "",
              "| generated set | KID (mean ± SD) | FID | real | generated |", "|---|---|---|---|---|"]
    for name, r in summary["kid_fid"].items():
        lines.append(f"| {name} | {fmt(r['kid_mean'], 4)} ± {fmt(r['kid_std'], 4)} | {fmt(r['fid'], 1)} | {r['n_real']} | {r['n_gen']} |")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    if args.smoke:
        args.n_boot = min(args.n_boot, 200)
    run = args.run_dir
    device = pick_device(args.device)
    out = os.path.join(run, "metrics")
    os.makedirs(os.path.join(out, "pairs"), exist_ok=True)

    sets = {n: GenSet(os.path.join(run, "gen", n)) for n in GEN_SETS}
    provenance = check_provenance(sets, run)
    real_held = sorted(glob.glob(os.path.join(run, "real", "heldout", "*.png")))
    real_v1 = sorted(glob.glob(os.path.join(run, "real", "v1_raw", "*.png")))
    scorer = LpipsScorer(device, args.smoke)
    rows = []

    def run_pairs(name, block, description, pairs, clusters):
        values = scorer.score(pairs)
        with open(os.path.join(out, "pairs", f"{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["a", "b", "cluster", "lpips"])
            w.writerows((os.path.basename(a), os.path.basename(b), c, f"{v:.6f}") for (a, b), c, v in zip(pairs, clusters, values))
        rows.append(describe(name, block, description, values, clusters, args.n_boot))
        return values

    def set_level(name, block, description, anchors, partners):
        pairs = set_pairs(anchors, partners, args.k, args.partner_seed)
        run_pairs(name, block, description, pairs, [os.path.basename(a) for a, _ in pairs])

    hb, hf, hbp = sets["heldout_base"], sets["heldout_ft"], sets["heldout_base_ftprompt"]
    ids = sorted(set(hf.first) & set(hb.first))
    base_first, ft_first = [hb.first[i] for i in ids], [hf.first[i] for i in ids]
    set_level("real_vs_real", "heldout_set", "real vs real", real_held, real_held)
    set_level("base_vs_base", "heldout_set", "base vs base", base_first, base_first)
    set_level("ft_vs_ft", "heldout_set", "fine-tuned vs fine-tuned", ft_first, ft_first)
    set_level("real_vs_base", "heldout_set", "real vs base", real_held, base_first)
    set_level("real_vs_ft", "heldout_set", "real vs fine-tuned", real_held, ft_first)
    set_level("ft_vs_base", "heldout_set", "fine-tuned vs base", ft_first, base_first)

    def paired(name, description, keyed_pairs):
        """keyed_pairs: list of (mask_id, path_a, path_b). Returns the per-mask mean distance."""
        vals = run_pairs(name, "heldout_paired", description, [(a, b) for _, a, b in keyed_pairs], [i for i, _, _ in keyed_pairs])
        per_mask = defaultdict(list)
        for (i, _, _), v in zip(keyed_pairs, vals):
            per_mask[i].append(v)
        return {i: float(np.mean(v)) for i, v in per_mask.items()}

    two_seed = [i for i in ids if len(hf.seeds[i]) > 1 and len(hb.seeds[i]) > 1]
    ft_effect = paired("ft_effect", "base vs fine-tuned, same mask and seed",
                       [(i, hb.by_key[(i, s)], hf.by_key[(i, s)]) for i in ids for s in hf.seeds[i] if (i, s) in hb.by_key])
    seed_ft = paired("seed_ft", "fine-tuned, seed 1 vs seed 2",
                     [(i, hf.by_key[(i, hf.seeds[i][0])], hf.by_key[(i, hf.seeds[i][1])]) for i in two_seed])
    seed_base = paired("seed_base", "base, seed 1 vs seed 2",
                       [(i, hb.by_key[(i, hb.seeds[i][0])], hb.by_key[(i, hb.seeds[i][1])]) for i in two_seed])
    common_p = [i for i in ids if i in hbp.first]
    ft_effect_fixed = paired("ft_effect_same_prompt", "base vs fine-tuned, same mask, seed and prompt",
                             [(i, hbp.first[i], hf.by_key[(i, hbp.seeds[i][0])]) for i in common_p
                              if (i, hbp.seeds[i][0]) in hf.by_key])
    paired("prompt_effect", "base model, its own prompt vs the leprosy prompt",
           [(i, hb.by_key[(i, hbp.seeds[i][0])], hbp.first[i]) for i in common_p if (i, hbp.seeds[i][0]) in hb.by_key])

    tests = [
        paired_test("ft_effect_vs_seed_ft", "x = base vs fine-tuned, y = fine-tuned seed vs seed", ft_effect, seed_ft, args.n_boot),
        paired_test("ft_effect_vs_seed_base", "x = base vs fine-tuned, y = base seed vs seed", ft_effect, seed_base, args.n_boot),
        paired_test("ft_effect_same_prompt_vs_seed_ft", "x = base vs fine-tuned (same prompt), y = fine-tuned seed vs seed",
                    ft_effect_fixed, seed_ft, args.n_boot),
    ]

    kid = {
        "base": kid_fid(real_held, hb.files, device, args.smoke),
        "fine-tuned": kid_fid(real_held, hf.files, device, args.smoke),
        "base, leprosy prompt": kid_fid(real_held, hbp.files, device, args.smoke),
    }

    kb, kf = sets["kaggle_base"], sets["kaggle_ft"]
    kids = sorted(set(kf.first) & set(kb.first))
    kb_first, kf_first = [kb.first[i] for i in kids], [kf.first[i] for i in kids]
    set_level("rep_real_vs_real", "replication", "real vs real (v1 raw crops)", real_v1, real_v1)
    set_level("rep_ft_vs_ft", "replication", "fine-tuned vs fine-tuned (Kaggle masks)", kf_first, kf_first)
    set_level("rep_ft_vs_base", "replication", "fine-tuned vs base (Kaggle masks)", kf_first, kb_first)
    set_level("rep_real_vs_ft", "replication", "real vs fine-tuned (Kaggle masks)", real_v1, kf_first)

    examples_figure(run, hb, hf, ids, os.path.join(out, "heldout_examples.png"))
    summary = {"provenance": provenance, "lpips": rows, "tests": tests, "kid_fid": kid, "v1_reference": V1,
               "n_boot": args.n_boot, "k": args.k, "smoke": args.smoke}
    save_json(summary, os.path.join(out, "summary.json"))
    write_markdown(os.path.join(out, "summary.md"), summary)
    print(open(os.path.join(out, "summary.md")).read())


if __name__ == "__main__":
    main()
