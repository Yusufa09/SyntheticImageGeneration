"""End-to-end check of the rerun pipeline on a laptop, with tiny random models.

    python smoke_test.py          # about a few minutes on a CPU; downloads nothing

It builds a small fake dataset with the same folder layout as final_dataset,
interrupts a training run and checks that it resumes, runs run_all.sh with
--smoke, checks the outputs, and runs run_all.sh a second time to confirm that
finished steps are skipped.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))


def fake_pair(rng, size=64, empty=False):
    img = np.clip(rng.uniform(80, 200, 3) + rng.normal(0, 20, (size, size, 3)), 0, 255)
    mask = np.zeros((size, size), np.uint8)
    if not empty:
        yy, xx = np.mgrid[:size, :size]
        cy, cx = rng.integers(20, 44, 2)
        ry, rx = rng.integers(8, 16, 2)
        inside = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
        mask[inside] = 255
        img[inside] = np.clip(img[inside] * [1.2, 0.6, 0.6], 0, 255)
    return Image.fromarray(img.astype(np.uint8)), Image.fromarray(mask)


def make_dataset(root, rng):
    layout = [("AZH", "AZH_masks", "AZH_{:05d}", 6),
              ("Kaggle_images", "Kaggle_Masks", "wsnet_{:04d}", 10),
              ("WoundDB", "WoundDB_masks", "WoundDB_{:05d}", 4)]
    for img_dir, mask_dir, pattern, n in layout:
        os.makedirs(os.path.join(root, img_dir))
        os.makedirs(os.path.join(root, mask_dir))
        for i in range(n):
            img, msk = fake_pair(rng, empty=(i == 0))  # one empty mask per corpus, as in the real data
            img.save(os.path.join(root, img_dir, pattern.format(i) + ".png"))
            msk.save(os.path.join(root, mask_dir, pattern.format(i) + ".png"))
    lep = os.path.join(root, "resized_leprosy")
    for sub in ("images", "cleaned_images", "masks"):
        os.makedirs(os.path.join(lep, sub))
    for i in range(20):
        name = f"IMG{1000 + i}.png"
        img, msk = fake_pair(rng, empty=(i == 3))
        img.save(os.path.join(lep, "images", name))
        cleaned = img.copy()
        if i % 3 == 0:  # stand-in for a removed colour card
            ImageDraw.Draw(cleaned).rectangle((2, 2, 18, 12), fill=(120, 110, 100))
        cleaned.save(os.path.join(lep, "cleaned_images", name))
        msk.save(os.path.join(lep, "masks", name))


def run(cmd, env=None, expect=0):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    code = subprocess.run([str(c) for c in cmd], cwd=HERE, env=env).returncode
    if code != expect:
        raise SystemExit(f"FAILED: exit code {code}, expected {expect}: {' '.join(str(c) for c in cmd)}")


def load(path):
    with open(path) as f:
        return json.load(f)


def check(condition, message):
    if not condition:
        raise SystemExit(f"CHECK FAILED: {message}")
    print("ok:", message)


def main():
    keep = "--keep" in sys.argv
    tmp = tempfile.mkdtemp(prefix="rerun_smoke_")
    data, run_dir, resume_dir = (os.path.join(tmp, n) for n in ("data", "run", "resume"))
    make_dataset(data, np.random.default_rng(0))

    # 1. A run stopped after its first checkpoint resumes and finishes with a continuous history.
    train = [sys.executable, "train.py", "--stage", "base", "--data_root", data, "--out_dir", resume_dir,
             "--smoke", "--max_steps", "6", "--ckpt_every", "2"]
    run(train + ["--stop_at", "2"], expect=3)
    check(os.path.exists(os.path.join(resume_dir, "last", "trainer_state.pt")), "interrupted run left a checkpoint")
    run(train)
    meta = load(os.path.join(resume_dir, "final", "train_config.json"))
    hist = load(os.path.join(resume_dir, "final", "history.json"))
    check(meta["resumed"] and hist["step"] == list(range(1, 7)), "resumed run finished with steps 1-6 recorded once")
    check(not os.path.exists(os.path.join(resume_dir, "last")), "resume checkpoint removed after finishing")

    # 2. The full pipeline.
    env = dict(os.environ, DATA_ROOT=data, RUN_DIR=run_dir, EXTRA_ARGS="--smoke", PY=sys.executable, SHUTDOWN_WHEN_DONE="0")
    run(["bash", "run_all.sh"], env=env)

    split = load(os.path.join(run_dir, "split.json"))
    check(not set(split["train"]) & set(split["heldout"]), "train and held-out images do not overlap")
    check(len(split["train"]) + len(split["heldout"]) == 20, "split covers every leprosy image")

    base = load(os.path.join(run_dir, "checkpoints", "base", "final", "train_config.json"))
    ft = load(os.path.join(run_dir, "checkpoints", "finetune", "final", "train_config.json"))
    check(base["weights_sha256"] != ft["weights_sha256"], "base and fine-tuned weights differ")
    check(ft["init_unet_sha256"] == base["weights_sha256"], "fine-tuning started from the saved base weights")
    check(ft["n_pairs"] == len(split["train"]), "fine-tuning used only the training split")
    check(base["n_pairs"] == 20, "base training used all wound pairs")

    for name in ("heldout_base", "heldout_ft", "heldout_base_ftprompt", "kaggle_base", "kaggle_ft"):
        rec = load(os.path.join(run_dir, "gen", name, "manifest.json"))
        files = [os.path.join(run_dir, "gen", name, it["file"]) for it in rec["items"]]
        check(files and all(os.path.exists(f) for f in files), f"{name}: all {len(files)} images exist")

    hb = load(os.path.join(run_dir, "gen", "heldout_base", "manifest.json"))["items"][0]["file"]
    a = np.asarray(Image.open(os.path.join(run_dir, "gen", "heldout_base", hb)), float)
    b = np.asarray(Image.open(os.path.join(run_dir, "gen", "heldout_ft", hb)), float)
    check(np.abs(a - b).mean() > 0, "base and fine-tuned images for the same mask and seed differ")

    summary = load(os.path.join(run_dir, "metrics", "summary.json"))
    names = {r["name"] for r in summary["lpips"]}
    expected = {"real_vs_real", "ft_vs_ft", "ft_effect", "seed_ft", "rep_ft_vs_base", "rep_real_vs_ft"}
    check(expected <= names, "summary has the main comparisons")
    check(len(summary["kid_fid"]) == 3 and len(summary["tests"]) == 3, "summary has KID/FID and the paired tests")
    check(os.path.exists(os.path.join(run_dir, "metrics", "heldout_examples.png")), "example figure written")
    check(os.path.exists(os.path.join(run_dir, "results.tar.gz")), "results bundle written")

    # 3. Running again skips every finished step.
    run(["bash", "run_all.sh"], env=env)
    check(load(os.path.join(run_dir, "checkpoints", "base", "final", "train_config.json"))["weights_sha256"]
          == base["weights_sha256"], "second run left the base checkpoint unchanged")

    print("\nSMOKE TEST PASSED")
    if keep:
        print("Outputs kept in", tmp)
    else:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
