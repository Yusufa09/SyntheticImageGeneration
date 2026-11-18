"""Data loading for the rerun, copied from model_v5_saved/v5-7.ipynb.

The ROI crop, mask features and context hint match ROIWoundDatasetV3 and
generate_v3_image exactly, including their rounding behaviour.
"""
import glob
import os

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

# (image folder, mask folder) for each wound corpus, relative to final_dataset/
WOUND_SOURCES = [
    ("AZH", "AZH_masks"),
    ("Kaggle_images", "Kaggle_Masks"),
    ("WoundDB", "WoundDB_masks"),
]
KAGGLE_MASKS = "Kaggle_Masks"
LIST_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def list_images(folder):
    """Sorted image paths in a folder (the v1 evaluation's list_images)."""
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(LIST_EXTS))


def collect_pairs(image_dirs, mask_dirs, keep_stems=None):
    """Pairs images with masks by filename stem (v1 collect_pairs)."""
    img_map = {}
    for d in image_dirs:
        for ext in ("jpg", "jpeg", "png"):
            for p in glob.glob(os.path.join(d, f"*.{ext}")):
                img_map[stem(p)] = p
    pairs = []
    for d in mask_dirs:
        for ext in ("png", "jpg", "jpeg"):
            for m in glob.glob(os.path.join(d, f"*.{ext}")):
                s = stem(m)
                if s in img_map and (keep_stems is None or s in keep_stems):
                    pairs.append((img_map[s], m))
    pairs.sort()
    return [p[0] for p in pairs], [p[1] for p in pairs]


def wound_pairs(data_root):
    return collect_pairs(
        [os.path.join(data_root, i) for i, _ in WOUND_SOURCES],
        [os.path.join(data_root, m) for _, m in WOUND_SOURCES],
    )


def leprosy_dirs(data_root):
    base = os.path.join(data_root, "resized_leprosy")
    return {
        "raw": os.path.join(base, "images"),
        "cleaned": os.path.join(base, "cleaned_images"),
        "masks": os.path.join(base, "masks"),
    }


def load_rgb01(path):
    return np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def load_mask01(path):
    return np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0


def mask_bbox(mask01):
    ys, xs = np.where(mask01 > 0.5)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def expand_bbox(x0, y0, x1, y1, w, h, margin_frac):
    mx = int((x1 - x0) * margin_frac)
    my = int((y1 - y0) * margin_frac)
    return max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)


def roi_box(mask01, cfg, check_size=True):
    """Expanded lesion box, or None when the mask is empty (or too small, if checked).

    Training checks the size; v1 generation and real-crop code only checked emptiness.
    """
    bb = mask_bbox(mask01)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    if check_size and ((x1 - x0) < cfg.min_roi_pixels or (y1 - y0) < cfg.min_roi_pixels):
        return None
    h, w = mask01.shape[:2]
    return expand_bbox(x0, y0, x1, y1, w, h, cfg.roi_margin_frac)


def crop_image(img01, box, size):
    x0, y0, x1, y1 = box
    crop = (img01[y0:y1, x0:x1, :] * 255).astype(np.uint8)
    return Image.fromarray(crop).resize((size, size), Image.BICUBIC)


def crop_mask(mask01, box, size):
    x0, y0, x1, y1 = box
    crop = (mask01[y0:y1, x0:x1] * 255).astype(np.uint8)
    resized = Image.fromarray(crop).resize((size, size), Image.NEAREST)
    return np.array(resized).astype(np.float32) / 255.0


def mask_features(mask01):
    """[3, H, W]: binary mask plus inward and outward distance maps, each scaled to [0, 1]."""
    bin_mask = (mask01 > 0.5).astype(np.float32)
    inside = bin_mask.astype(bool)
    dist_in = distance_transform_edt(inside).astype(np.float32)
    dist_out = distance_transform_edt(~inside).astype(np.float32)
    eps = 1e-6
    return np.stack([bin_mask, dist_in / (dist_in.max() + eps), dist_out / (dist_out.max() + eps)], axis=0)


class ROIDataset(Dataset):
    """Lesion-centred crops with mask features and a blurred context hint (ROIWoundDatasetV3)."""

    def __init__(self, image_paths, mask_paths, cfg):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.cfg = cfg

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        cfg = self.cfg
        img01 = load_rgb01(self.image_paths[idx])
        msk01 = load_mask01(self.mask_paths[idx])
        box = roi_box(msk01, cfg)
        if box is None:
            return {"skip": True}

        img_pil = crop_image(img01, box, cfg.image_size)
        msk = crop_mask(msk01, box, cfg.image_size)
        if cfg.use_context_hint and cfg.context_blur_radius > 0:
            ctx_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=cfg.context_blur_radius))
        else:
            ctx_pil = img_pil

        img = np.array(img_pil).astype(np.float32) / 255.0 * 2.0 - 1.0
        ctx = np.array(ctx_pil).astype(np.float32) / 255.0 * 2.0 - 1.0
        return {
            "pixel_values": torch.from_numpy(img).permute(2, 0, 1).contiguous(),
            "ctx_pixels": torch.from_numpy(ctx).permute(2, 0, 1).contiguous(),
            "mask_feats": torch.from_numpy(mask_features(msk)).contiguous(),
            "skip": False,
        }


def collate_skip(batch):
    """Drops skipped samples; a batch can therefore be smaller than batch_size, as in v1."""
    batch = [b for b in batch if not b.get("skip", False)]
    if not batch:
        return None
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in ("pixel_values", "ctx_pixels", "mask_feats")}


def generation_mask(mask_path, cfg):
    """ROI-cropped mask at cfg.image_size, built the way generate_v3_image built it."""
    msk01 = load_mask01(mask_path)
    box = roi_box(msk01, cfg, check_size=False)
    if box is None:
        raise ValueError(f"Mask has no lesion pixels: {mask_path}")
    return crop_mask(msk01, box, cfg.image_size)


def real_roi(image_path, mask_path, cfg):
    """ROI crop of a real image, built the way the v1 evaluation built it."""
    msk01 = load_mask01(mask_path)
    box = roi_box(msk01, cfg, check_size=False)
    if box is None:
        return None
    return crop_image(load_rgb01(image_path), box, cfg.image_size)
