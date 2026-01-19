import os
from pathlib import Path
from PIL import Image

LEP_IMG_DIR = Path("/leprosy dataset/imgs")   # .jpg images
LEP_MSK_DIR = Path("/leprosy_masks")                 # .png masks

OUT_IMG_DIR = Path("resized_leprosy/images")
OUT_MSK_DIR = Path("resized_leprosy/masks")

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_MSK_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (512, 512)

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]

mask_lookup = {p.stem: p for p in LEP_MSK_DIR.iterdir() if p.is_file() and is_image_file(p)}

done = 0
missing = 0

for img_path in sorted([p for p in LEP_IMG_DIR.iterdir() if p.is_file() and is_image_file(p)]):
    msk_path = mask_lookup.get(img_path.stem)
    if msk_path is None:
        missing += 1
        continue

    img = Image.open(img_path).convert("RGB")
    msk = Image.open(msk_path).convert("L")

    img_r = img.resize(TARGET_SIZE, Image.BICUBIC)
    msk_r = msk.resize(TARGET_SIZE, Image.NEAREST)

    # save with matching stems
    img_r.save(OUT_IMG_DIR / f"{img_path.stem}.png")
    msk_r.save(OUT_MSK_DIR / f"{img_path.stem}.png")

    done += 1
def collect_pairs(img_dir: Path, msk_dir: Path):
    imgs = sorted([p for p in img_dir.iterdir() if p.is_file() and is_image_file(p)])
    image_paths, mask_paths = [], []

    for p in imgs:
        m = msk_dir / p.name
        if m.exists():
            image_paths.append(str(p))
            mask_paths.append(str(m))

    return image_paths, mask_paths

lep_image_paths, lep_mask_paths = collect_pairs(OUT_IMG_DIR, OUT_MSK_DIR)

print("Leprosy pairs:", len(lep_image_paths))
print("Example image:", lep_image_paths[0] if lep_image_paths else None)
print("Example mask :", lep_mask_paths[0] if lep_mask_paths else None)
