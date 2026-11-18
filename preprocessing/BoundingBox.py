import numpy as np
from PIL import Image
import os


def mask_bbox(mask_np):
    ys, xs = np.where(mask_np > 0.5)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return x0, y0, x1, y1


def expand_bbox(x0, y0, x1, y1, w, h, margin_frac):
    bw = x1 - x0
    bh = y1 - y0
    mx = int(bw * margin_frac)
    my = int(bh * margin_frac)

    nx0 = max(0, x0 - mx)
    ny0 = max(0, y0 - my)
    nx1 = min(w, x1 + mx)
    ny1 = min(h, y1 + my)

    return nx0, ny0, nx1, ny1


def roi_crop(image_path, mask_path, save_dir="roi_output", image_size=512):
    
    os.makedirs(save_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    msk = Image.open(mask_path).convert("L")

    img_np = np.array(img).astype(np.float32) / 255.0
    msk_np = np.array(msk).astype(np.float32) / 255.0

    h, w = msk_np.shape

    bbox = mask_bbox(msk_np)
    if bbox is None:
        raise ValueError("Mask has no positive pixels")

    x0, y0, x1, y1 = bbox
    x0, y0, x1, y1 = expand_bbox(x0, y0, x1, y1, w, h, 0.30)

    img_crop = img_np[y0:y1, x0:x1]
    msk_crop = msk_np[y0:y1, x0:x1]

    img_crop = Image.fromarray((img_crop * 255).astype(np.uint8)).resize(
        (image_size, image_size), Image.BICUBIC
    )

    msk_crop = Image.fromarray((msk_crop * 255).astype(np.uint8)).resize(
        (image_size, image_size), Image.NEAREST
    )

    name = os.path.splitext(os.path.basename(image_path))[0]

    img_save = os.path.join(save_dir, f"{name}_roi.png")
    mask_save = os.path.join(save_dir, f"{name}_roi_mask.png")

    img_crop.save(img_save)
    msk_crop.save(mask_save)

    print("Saved image:", img_save)
    print("Saved mask :", mask_save)


# Example
image_path = "/Users/yusufabdulkadir/Leprosy AI Project/final_dataset/resized_leprosy/images/IMG990.png"
mask_path = "/Users/yusufabdulkadir/Leprosy AI Project/final_dataset/resized_leprosy/masks/IMG990.png"

roi_crop(image_path, mask_path)