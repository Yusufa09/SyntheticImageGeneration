import os
from PIL import Image
from tqdm import tqdm

DATASETS = {
    "AZH": [
        "data/AZH",
    ],
    "Kaggle": [
        "/data/kaggle/train_images",
        "/data/kaggle/test_images"
    ],
    "WoundDB": [
        "/data/woundDB"
    ]
}
OUTPUT_DIR = "/Downloads/final_dataset"
TARGET_SIZE = (512, 512)

def combine_datasets(datasets, output_dir, target_size=(512, 512)):
    os.makedirs(output_dir, exist_ok=True)
    total_counter = 0

    for tag, paths in datasets.items():
        tag_dir = os.path.join(output_dir, tag)
        os.makedirs(tag_dir, exist_ok=True)
        counter = 0

        for dataset_path in paths:
            if not os.path.exists(dataset_path):
                print(f"⚠️ Skipping missing folder: {dataset_path}")
                continue

            for file_name in tqdm(os.listdir(dataset_path), desc=f"Processing {tag}"):
                if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                try:
                    img_path = os.path.join(dataset_path, file_name)
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(target_size, Image.LANCZOS)

                    # name format: TAG_00001.png
                    new_name = f"{tag}_{counter:05d}.png"
                    counter += 1
                    total_counter += 1

                    img.save(os.path.join(tag_dir, new_name))
                except Exception as e:
                    print(f" Error processing {file_name}: {e}")

        print(f" {tag}: {counter} images processed and saved in {tag_dir}")

    print(f"\n Combined dataset ready — total images: {total_counter}")
    print(f" Output directory: {os.path.abspath(output_dir)}")

# --- Run ---
if __name__ == "__main__":
    combine_datasets(DATASETS, OUTPUT_DIR, TARGET_SIZE)