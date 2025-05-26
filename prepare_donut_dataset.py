import os
import json
import shutil
from sklearn.model_selection import train_test_split

# Paths
image_dir = "./CM1-Dataset"
json_path = "./data/donut_ready_personen.json"
output_dir = "./donut_dataset"

# Create output folders
for split in ["train", "val"]:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)

# Load the data
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Split the data
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Save metadata and copy images
for split, dataset in [("train", train_data), ("val", val_data)]:
    with open(os.path.join(output_dir, split, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    for item in dataset:
        src = os.path.join(image_dir, item["image"])
        dst = os.path.join(output_dir, split, "images", item["image"])
        if os.path.exists(src):
            shutil.copy2(src, dst)

print("✅ Dataset split and copied.")
