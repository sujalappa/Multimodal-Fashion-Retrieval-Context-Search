import os
import json
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

NUM_IMAGES = 700
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
IMAGES_DIR = os.path.join(DATA_DIR, "images")
HF_METADATA_PATH = os.path.join(DATA_DIR, "hf_metadata.json")

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)

def download_dataset():
    print(f"[*] Starting download of {NUM_IMAGES} images from Fashionpedia HF dataset...")
    
    try:
        # Stream the dataset to avoid downloading the entire 3.5GB file first
        dataset = load_dataset("detection-datasets/fashionpedia", split="train", streaming=True)
    except Exception as e:
        print(f"[!] Error loading dataset from HF: {e}")
        return

    hf_metadata = {}
    existing_images = set(os.listdir(IMAGES_DIR))
    
    count = 0
    pbar = tqdm(total=NUM_IMAGES, desc="Downloading images")
    
    for item in dataset:
        if count >= NUM_IMAGES:
            break
            
        img_id = item.get("image_id", count)
        filename = f"img_{img_id:06d}.jpg"
        image_path = os.path.join(IMAGES_DIR, filename)
        
        # Save the image if not already downloaded
        if filename not in existing_images:
            try:
                img_data = item["image"]
                img_data.save(image_path)
            except Exception as e:
                print(f"\n[!] Error saving image {filename}: {e}")
                continue
        
        # Save HF annotations (bounding boxes and categories)
        hf_metadata[filename] = {
            "image_id": img_id,
            "width": item.get("width"),
            "height": item.get("height"),
            "objects": item.get("objects", {})
        }
        
        count += 1
        pbar.update(1)
        
    pbar.close()
    
    # Save the Hugging Face annotations to a local JSON file
    with open(HF_METADATA_PATH, "w") as f:
        json.dump(hf_metadata, f, indent=2)
        
    print(f"\n[*] Success! 700 images downloaded to {IMAGES_DIR}")
    print(f"[*] Hugging Face metadata saved to {HF_METADATA_PATH}")

if __name__ == "__main__":
    download_dataset()
