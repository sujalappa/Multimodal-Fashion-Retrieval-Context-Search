import os
import json
import time
import threading
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Constants
MODEL_NAME = "gemini-3.1-flash-lite"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
IMAGES_DIR = os.path.join(DATA_DIR, "images")
HF_METADATA_PATH = os.path.join(DATA_DIR, "hf_metadata.json")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

# Concurrency Settings
# Note: Free tier Gemini API limit is 15 requests per minute.
# Paid tier allows much higher concurrency. Default to 15 workers for paid tier.
MAX_WORKERS = 15 

# Lock for thread-safe file operations
file_lock = threading.Lock()

# Standard Fashionpedia main garment categories (indices 0 to 26)
CATEGORY_NAMES = [
    "shirt", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest",
    "pants", "shorts", "skirt", "coat", "dress", "jumpsuit", "cape", "glasses",
    "hat", "headband, head covering, hair accessory", "tie", "glove", "watch",
    "belt", "tights, stockings", "sock", "shoe", "bag, wallet", "scarf", "hood", "bow"
]

def initialize_gemini():
    if not HAS_GEMINI:
        print("[!] google-generativeai is not installed. Please run 'pip install google-generativeai'.")
        return None
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[!] GEMINI_API_KEY environment variable is not set.")
        print("    Set it using: set GEMINI_API_KEY=your_key (Windows CMD) or $env:GEMINI_API_KEY='your_key' (PowerShell)")
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)

def get_garments_for_image(item):
    """Extracts unique main garment names (categories 0 to 26) from HF annotations."""
    objects = item.get("objects", {})
    categories = objects.get("category", [])
    
    garments = set()
    for cat_id in categories:
        if 0 <= cat_id < len(CATEGORY_NAMES):
            garments.add(CATEGORY_NAMES[cat_id])
            
    return list(garments)

def label_image(model, image_path, garments):
    """Calls Gemini to label the image background environment and garment colors."""
    if not garments:
        raise ValueError(f"No main garments found in Hugging Face annotations for {os.path.basename(image_path)}")
        
    garment_str = ", ".join([f"'{g}'" for g in garments])
    prompt = f"""
    This image contains the following clothing items: [{garment_str}].

    Identify the dominant color of each of these items, and describe the background environment.
    Return the response in this exact JSON structure:
    {{
      "environment": "Describe the scene background/setting (e.g. office, street, park, beach, home, indoor studio)",
      "colors": {{
        {", ".join([f'"{g}": "color"' for g in garments])}
      }}
    }}

    Return raw JSON only. Do not wrap in markdown blocks.
    """

    try:
        img = Image.open(image_path)
        response = model.generate_content(
            [img, prompt],
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text.strip())
    except Exception as e:
        # If rate limited (HTTP 429), sleep and raise exception for executor to retry
        if "429" in str(e) or "ResourceExhausted" in str(e):
            time.sleep(5.0)
        return None

def process_single_image(model, filename, item, metadata):
    """Worker function to process a single image with retry logic."""
    image_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(image_path):
        return filename, None
        
    garments = get_garments_for_image(item)
    if not garments:
        print(f"\n[!] Skipping {filename} (no main garments detected in HF metadata).")
        return filename, None
    
    max_retries = 3
    labels = None
    
    for attempt in range(max_retries):
        labels = label_image(model, image_path, garments)
        if labels:
            break
        # Back off before retrying
        time.sleep(2.0 * (attempt + 1))
        
    if labels:
        # Extract objects and category IDs from the HF item
        objects = item.get("objects", {})
        cat_ids = objects.get("category", [])
        
        # Normalize Gemini color keys (lowercase and stripped)
        gemini_colors = {str(k).lower().strip(): str(v).lower().strip() for k, v in labels.get("colors", {}).items()}
        
        box_colors = []
        for cat_id in cat_ids:
            if 0 <= cat_id < len(CATEGORY_NAMES):
                cat_name = CATEGORY_NAMES[cat_id]
                normalized_name = cat_name.lower().strip()
                # Look up normalized name, default to 'unknown' if not found
                color = gemini_colors.get(normalized_name, "unknown")
                box_colors.append(color)
            else:
                box_colors.append("unknown")
        
        objects["colors"] = box_colors

        result = {
            "image_id": item["image_id"],
            "width": item.get("width"),
            "height": item.get("height"),
            "objects": objects,
            "environment": labels.get("environment", "unknown")
        }
        return filename, result
    else:
        return filename, None

def main():
    model = initialize_gemini()
    if not model:
        return

    if not os.path.exists(HF_METADATA_PATH):
        print(f"[!] HF metadata file not found at {HF_METADATA_PATH}. Please run download_dataset.py first.")
        return

    with open(HF_METADATA_PATH, "r") as f:
        hf_metadata = json.load(f)

    metadata = {}
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                metadata = json.load(f)
            print(f"[*] Loaded existing labels for {len(metadata)} images. Resuming...")
        except Exception:
            print("[*] No existing metadata file or it was corrupted. Starting fresh.")

    # Filter images to process
    images_to_process = {k: v for k, v in hf_metadata.items() if k not in metadata}
    
    if not images_to_process:
        print("[*] All images are already labeled!")
        return

    print(f"[*] Starting parallel labeling for {len(images_to_process)} images using {MODEL_NAME} ({MAX_WORKERS} threads)...")
    
    pbar = tqdm(total=len(images_to_process), desc="Labeling progress")
    
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_single_image, model, filename, item, metadata): filename
                for filename, item in images_to_process.items()
            }
            
            for future in as_completed(futures):
                filename, result = future.result()
                if result:
                    # Write result thread-safely
                    with file_lock:
                        metadata[filename] = result
                        with open(METADATA_PATH, "w") as f:
                            json.dump(metadata, f, indent=2)
                pbar.update(1)
                
    except KeyboardInterrupt:
        print("\n[!] Process interrupted by user. Exiting gracefully. Already labeled progress is saved.")
    finally:
        pbar.close()
        
    print(f"[*] Labeling cycle complete. Metadata saved to {METADATA_PATH}.")

if __name__ == "__main__":
    main()
