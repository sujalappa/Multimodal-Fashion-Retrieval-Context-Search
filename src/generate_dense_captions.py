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
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

# Concurrency settings for Gemini API rate limits (adjust based on tier)
MAX_WORKERS = 10

# Lock for thread-safe metadata updates
file_lock = threading.Lock()

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

def generate_caption_for_image(model, filename):
    img_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(img_path):
        return filename, None
        
    prompt = (
        "Describe this fashion photo in extreme detail. "
        "Include the person's gender, pose or action (e.g. sitting, standing, walking, leaning), "
        "all clothing items (detailing their cut, color, pattern, and fabric material), "
        "shoes, bags, hats, accessories, and the environment background or setting. "
        "Write a single, highly descriptive paragraph containing all these details."
    )
    
    try:
        img = Image.open(img_path)
        response = model.generate_content([prompt, img])
        caption = response.text.strip()
        return filename, caption
    except Exception as e:
        print(f"\n[!] Error generating caption for {filename}: {e}")
        return filename, None

def main():
    model = initialize_gemini()
    if not model:
        return

    if not os.path.exists(METADATA_PATH):
        print(f"[!] Metadata file not found at {METADATA_PATH}. Please run src/label_dataset.py first.")
        return

    # Load existing metadata
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    # Find images that need captioning (missing "dense_caption")
    images_to_caption = []
    for filename in os.listdir(IMAGES_DIR):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            if filename in metadata:
                # Check if "dense_caption" already exists
                if "dense_caption" not in metadata[filename] or not metadata[filename]["dense_caption"]:
                    images_to_caption.append(filename)

    if not images_to_caption:
        print("[*] All images already have dense captions!")
        return

    print(f"[*] Starting dense captioning for {len(images_to_caption)} images using {MODEL_NAME}...")
    pbar = tqdm(total=len(images_to_caption), desc="Captioning progress")
    
    # Thread pool for parallel API queries
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(generate_caption_for_image, model, fname): fname
            for fname in images_to_caption
        }
        
        for future in as_completed(futures):
            filename, caption = future.result()
            if caption:
                with file_lock:
                    metadata[filename]["dense_caption"] = caption
                    # Save progress incrementally to avoid data loss
                    with open(METADATA_PATH, "w") as f:
                        json.dump(metadata, f, indent=2)
            pbar.update(1)
            
    pbar.close()
    print("[*] Dense caption generation complete!")

if __name__ == "__main__":
    main()
