import os
import sys
from PIL import Image

# Resolve sibling imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from part_a_indexer.fashion_detector import FashionDetector

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CROPS_DIR = os.path.join(DATA_DIR, "test_crops")

# Ensure test crops directory exists
os.makedirs(CROPS_DIR, exist_ok=True)

def test_detector():
    print("[*] Initializing Fashion Detector (DETR)...")
    detector = FashionDetector()
    
    if not os.path.exists(IMAGES_DIR):
        print(f"[!] Images directory not found at {IMAGES_DIR}. Run download_dataset.py first.")
        return
        
    # Get first 5 images in the directory
    image_filenames = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])[:5]
    
    if not image_filenames:
        print("[!] No images found in data/images/")
        return
        
    print(f"[*] Testing detector on {len(image_filenames)} sample images...")
    
    for filename in image_filenames:
        img_path = os.path.join(IMAGES_DIR, filename)
        print(f"\n" + "-"*50)
        print(f"[*] Image: {filename}")
        print("-"*50)
        
        try:
            img = Image.open(img_path).convert("RGB")
            print(f"  Dimensions: {img.width}x{img.height}")
            
            # Detect objects with a threshold of 0.50
            detections = detector.detect_objects(img, threshold=0.50)
            print(f"  Detected {len(detections)} fashion items:")
            
            for idx, det in enumerate(detections):
                print(f"    [{idx+1}] Category: {det['category']:<8} | Score: {det['score']:.4f} | Box: {det['bbox']}")
            
            # Crop detections
            crops = detector.crop_detections(img, detections)
            print(f"  Successfully cropped {len(crops)} items.")
            
            # Save crops to disk for visual verification
            for idx, crop in enumerate(crops):
                crop_filename = f"crop_{filename.split('.')[0]}_{crop['category']}_{idx}.jpg"
                crop_path = os.path.join(CROPS_DIR, crop_filename)
                crop["crop"].save(crop_path)
                print(f"    -> Saved crop: data/test_crops/{crop_filename}")
                
        except Exception as e:
            print(f"  [!] Error processing {filename}: {e}")
            
    print("\n[*] Detection test complete! Check your crops in: data/test_crops/")

if __name__ == "__main__":
    test_detector()
