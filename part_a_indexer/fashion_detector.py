# ==============================================================================
# FALLBACK NOTICE:
# This script (fashion_detector.py) is NOT used in our main indexing pipeline.
# It is only included as a fallback reference for when datasets are completely 
# raw and do not contain pre-defined crop/bounding box coordinates.
#
# LIMITATIONS:
# 1. This model performs very poorly on complex runway or street outfits.
# 2. It struggles to detect overlapping garments (e.g. ties under jackets).
# 3. It is slow on CPU and has low precision, which degrades indexing quality.
# Use ground-truth bounding box coordinates from metadata.json instead.
# ==============================================================================

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FashionDetector:
    """Wrapper for the fine-tuned Conditional-DETR Fashion Object Detector"""
    def __init__(self, model_name="yainage90/fashion-object-detection"):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        self.id2label = self.model.config.id2label
        print(f"[*] FashionDetector loaded with categories: {list(self.id2label.values())}")

    def detect_objects(self, pil_image: Image.Image, threshold=0.5) -> list:
        width, height = pil_image.size
        inputs = self.processor(images=pil_image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        target_sizes = torch.tensor([[height, width]], device=DEVICE)
        results = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        
        detections = []
        for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
            box_coords = [float(coord) for coord in box.cpu().numpy()]
            label_name = self.id2label[int(label_id.cpu().numpy())]
            
            detections.append({
                "category": label_name.lower().strip(),
                "bbox": box_coords,
                "score": float(score.cpu().numpy())
            })
            
        return detections

    def crop_detections(self, pil_image: Image.Image, detections: list) -> list:
        cropped_objects = []
        for det in detections:
            xmin, ymin, xmax, ymax = det["bbox"]
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(pil_image.width, int(xmax))
            ymax = min(pil_image.height, int(ymax))
            
            w = xmax - xmin
            h = ymax - ymin
            if w < 10 or h < 10:
                continue
                
            crop_img = pil_image.crop((xmin, ymin, xmax, ymax))
            
            # Aspect-ratio-preserving padding to make crop a square
            max_dim = max(w, h)
            # Create a neutral grey background square image
            padded_img = Image.new("RGB", (max_dim, max_dim), (128, 128, 128))
            # Paste crop centered inside the square
            paste_x = (max_dim - w) // 2
            paste_y = (max_dim - h) // 2
            padded_img.paste(crop_img, (paste_x, paste_y))
            
            cropped_objects.append({
                "category": det["category"],
                "bbox": [xmin, ymin, xmax, ymax],
                "crop": padded_img
            })
            
        return cropped_objects
