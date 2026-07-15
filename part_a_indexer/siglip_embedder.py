import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SigLIPEmbedder:
    """Wrapper for Google SigLIP-2 Base (google/siglip2-base-patch16-256)"""
    def __init__(self, model_name="google/siglip2-base-patch16-256"):
        print(f"[*] Loading SigLIP-2 Base model: {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    def embed_images(self, pil_images: list, batch_size=16) -> np.ndarray:
        embeddings_list = []
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i+batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # L2 Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings_list.append(image_features.cpu().numpy())
        return np.concatenate(embeddings_list, axis=0)

    def embed_text(self, text_queries: list) -> np.ndarray:
        inputs = self.processor(text=text_queries, padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # L2 Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()
