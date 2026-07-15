import os
import re
import torch
import numpy as np
from PIL import Image
import open_clip

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MarqoFashionSigLIPEmbedder:
    """Wrapper for Marqo-FashionSigLIP (768-dim) using open_clip to avoid transformers meta tensor bugs"""
    def __init__(self, model_id="hf-hub:Marqo/marqo-fashionSigLIP"):
        print(f"[*] Initializing Marqo-FashionSigLIP using open_clip on {DEVICE}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_id)
        self.model = self.model.to(DEVICE)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_id)

    def embed_images(self, pil_images: list, batch_size: int = 16) -> np.ndarray:
        embeddings_list = []
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i : i + batch_size]
            # Preprocess batch images using OpenCLIP transform
            processed_images = [self.preprocess(img) for img in batch]
            image_tensor = torch.stack(processed_images).to(DEVICE)
            
            with torch.no_grad():
                # Encode and normalize visual features
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings_list.append(image_features.cpu().numpy())
                
        return np.concatenate(embeddings_list, axis=0)

    def embed_text(self, text_queries: list) -> np.ndarray:
        embeddings = []
        for query in text_queries:
            words = query.split()
            # If query is long (paragraph), chunk into sentences and average pooling
            if len(words) > 50:
                sentences = [s.strip() for s in re.split(r'[.!?\n]', query) if len(s.strip()) > 3]
                if not sentences:
                    sentences = [query]
                
                sentence_embs = []
                for sent in sentences:
                    text_tokens = self.tokenizer([sent]).to(DEVICE)
                    with torch.no_grad():
                        feat = self.model.encode_text(text_tokens)
                        feat = feat / feat.norm(dim=-1, keepdim=True)
                        sentence_embs.append(feat.cpu().numpy()[0])
                
                avg_emb = np.mean(sentence_embs, axis=0)
                avg_emb = avg_emb / np.linalg.norm(avg_emb)
                embeddings.append(avg_emb)
            else:
                # Direct embedding for short query text
                text_tokens = self.tokenizer([query]).to(DEVICE)
                with torch.no_grad():
                    feat = self.model.encode_text(text_tokens)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    embeddings.append(feat.cpu().numpy()[0])
                    
        return np.array(embeddings)
