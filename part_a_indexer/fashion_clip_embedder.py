import re
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FashionCLIPEmbedder:
    """Wrapper for Fashion-CLIP (CLIP fine-tuned on fashion datasets)"""
    def __init__(self, model_name="patrickjohncyh/fashion-clip"):
        print(f"[*] Loading Fashion-CLIP model: {model_name}...")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    def embed_images(self, pil_images: list, batch_size=32) -> np.ndarray:
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
        embeddings = []
        for query in text_queries:
            # Check if query is a long paragraph
            words = query.split()
            # A rough heuristic: 1 word ~ 1.3 tokens. 50 words is safe under 77 tokens.
            if len(words) > 50:
                # Split paragraph into sentences
                sentences = [s.strip() for s in re.split(r'[.!?\n]', query) if len(s.strip()) > 3]
                if not sentences:
                    sentences = [query]
                
                # Embed each sentence separately (each is under the 77 token limit)
                sentence_embs = []
                for sent in sentences:
                    # Force truncation at 77 tokens for safety
                    inputs = self.processor(text=[sent], padding=True, truncation=True, max_length=77, return_tensors="pt").to(DEVICE)
                    with torch.no_grad():
                        feat = self.model.get_text_features(**inputs)
                        feat = feat / feat.norm(dim=-1, keepdim=True)
                        sentence_embs.append(feat.cpu().numpy()[0])
                
                # Average sentence embeddings to get the unified paragraph embedding
                avg_emb = np.mean(sentence_embs, axis=0)
                avg_emb = avg_emb / np.linalg.norm(avg_emb)  # Re-normalize
                embeddings.append(avg_emb)
            else:
                # Short query - embed directly with safety truncation at 77 tokens
                inputs = self.processor(text=[query], padding=True, truncation=True, max_length=77, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    feat = self.model.get_text_features(**inputs)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    embeddings.append(feat.cpu().numpy()[0])
                    
        return np.array(embeddings)
