import os
import json
import numpy as np
import faiss  # Strict import; will raise ImportError if not installed

class VectorStore:
    """Efficient vector indexing and similarity search wrapper using FAISS (strict)"""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.ids = []
        self.metadatas = []
        # IndexFlatIP uses Inner Product. Since our vectors are L2-normalized, 
        # Inner Product is mathematically identical to Cosine Similarity.
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, vectors: np.ndarray, ids: list, metadatas: list):
        assert len(vectors) == len(ids) == len(metadatas), "Length mismatch between vectors, ids, and metadata."
        if len(vectors) == 0:
            return

        # Ensure vectors are float32 (FAISS requirement)
        # NOTE: Vectors are already L2-normalized by the embedder, no need to re-normalize
        vectors_f32 = vectors.astype(np.float32)

        self.ids.extend(ids)
        self.metadatas.extend(metadatas)
        self.index.add(vectors_f32)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list:
        # Format query vector
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        query_vector = query_vector.astype(np.float32)
        
        # L2 normalize query
        norm = np.linalg.norm(query_vector, axis=-1, keepdims=True)
        if norm > 0:
            query_vector = query_vector / norm

        num_elements = self.index.ntotal
        if num_elements == 0:
            return []
            
        k = min(k, num_elements)

        # FAISS search returns scores (inner product) and indices
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "id": self.ids[idx],
                "score": float(score),
                "metadata": self.metadatas[idx]
            })
        return results

    def save(self, directory: str, name: str):
        """Saves the FAISS index and its metadata mapping to disk."""
        os.makedirs(directory, exist_ok=True)
        meta_path = os.path.join(directory, f"{name}_meta.json")
        index_path = os.path.join(directory, f"{name}_faiss.index")
        
        # Save ID and metadata mapping
        with open(meta_path, "w") as f:
            json.dump({"ids": self.ids, "metadatas": self.metadatas}, f, indent=2)

        faiss.write_index(self.index, index_path)

    def load(self, directory: str, name: str):
        """Loads the FAISS index and its metadata mapping from disk."""
        meta_path = os.path.join(directory, f"{name}_meta.json")
        index_path = os.path.join(directory, f"{name}_faiss.index")
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found at {index_path}")
            
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
            self.ids = meta_data["ids"]
            self.metadatas = meta_data["metadatas"]

        self.index = faiss.read_index(index_path)
