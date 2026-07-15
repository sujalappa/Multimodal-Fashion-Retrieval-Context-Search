import os
import sys

# Resolve sibling imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from part_a_indexer.marqo_siglip_embedder import MarqoFashionSigLIPEmbedder
from part_a_indexer.vector_store import VectorStore

class MarqoFashionSigLIPRetriever:
    """Retriever for Baseline Approach: Marqo-FashionSigLIP Model (768-dim)"""
    def __init__(self, index_dir=None):
        if index_dir is None:
            index_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "indexes"))
        
        self.index_dir = index_dir
        self.embedder = MarqoFashionSigLIPEmbedder()
        self.vector_store = VectorStore(dimension=768)
        
        self.load_index()

    def load_index(self):
        print(f"[*] Loading Marqo-FashionSigLIP index from {self.index_dir}...")
        self.vector_store.load(self.index_dir, "marqo_fashion_siglip")

    def search(self, query_text: str, k: int = 5) -> list:
        # Convert text query to 768-dim text embedding vector
        query_vector = self.embedder.embed_text([query_text])[0]
        
        # Query the FAISS store
        results = self.vector_store.search(query_vector, k=k)
        return results

if __name__ == "__main__":
    retriever = MarqoFashionSigLIPRetriever()
    query = "Someone wearing a blue shirt sitting on a park bench"
    print(f"\n[*] Testing search for: '{query}'")
    results = retriever.search(query, k=3)
    for idx, r in enumerate(results):
        print(f"  [{idx+1}] File: {r['id']} | Score: {r['score']:.4f}")
