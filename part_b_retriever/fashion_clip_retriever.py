import os
import sys

# Resolve sibling imports by adding the parent folder to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from part_a_indexer.fashion_clip_embedder import FashionCLIPEmbedder
from part_a_indexer.vector_store import VectorStore

class FashionCLIPRetriever:
    """Retriever for Domain-Specific Fashion-CLIP baseline"""
    def __init__(self, index_dir=None):
        if index_dir is None:
            index_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "indexes"))
        self.index_dir = index_dir
        self.embedder = FashionCLIPEmbedder()
        self.vector_store = VectorStore(dimension=512)
        self.load_index()

    def load_index(self):
        print(f"[*] Loading Fashion-CLIP index from {self.index_dir}...")
        self.vector_store.load(self.index_dir, "fashion_clip")

    def search(self, query_text: str, k: int = 5) -> list:
        """Searches the Fashion-CLIP index using query text."""
        # Extract normalized embedding for the text query
        query_vector = self.embedder.embed_text([query_text])[0]
        # Perform FAISS/NumPy search
        results = self.vector_store.search(query_vector, k=k)
        return results

if __name__ == "__main__":
    # Test execution
    retriever = FashionCLIPRetriever()
    query = "A person in a yellow t-shirt in an outdoor park"
    print(f"\n[*] Testing search for: '{query}'")
    results = retriever.search(query, k=3)
    for idx, r in enumerate(results):
        print(f"  [{idx+1}] File: {r['id']} | Score: {r['score']:.4f}")
