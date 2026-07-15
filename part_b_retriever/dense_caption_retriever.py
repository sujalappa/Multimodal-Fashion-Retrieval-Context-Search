import os
import sys

# Resolve sibling imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer
from part_a_indexer.vector_store import VectorStore

class DenseCaptionRetriever:
    """Retriever for Approach D: Dense Caption Indexing (VLM-Caption Search)
    
    This retriever executes a pure text-to-text vector search using FAISS:
    1. Query text is converted to a Sentence-Transformer embedding (384-dim).
    2. FAISS performs a direct vector search against indexed image captions.
    
    It is extremely fast (sub-milliseconds) and does not read any files at search time.
    """
    def __init__(self, index_dir=None):
        if index_dir is None:
            index_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "indexes"))
        
        self.index_dir = index_dir
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = VectorStore(dimension=384)
        
        self.load_index()

    def load_index(self):
        print(f"[*] Loading Dense Caption index from {self.index_dir}...")
        self.vector_store.load(self.index_dir, "dense_caption")

    def search(self, query_text: str, k: int = 5) -> list:
        # 1. Convert text query to 384-dim Sentence-Transformer embedding
        query_vector = self.embedder.encode([query_text])[0]
        
        # 2. Perform direct FAISS vector search
        results = self.vector_store.search(query_vector, k=k)
        return results

if __name__ == "__main__":
    # Test execution
    retriever = DenseCaptionRetriever()
    query = "Someone wearing a blue shirt sitting on a park bench"
    print(f"\n[*] Testing search for: '{query}'")
    results = retriever.search(query, k=3)
    for idx, r in enumerate(results):
        print(f"  [{idx+1}] File: {r['id']} | Score: {r['score']:.4f}")
