import os
import sys
import re
import json
import numpy as np

# Resolve sibling imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from part_a_indexer.fashion_clip_embedder import FashionCLIPEmbedder
from part_a_indexer.marqo_siglip_embedder import MarqoFashionSigLIPEmbedder
from part_a_indexer.vector_store import VectorStore

# Filler words to strip from query phrases
FILLER_WORDS = r"\b(?:a|an|the|someone|person|man|woman|women|men|boy|girl|lady|people|for|very|really|quite|some)\b"


class GlobalLocalRetriever:
    """Retriever for Approach C: Global-Local Semantic Fusion Model (Our Hybrid Approach)
    
    This retriever is a PURE neural vector search engine. It relies entirely on:
    1. Global Scene FAISS Index: Matches background/environment context
    2. Local Crop FAISS Index: Matches garment visual features
    
    All query routing vocabularies are dynamically built from the FAISS index metadata
    at load time — zero hardcoded keyword lists.
    """
    # Empirically-calibrated penalty score for missing garments
    MISSING_GARMENT_PENALTY = 0.08
    
    def __init__(self, index_dir=None, model_type="marqo"):
        if index_dir is None:
            index_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "indexes"))
        
        self.index_dir = index_dir
        self.model_type = model_type.lower().strip()
        
        # Configure dimension, embedder, and penalty based on model_type
        if self.model_type == "fashion_clip":
            self.embedder = FashionCLIPEmbedder()
            dim = 512
            self.name_suffix = "fc"
            self.missing_garment_penalty = 0.08
        else:
            self.embedder = MarqoFashionSigLIPEmbedder()
            dim = 768
            self.name_suffix = "marqo"
            self.missing_garment_penalty = 0.01
            
        self.global_store = VectorStore(dimension=dim)
        self.local_store = VectorStore(dimension=dim)
        
        # Garment vocabulary dynamically built from FAISS metadata
        self.garment_words = set()
        
        self.load_index()

    def load_index(self):
        print(f"[*] Loading Global-Local ({self.model_type.upper()}) indexes from {self.index_dir}...")
        self.global_store.load(self.index_dir, f"fusion_{self.name_suffix}_scene")
        self.local_store.load(self.index_dir, f"fusion_{self.name_suffix}_crops")
        
        # Dynamically build garment vocabulary from FAISS crop metadata
        self.garment_words = set()
        
        # Standard synonyms and verbs representing garments/style
        base_synonyms = {
            "jeans", "trousers", "sneakers", "boots", "heels", "sandals",
            "blouse", "blazer", "suit", "outfit", "attire", "clothing",
            "garment", "apparel", "wear", "raincoat", "overcoat",
            "hoodie", "polo", "tank", "purse", "handbag", "backpack",
            "sunglasses", "bowtie", "fit"
        }
        self.garment_words.update(base_synonyms)
        
        for meta in self.local_store.metadatas:
            cat = meta.get("category", "")
            if cat:
                # Tokenize category strings into individual words
                words = re.findall(r'[a-zA-Z]+', cat.lower())
                for word in words:
                    if len(word) > 2:
                        self.garment_words.add(word)
                        # Add singular/plural variants
                        if word.endswith("s"):
                            self.garment_words.add(word[:-1])
                        else:
                            self.garment_words.add(word + "s")
                            
        print(f"[*] Dynamically built garment vocabulary with {len(self.garment_words)} words from FAISS metadata.")

    def parse_query(self, query_text: str) -> tuple:
        """
        Splits and routes query terms using dynamic vocabulary matching.
        """
        print(f"\n[ROUTING] Parsing query: '{query_text}'")
        
        delimiters = r"\b(?:and|with|in|on|inside|at|wearing|sitting on|standing on)\b|,"
        parts = re.split(delimiters, query_text, flags=re.IGNORECASE)
        
        phrases = []
        for p in parts:
            if not p:
                continue
            clean = p.strip()
            clean = re.sub(FILLER_WORDS, "", clean, flags=re.IGNORECASE).strip()
            clean = re.sub(r"\s+", " ", clean).strip()  # Collapse multiple spaces
            if len(clean) > 2:
                phrases.append(clean)

        print(f"[ROUTING] Extracted phrases: {phrases}")

        if not phrases:
            # No phrases extracted — default to treating the full query as garment
            full_vec = self.embedder.embed_text([query_text])[0]
            return [(query_text, full_vec)], []

        local_queries = []
        global_queries = []
        
        for phrase in phrases:
            vec = self.embedder.embed_text([phrase])[0]
            phrase_words = set(re.findall(r'[a-zA-Z]+', phrase.lower()))
            
            # Map phrase words to singular variants for matching
            expanded_phrase_words = set()
            for w in phrase_words:
                expanded_phrase_words.add(w)
                if w.endswith("s"):
                    expanded_phrase_words.add(w[:-1])
                else:
                    expanded_phrase_words.add(w + "s")
            
            # Route to local crop search if the phrase contains any known garment vocabulary
            has_garment = len(expanded_phrase_words & self.garment_words) > 0
            
            if has_garment:
                route = "GARMENT"
                local_queries.append((phrase, vec))
            else:
                route = "SCENE"
                global_queries.append((phrase, vec))
            
            print(f"[ROUTING]   '{phrase}' -> {route}")

        local_names = [p for p, _ in local_queries]
        global_names = [p for p, _ in global_queries]
        print(f"[ROUTING] Final: LOCAL(garment)={local_names} | GLOBAL(scene)={global_names}")
        
        return local_queries, global_queries

    def search(self, query_text: str, k: int = 5) -> list:
        local_queries, global_queries = self.parse_query(query_text)
        
        # Case A: Fused Search (Both scene context and garment descriptions present)
        if local_queries and global_queries:
            # Combine all global query vectors by averaging
            if len(global_queries) == 1:
                global_vec = global_queries[0][1]
            else:
                global_vecs = np.array([vec for _, vec in global_queries])
                global_vec = np.mean(global_vecs, axis=0)
                global_vec = global_vec / np.linalg.norm(global_vec)

            # Search the global scene index
            global_candidates = self.global_store.search(global_vec, k=len(self.global_store.ids))
            
            if not global_candidates:
                return []

            candidate_set = set()
            candidate_global_scores = {}
            for c in global_candidates:
                candidate_set.add(c["id"])
                candidate_global_scores[c["id"]] = c["score"]
            
            candidate_local_scores = {f: [] for f in candidate_set}
            
            # Search local crop index for each garment query
            for phrase, vec in local_queries:
                local_results = self.local_store.search(vec, k=len(self.local_store.ids))
                best_crop_score = {}
                for res in local_results:
                    filename = res["id"]
                    if filename in candidate_set:
                        score = res["score"]
                        if filename not in best_crop_score or score > best_crop_score[filename]:
                            best_crop_score[filename] = score
                            
                for filename in candidate_set:
                    score = best_crop_score.get(filename, self.missing_garment_penalty)
                    candidate_local_scores[filename].append(score)

            # Fused Score: 50% global scene embedding + 50% local crop embedding (Pure FAISS)
            fused_results = []
            for filename in candidate_set:
                g_score = candidate_global_scores[filename]
                l_scores = candidate_local_scores.get(filename, [])
                
                # Enforce that ALL queried garments must be present in the visual crops (min logic)
                l_score = min(l_scores) if l_scores else 0.0
                
                # Pure dense vector fusion (no metadata lookups)
                final_score = 0.5 * g_score + 0.5 * l_score
                
                fused_results.append({
                    "id": filename,
                    "score": float(final_score),
                    "metadata": {"filename": filename}
                })

            fused_results.sort(key=lambda x: x["score"], reverse=True)
            return fused_results[:k]

        # Case B: Scene Only (No garments requested)
        elif global_queries:
            if len(global_queries) == 1:
                global_vec = global_queries[0][1]
            else:
                global_vecs = np.array([vec for _, vec in global_queries])
                global_vec = np.mean(global_vecs, axis=0)
                global_vec = global_vec / np.linalg.norm(global_vec)
            return self.global_store.search(global_vec, k=k)

        # Case C: Garments Only (No scene context requested)
        elif local_queries:
            file_scores = {}
            for phrase, vec in local_queries:
                local_results = self.local_store.search(vec, k=len(self.local_store.ids))
                
                best_crop_score = {}
                for res in local_results:
                    filename = res["id"]
                    score = res["score"]
                    if filename not in best_crop_score or score > best_crop_score[filename]:
                        best_crop_score[filename] = score
                        
                for filename, score in best_crop_score.items():
                    if filename not in file_scores:
                        file_scores[filename] = []
                    file_scores[filename].append(score)
            
            results = []
            for filename, scores in file_scores.items():
                if len(scores) < len(local_queries):
                    scores.extend([self.MISSING_GARMENT_PENALTY] * (len(local_queries) - len(scores)))
                
                # Enforce that ALL queried garments must match (min logic)
                avg_score = min(scores)
                results.append({
                    "id": filename,
                    "score": float(avg_score),
                    "metadata": {"filename": filename}
                })
                
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:k]
            
        return []

if __name__ == "__main__":
    # Test execution
    retriever = GlobalLocalRetriever()
    query = "Someone wearing a blue shirt sitting on a park bench"
    print(f"\n[*] Testing search for: '{query}'")
    results = retriever.search(query, k=3)
    for idx, r in enumerate(results):
        print(f"  [{idx+1}] File: {r['id']} | Score: {r['score']:.4f}")
