"""
Score Distribution Calibration Script

This script measures the actual SigLIP-2 cosine similarity score distributions
for the crop index, so we can set an empirically-calibrated penalty constant
instead of guessing.

It also validates the query routing heuristic by testing all benchmark queries
and printing their routing decisions.
"""
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from part_a_indexer.siglip_embedder import SigLIPEmbedder
from part_a_indexer.vector_store import VectorStore

INDEX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "indexes"))

def calibrate_scores():
    """Measure the actual similarity score distribution in the crop index."""
    print("="*60)
    print("  PART 1: CROP SIMILARITY SCORE DISTRIBUTION")
    print("="*60)
    
    embedder = SigLIPEmbedder()
    
    # Load crop index
    local_store = VectorStore(dimension=768)
    local_store.load(INDEX_DIR, "global_fusion_crops")
    print(f"\n[*] Loaded {len(local_store.ids)} crop vectors.\n")
    
    # Test queries: things that SHOULD match specific crops
    positive_queries = [
        "red dress",
        "blue shirt",
        "black shoes",
        "white t-shirt",
        "brown jacket",
        "yellow coat",
        "green sweater",
        "black pants",
    ]
    
    # Test queries: things that should NOT match any crop well
    negative_queries = [
        "park bench in autumn",
        "modern office building",
        "beach sunset",
        "city street at night",
        "studio lighting setup",
    ]
    
    print("[*] Positive queries (garments that should match crops):")
    print("-"*60)
    pos_top1_scores = []
    pos_top5_scores = []
    for query in positive_queries:
        vec = embedder.embed_text([query])[0]
        results = local_store.search(vec, k=10)
        top1 = results[0]["score"] if results else 0
        top5_avg = np.mean([r["score"] for r in results[:5]]) if results else 0
        pos_top1_scores.append(top1)
        pos_top5_scores.append(top5_avg)
        print(f"  '{query:<25}' -> Top-1: {top1:.4f} | Top-5 avg: {top5_avg:.4f}")
    
    print(f"\n  POSITIVE SUMMARY:")
    print(f"    Top-1 range: [{min(pos_top1_scores):.4f}, {max(pos_top1_scores):.4f}]")
    print(f"    Top-1 mean:  {np.mean(pos_top1_scores):.4f}")
    print(f"    Top-5 mean:  {np.mean(pos_top5_scores):.4f}")
    
    print(f"\n[*] Negative queries (scenes that should NOT match crops):")
    print("-"*60)
    neg_top1_scores = []
    neg_top5_scores = []
    for query in negative_queries:
        vec = embedder.embed_text([query])[0]
        results = local_store.search(vec, k=10)
        top1 = results[0]["score"] if results else 0
        top5_avg = np.mean([r["score"] for r in results[:5]]) if results else 0
        neg_top1_scores.append(top1)
        neg_top5_scores.append(top5_avg)
        print(f"  '{query:<35}' -> Top-1: {top1:.4f} | Top-5 avg: {top5_avg:.4f}")
    
    print(f"\n  NEGATIVE SUMMARY:")
    print(f"    Top-1 range: [{min(neg_top1_scores):.4f}, {max(neg_top1_scores):.4f}]")
    print(f"    Top-1 mean:  {np.mean(neg_top1_scores):.4f}")
    print(f"    Top-5 mean:  {np.mean(neg_top5_scores):.4f}")
    
    # Recommended penalty = midpoint between negative top-1 mean and positive top-5 mean
    recommended_penalty = (np.mean(neg_top1_scores) + np.mean(pos_top5_scores)) / 2
    print(f"\n  RECOMMENDED MISSING-GARMENT PENALTY: {recommended_penalty:.4f}")
    print(f"  (Midpoint between negative top-1 mean and positive top-5 mean)")
    
    # --- PART 2: Global scene score distribution ---
    print(f"\n\n{'='*60}")
    print("  PART 2: GLOBAL SCENE SCORE DISTRIBUTION")
    print("="*60)
    
    global_store = VectorStore(dimension=768)
    global_store.load(INDEX_DIR, "global_fusion_scene")
    print(f"\n[*] Loaded {len(global_store.ids)} scene vectors.\n")
    
    scene_queries = [
        "outdoor park with trees and grass",
        "indoor modern office",
        "city street sidewalk",
        "beach near ocean",
        "fashion runway show",
        "studio with white background",
    ]
    
    for query in scene_queries:
        vec = embedder.embed_text([query])[0]
        results = global_store.search(vec, k=5)
        top1 = results[0]["score"] if results else 0
        top5_avg = np.mean([r["score"] for r in results[:5]]) if results else 0
        print(f"  '{query:<40}' -> Top-1: {top1:.4f} | Top-5 avg: {top5_avg:.4f}")


def validate_query_routing():
    """Test the query routing heuristic on all benchmark queries."""
    print(f"\n\n{'='*60}")
    print("  PART 3: QUERY ROUTING VALIDATION")
    print("="*60)
    
    embedder = SigLIPEmbedder()
    
    # Create reference vectors
    ref_garment = embedder.embed_text(["clothing item, garment, apparel, shoes, bag, hat, tie"])[0]
    ref_scene = embedder.embed_text(["background scene, location, place, setting, environment, background"])[0]
    
    # All individual phrases we need to route correctly
    test_phrases = [
        # Expected: GARMENT (local)
        ("bright yellow raincoat", "GARMENT"),
        ("blue shirt", "GARMENT"),
        ("red tie", "GARMENT"),
        ("white shirt", "GARMENT"),
        ("black shoes", "GARMENT"),
        ("blue jeans", "GARMENT"),
        ("red sweater", "GARMENT"),
        ("black skirt", "GARMENT"),
        ("yellow t-shirt", "GARMENT"),
        ("brown coat", "GARMENT"),
        ("formal black dress", "GARMENT"),
        ("business attire", "GARMENT"),
        ("casual weekend outfit", "GARMENT"),
        ("Professional business attire", "GARMENT"),
        
        # Expected: SCENE (global)
        ("park bench", "SCENE"),
        ("modern office", "SCENE"),
        ("city walk", "SCENE"),
        ("formal setting", "SCENE"),
        ("outdoor park", "SCENE"),
        ("city street", "SCENE"),
        ("evening event", "SCENE"),
        ("inside a modern office", "SCENE"),
    ]
    
    correct = 0
    total = len(test_phrases)
    
    print(f"\n{'Phrase':<35} {'Expected':>10} {'Predicted':>10} {'Garment Sim':>12} {'Scene Sim':>12} {'Result':>8}")
    print("-"*95)
    
    for phrase, expected in test_phrases:
        vec = embedder.embed_text([phrase])[0]
        sim_garment = float(np.dot(vec, ref_garment))
        sim_scene = float(np.dot(vec, ref_scene))
        predicted = "GARMENT" if sim_garment > sim_scene else "SCENE"
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        status = "OK" if is_correct else "WRONG"
        
        print(f"  {phrase:<33} {expected:>10} {predicted:>10} {sim_garment:>12.4f} {sim_scene:>12.4f} {status:>8}")
    
    print(f"\n  ROUTING ACCURACY: {correct}/{total} ({100*correct/total:.1f}%)")
    
    if correct < total:
        print("\n  [!] WARNING: Some phrases are being misrouted!")
        print("  [!] Consider adjusting reference vectors or using a different routing strategy.")


if __name__ == "__main__":
    calibrate_scores()
    validate_query_routing()
