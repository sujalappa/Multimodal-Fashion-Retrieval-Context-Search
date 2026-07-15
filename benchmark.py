import os
import json
import time
import numpy as np

# Import retrievers from Part B
from part_b_retriever.fashion_clip_retriever import FashionCLIPRetriever
from part_b_retriever.global_local_retriever import GlobalLocalRetriever
from part_b_retriever.dense_caption_retriever import DenseCaptionRetriever
from part_b_retriever.marqo_retriever import MarqoFashionSigLIPRetriever

# Paths
DATA_DIR = os.path.abspath("data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
INDEX_DIR = os.path.join(DATA_DIR, "indexes")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

# Category names list matching ontology
CATEGORY_NAMES = [
    "shirt", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest",
    "pants", "shorts", "skirt", "coat", "dress", "jumpsuit", "cape", "glasses",
    "hat", "headband, head covering, hair accessory", "tie", "glove", "watch",
    "belt", "tights, stockings", "sock", "shoe", "bag, wallet", "scarf", "hood", "bow"
]

QUERIES = {
    1: "A person in a bright yellow raincoat.",
    2: "Professional business attire inside a modern office.",
    3: "Someone wearing a blue shirt sitting on a park bench.",
    4: "Casual weekend outfit for a city walk.",
    5: "A red tie and a white shirt in a formal setting.",
    6: "A person wearing black shoes and blue jeans.",
    7: "A red sweater and a black skirt.",
    8: "A person in a yellow t-shirt in an outdoor park.",
    9: "Someone in a brown coat on a city street.",
    10: "A formal black dress for an evening event."
}

def get_ground_truth(metadata):
    """Scans the metadata to find matching images for each query using programmatic rules."""
    ground_truth = {i: [] for i in QUERIES.keys()}
    
    for filename, info in metadata.items():
        env = info.get("environment", "").lower()
        objects = info.get("objects", {})
        garments = objects.get("category", [])
        colors = objects.get("colors", [])
        
        def has_garment_color(target_cat_name, target_color):
            for cat_id, color in zip(garments, colors):
                if 0 <= cat_id < len(CATEGORY_NAMES):
                    cat_name = CATEGORY_NAMES[cat_id]
                    if target_cat_name in cat_name and target_color in color.lower():
                        return True
            return False
            
        def env_contains(keywords):
            return any(k in env for k in keywords)

        # Q1: A person in a bright yellow raincoat.
        if (has_garment_color("coat", "yellow") or has_garment_color("jacket", "yellow") or has_garment_color("top", "yellow")):
            ground_truth[1].append(filename)

        # Q2: Professional business attire inside a modern office.
        if env_contains(["office", "desk", "indoor", "workplace"]) and (has_garment_color("blazer", "") or has_garment_color("tie", "") or has_garment_color("jacket", "") or has_garment_color("shirt", "") or has_garment_color("top", "")):
            ground_truth[2].append(filename)

        # Q3: Someone wearing a blue shirt sitting on a park bench.
        if env_contains(["park", "bench", "garden", "outdoor", "grass", "tree"]) and (has_garment_color("shirt", "blue") or has_garment_color("top", "blue") or has_garment_color("sweater", "blue")):
            ground_truth[3].append(filename)

        # Q4: Casual weekend outfit for a city walk.
        if env_contains(["street", "urban", "city", "sidewalk", "outdoor"]) and (has_garment_color("t-shirt", "") or has_garment_color("pants", "") or has_garment_color("shorts", "") or has_garment_color("shoe", "")):
            ground_truth[4].append(filename)

        # Q5: A red tie and a white shirt in a formal setting. (Compositional Adversarial)
        if has_garment_color("tie", "red") and (has_garment_color("shirt", "white") or has_garment_color("top, t-shirt", "white")):
            ground_truth[5].append(filename)
            
        # Q6: A person wearing black shoes and blue jeans.
        if (has_garment_color("shoe", "black") or has_garment_color("shoe", "dark")) and (has_garment_color("pants", "blue") or has_garment_color("pants", "denim")):
            ground_truth[6].append(filename)
            
        # Q7: A red sweater and a black skirt.
        if has_garment_color("sweater", "red") and has_garment_color("skirt", "black"):
            ground_truth[7].append(filename)
            
        # Q8: A person in a yellow t-shirt in an outdoor park.
        if env_contains(["park", "garden", "outdoors", "grass"]) and (has_garment_color("top, t-shirt", "yellow") or has_garment_color("shirt", "yellow")):
            ground_truth[8].append(filename)
            
        # Q9: Someone in a brown coat on a city street.
        if env_contains(["street", "urban", "sidewalk", "road"]) and (has_garment_color("coat", "brown") or has_garment_color("jacket", "brown")):
            ground_truth[9].append(filename)
            
        # Q10: A formal black dress for an evening event.
        if has_garment_color("dress", "black"):
            ground_truth[10].append(filename)
            
    return ground_truth

def calculate_metrics(retrieved_ids, ground_truth_ids):
    """Computes Recall@K, AP@K, and MRR for a single query."""
    if not ground_truth_ids:
        return None
        
    gt_set = set(ground_truth_ids)
    
    r_5 = len(set(retrieved_ids[:5]) & gt_set) / len(gt_set)
    r_10 = len(set(retrieved_ids[:10]) & gt_set) / len(gt_set)
    
    ap_sum = 0.0
    hits = 0
    for idx, rid in enumerate(retrieved_ids[:5]):
        if rid in gt_set:
            hits += 1
            precision_at_idx = hits / (idx + 1)
            ap_sum += precision_at_idx
    ap_5 = ap_sum / min(len(gt_set), 5)
    
    mrr = 0.0
    for idx, rid in enumerate(retrieved_ids):
        if rid in gt_set:
            mrr = 1.0 / (idx + 1)
            break
            
    return {"recall@5": r_5, "recall@10": r_10, "ap@5": ap_5, "mrr": mrr}

def run_evaluation(retriever, ground_truth):
    """Runs the full evaluation query set on a specific retriever."""
    metrics_list = []
    latencies = []
    
    for q_id, query_text in QUERIES.items():
        gt_ids = ground_truth[q_id]
        if len(gt_ids) == 0:
            continue
            
        start_time = time.time()
        results = retriever.search(query_text, k=10)
        latency = (time.time() - start_time) * 1000.0
        
        retrieved_ids = [r["id"] for r in results]
        metrics = calculate_metrics(retrieved_ids, gt_ids)
        
        if metrics:
            metrics_list.append(metrics)
            latencies.append(latency)
            
    if not metrics_list:
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    avg_r5 = np.mean([m["recall@5"] for m in metrics_list])
    avg_r10 = np.mean([m["recall@10"] for m in metrics_list])
    avg_ap5 = np.mean([m["ap@5"] for m in metrics_list])
    avg_mrr = np.mean([m["mrr"] for m in metrics_list])
    avg_latency = np.mean(latencies)
    
    return avg_r5, avg_r10, avg_ap5, avg_mrr, avg_latency

from tabulate import tabulate

# Retrieve metrics calculation functions...
def print_table(headers, rows):
    print(tabulate(rows, headers=headers, tablefmt="github", floatfmt=".4f"))

def main():
    print("[*] Starting Benchmark Evaluation Suite...")
    
    if not os.path.exists(METADATA_PATH):
        print(f"[!] Metadata file not found at {METADATA_PATH}. Please run src/label_dataset.py first.")
        return
        
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    # 1. Compute ground truth mapping counts
    ground_truth = get_ground_truth(metadata)
    print("\n[*] Ground truth mapping targets counts in dataset:")
    for q_id, gt_list in ground_truth.items():
        print(f"  Query {q_id}: '{QUERIES[q_id]}' -> {len(gt_list)} matching target images.")
        
    # 2. Check if indexes exist; if not, index the dataset
    index_files = [
        "siglip_meta.json", "siglip_faiss.index",
        "fashion_clip_meta.json", "fashion_clip_faiss.index",
        "marqo_fashion_siglip_meta.json", "marqo_fashion_siglip_faiss.index",
        "fusion_fc_scene_meta.json", "fusion_fc_scene_faiss.index",
        "fusion_fc_crops_meta.json", "fusion_fc_crops_faiss.index",
        "fusion_marqo_scene_meta.json", "fusion_marqo_scene_faiss.index",
        "fusion_marqo_crops_meta.json", "fusion_marqo_crops_faiss.index",
        "dense_caption_meta.json", "dense_caption_faiss.index"
    ]
    
    # Simple check for NumPy fallbacks too
    npy_files = ["siglip_vectors.npy", "fashion_clip_vectors.npy", "global_fusion_scene_vectors.npy", "global_fusion_crops_vectors.npy"]
    
    index_exists = all(os.path.exists(os.path.join(INDEX_DIR, f)) for f in index_files) or \
                   all(os.path.exists(os.path.join(INDEX_DIR, f)) for f in npy_files if "faiss" not in f)

    if not index_exists:
        print("\n[*] Vector indexes not found. Running master dataset indexer (Part A)...")
        # Run indexer script
        import subprocess
        # Execute indexer script dynamically
        subprocess.run(["python", "part_a_indexer/index_dataset.py"], check=True)
        print("[*] Indexes successfully created!")
        
    # 3. Load Retrievers (Part B)
    print("\n[*] Initializing retrievers...")
    retrievers = {
        "Approach B: Fashion-CLIP (Baseline)": FashionCLIPRetriever(),
        "Approach E: Marqo-FashionSigLIP (Baseline)": MarqoFashionSigLIPRetriever(),
        "Approach C1: Global-Local Fusion (Fashion-CLIP)": GlobalLocalRetriever(model_type="fashion_clip"),
        "Approach C2: Global-Local Fusion (Marqo-SigLIP)": GlobalLocalRetriever(model_type="marqo"),
        "Approach D: Dense Caption Indexing (VLM)": DenseCaptionRetriever()
    }
    
    # 4. Run Benchmarks
    print("\n[*] Running retrieval evaluation queries...")
    results = []
    
    for name, retriever in retrievers.items():
        print(f"  Evaluating {name}...")
        r5, r10, ap5, mrr, latency = run_evaluation(retriever, ground_truth)
        results.append([name, r5, r10, ap5, mrr, latency])
        
    # 5. Output comparison table
    print("\n" + "="*50)
    print("           BENCHMARK PERFORMANCE COMPARISON")
    print("="*50)
    headers = ["Model / Pipeline", "Recall@5", "Recall@10", "mAP@5", "MRR", "Latency (ms)"]
    print_table(headers, results)

if __name__ == "__main__":
    main()
