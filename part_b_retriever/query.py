import os
import sys
import argparse

# Resolve sibling imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from siglip_retriever import SigLIPRetriever
from fashion_clip_retriever import FashionCLIPRetriever
from global_local_retriever import GlobalLocalRetriever
from dense_caption_retriever import DenseCaptionRetriever
from marqo_retriever import MarqoFashionSigLIPRetriever

def main():
    parser = argparse.ArgumentParser(description="Multimodal Fashion & Context Retrieval CLI Search Engine")
    parser.add_argument(
        "--query", 
        type=str, 
        required=True, 
        help="Natural language search query"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["siglip", "fashion_clip", "marqo", "fusion_fc", "fusion_marqo", "vlm"], 
        default="fusion_marqo",
        help="Model choice: siglip, fashion_clip, marqo, fusion_fc, fusion_marqo, or vlm"
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=5, 
        help="Number of matching images to return"
    )
    
    args = parser.parse_args()
    
    print(f"\n[*] Initializing Retriever using model: {args.model.upper()}...")
    
    try:
        if args.model == "siglip":
            retriever = SigLIPRetriever()
        elif args.model == "fashion_clip":
            retriever = FashionCLIPRetriever()
        elif args.model == "marqo":
            retriever = MarqoFashionSigLIPRetriever()
        elif args.model == "fusion_fc":
            retriever = GlobalLocalRetriever(model_type="fashion_clip")
        elif args.model == "fusion_marqo":
            retriever = GlobalLocalRetriever(model_type="marqo")
        elif args.model == "vlm":
            retriever = DenseCaptionRetriever()
    except Exception as e:
        print(f"[!] Error loading retriever: {e}")
        print("[!] Ensure you have run indexing first: 'python part_a_indexer/index_dataset.py'")
        return

    print(f"[*] Searching for query: '{args.query}' (k={args.k})")
    
    results = retriever.search(args.query, k=args.k)
    
    print("\n" + "="*50)
    print(f"            SEARCH RESULTS ({args.model.upper()})")
    print("="*50)
    if not results:
        print("  No matching images found.")
    else:
        for idx, r in enumerate(results):
            print(f"  [{idx+1}] Image: {r['id']} | Score: {r['score']:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
