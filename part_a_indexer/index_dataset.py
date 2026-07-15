import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from fashion_clip_embedder import FashionCLIPEmbedder
from marqo_siglip_embedder import MarqoFashionSigLIPEmbedder
from vector_store import VectorStore

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
IMAGES_DIR = os.path.join(DATA_DIR, "images")
INDEX_DIR = os.path.join(DATA_DIR, "indexes")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
HF_METADATA_PATH = os.path.join(DATA_DIR, "hf_metadata.json")

# Standard Fashionpedia main garment categories (indices 0 to 26)
CATEGORY_NAMES = [
    "shirt", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest",
    "pants", "shorts", "skirt", "coat", "dress", "jumpsuit", "cape", "glasses",
    "hat", "headband, head covering, hair accessory", "tie", "glove", "watch",
    "belt", "tights, stockings", "sock", "shoe", "bag, wallet", "scarf", "hood", "bow"
]

def index_siglip(image_filenames):
    from siglip_embedder import SigLIPEmbedder
    print("\n[*] Indexing dataset with Vanilla SigLIP-2 (google/siglip2-base-patch16-256)...")
    embedder = SigLIPEmbedder()
    store = VectorStore(dimension=768)
    
    images = []
    ids = []
    metadatas = []
    
    for filename in tqdm(image_filenames, desc="Loading images"):
        img_path = os.path.join(IMAGES_DIR, filename)
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            ids.append(filename)
            metadatas.append({"filename": filename})
        except Exception as e:
            print(f"[!] Error loading {filename}: {e}")
            
    if images:
        embeddings = embedder.embed_images(images, batch_size=16)
        store.add(embeddings, ids, metadatas)
        store.save(INDEX_DIR, "siglip")
        print(f"[*] SigLIP-2 indexing complete! Saved to data/indexes/siglip")

def index_fashion_clip(image_filenames):
    print("\n[*] Indexing dataset with Fashion-CLIP (512-dim)...")
    embedder = FashionCLIPEmbedder()
    store = VectorStore(dimension=512)
    
    images = []
    ids = []
    metadatas = []
    
    for filename in tqdm(image_filenames, desc="Loading images"):
        img_path = os.path.join(IMAGES_DIR, filename)
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            ids.append(filename)
            metadatas.append({"filename": filename})
        except Exception as e:
            print(f"[!] Error loading {filename}: {e}")
            
    if images:
        embeddings = embedder.embed_images(images, batch_size=32)
        store.add(embeddings, ids, metadatas)
        store.save(INDEX_DIR, "fashion_clip")
        print(f"[*] Fashion-CLIP indexing complete! Saved to data/indexes/fashion_clip")

def index_marqo_siglip(image_filenames):
    print("\n[*] Indexing dataset with Marqo-FashionSigLIP (768-dim)...")
    embedder = MarqoFashionSigLIPEmbedder()
    store = VectorStore(dimension=768)
    
    images = []
    ids = []
    metadatas = []
    
    for filename in tqdm(image_filenames, desc="Loading images"):
        img_path = os.path.join(IMAGES_DIR, filename)
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            ids.append(filename)
            metadatas.append({"filename": filename})
        except Exception as e:
            print(f"[!] Error loading {filename}: {e}")
            
    if images:
        embeddings = embedder.embed_images(images, batch_size=16)
        store.add(embeddings, ids, metadatas)
        store.save(INDEX_DIR, "marqo_fashion_siglip")
        print(f"[*] Marqo-FashionSigLIP indexing complete! Saved to data/indexes/marqo_fashion_siglip")

def index_global_local(image_filenames, model_type="fashion_clip"):
    dim = 512 if model_type == "fashion_clip" else 768
    name_suffix = "fc" if model_type == "fashion_clip" else "marqo"
    
    print(f"\n[*] Indexing dataset with Global-Local Semantic Fusion ({model_type.upper()}, {dim}-dim)...")
    embedder = FashionCLIPEmbedder() if model_type == "fashion_clip" else MarqoFashionSigLIPEmbedder()
    
    global_store = VectorStore(dimension=dim)
    local_store = VectorStore(dimension=dim)
    
    # Load metadata for ground-truth cropping
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
    elif os.path.exists(HF_METADATA_PATH):
        with open(HF_METADATA_PATH, "r") as f:
            metadata = json.load(f)
    else:
        raise FileNotFoundError("[!] No metadata file found for ground-truth crop coordinates.")

    batch_size = 32
    # Use a manual tqdm loop updating step-by-step
    pbar = tqdm(total=len(image_filenames), desc=f"Processing images (Fusion {name_suffix.upper()})")
    
    for i in range(0, len(image_filenames), batch_size):
        batch_filenames = image_filenames[i:i+batch_size]
        
        batch_images = []
        loaded_filenames = []
        
        # 1. Load images in this batch
        for filename in batch_filenames:
            img_path = os.path.join(IMAGES_DIR, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
                loaded_filenames.append(filename)
            except Exception as e:
                print(f"\n[!] Error loading {filename} in batch: {e}")
                
        if not batch_images:
            pbar.update(len(batch_filenames))
            continue
            
        try:
            # 2. Extract Global scene background embeddings for the entire batch in parallel
            global_embs = embedder.embed_images(batch_images, batch_size=len(batch_images))
            global_store.add(
                vectors=global_embs,
                ids=loaded_filenames,
                metadatas=[{"filename": fname} for fname in loaded_filenames]
            )
            
            # 3. Extract and pad crops across all loaded images in the batch
            batch_crops = []
            crop_mappings = []  # List of tuples: (filename, category, bbox)
            
            for img, filename in zip(batch_images, loaded_filenames):
                img_info = metadata.get(filename, {})
                objects = img_info.get("objects", {})
                bboxes = objects.get("bbox", [])
                categories = objects.get("category", [])
                
                for bbox, cat_id in zip(bboxes, categories):
                    if 0 <= cat_id < len(CATEGORY_NAMES):
                        cat_name = CATEGORY_NAMES[cat_id]
                        
                        xmin, ymin, xmax, ymax = bbox
                        xmin = max(0, int(xmin))
                        ymin = max(0, int(ymin))
                        xmax = min(img.width, int(xmax))
                        ymax = min(img.height, int(ymax))
                        
                        w = xmax - xmin
                        h = ymax - ymin
                        if w >= 10 and h >= 10:
                            crop_img = img.crop((xmin, ymin, xmax, ymax))
                            
                            max_dim = max(w, h)
                            padded_img = Image.new("RGB", (max_dim, max_dim), (128, 128, 128))
                            paste_x = (max_dim - w) // 2
                            paste_y = (max_dim - h) // 2
                            padded_img.paste(crop_img, (paste_x, paste_y))
                            
                            batch_crops.append(padded_img)
                            crop_mappings.append((filename, cat_name, [xmin, ymin, xmax, ymax]))
            
            # 4. Embed all crops in the batch in parallel
            if batch_crops:
                crop_embs = embedder.embed_images(batch_crops, batch_size=len(batch_crops))
                
                ids = [m[0] for m in crop_mappings]
                metadatas = [
                    {
                        "filename": m[0],
                        "category": m[1],
                        "bbox": m[2]
                    } for m in crop_mappings
                ]
                
                local_store.add(crop_embs, ids, metadatas)
                
            # Close PIL Images to release memory
            for img in batch_images:
                img.close()
                
        except Exception as e:
            print(f"\n[!] Error processing batch: {e}")
            
        pbar.update(len(batch_filenames))
        
    pbar.close()
    
    global_store.save(INDEX_DIR, f"fusion_{name_suffix}_scene")
    local_store.save(INDEX_DIR, f"fusion_{name_suffix}_crops")
    print(f"[*] Global-Local ({name_suffix.upper()}) indexing complete!")

def index_dense_captions(image_filenames):
    print("\n[*] Indexing dataset with VLM Dense Captions (Sentence-Transformers)...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    store = VectorStore(dimension=384)
    
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"[!] Metadata file not found at {METADATA_PATH}. Run generate_dense_captions.py first.")
        
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
        
    captions = []
    ids = []
    metadatas = []
    
    for filename in tqdm(image_filenames, desc="Loading dense captions"):
        if filename in metadata:
            caption = metadata[filename].get("dense_caption", "")
            if caption:
                captions.append(caption)
                ids.append(filename)
                metadatas.append({"filename": filename, "dense_caption": caption})
                
    if captions:
        print(f"[*] Extracting Sentence-Transformer embeddings for {len(captions)} captions...")
        embeddings = embedder.encode(captions, batch_size=32, show_progress_bar=True)
        store.add(embeddings, ids, metadatas)
        store.save(INDEX_DIR, "dense_caption")
        print(f"[*] Dense Caption indexing complete!")
    else:
        print("[!] No dense captions found to index.")

def main():
    parser = argparse.ArgumentParser(description="Part A: Dataset Indexer Workflow")
    parser.add_argument(
        "--model",
        type=str,
        choices=["siglip", "fashion_clip", "marqo", "fusion_fc", "fusion_marqo", "dense_caption", "all"],
        default="all",
        help="Specify which approach to index (default: all)"
    )
    args = parser.parse_args()

    if not os.path.exists(IMAGES_DIR):
        print(f"[!] Images directory not found at {IMAGES_DIR}.")
        return
        
    image_filenames = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_filenames:
        print("[!] No images found in data/images/")
        return
        
    print(f"[*] Found {len(image_filenames)} images. Target: {args.model.upper()}")
    
    if args.model == "siglip":
        index_siglip(image_filenames)
    elif args.model == "fashion_clip":
        index_fashion_clip(image_filenames)
    elif args.model == "marqo":
        index_marqo_siglip(image_filenames)
    elif args.model == "fusion_fc":
        index_global_local(image_filenames, model_type="fashion_clip")
    elif args.model == "fusion_marqo":
        index_global_local(image_filenames, model_type="marqo")
    elif args.model == "dense_caption":
        index_dense_captions(image_filenames)
    else:
        index_siglip(image_filenames)
        index_fashion_clip(image_filenames)
        index_marqo_siglip(image_filenames)
        index_global_local(image_filenames, model_type="fashion_clip")
        index_global_local(image_filenames, model_type="marqo")
        index_dense_captions(image_filenames)
        
    print(f"\n[*] Indexing pipeline complete for: {args.model.upper()}")

if __name__ == "__main__":
    main()
