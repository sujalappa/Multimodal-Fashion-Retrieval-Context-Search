# Multimodal Fashion Retrieval & Context Search

This repository contains the source code for a multimodal fashion search engine. The project implements and benchmarks multiple search architectures, comparing single-stage global baseline models against segmented Global-Local Fusion models and generative Vision-Language Model (VLM) caption searches.

---

## 1. Directory Structure & File Map

### Indexing Engine (`part_a_indexer/`)
*   `index_dataset.py`: The master indexer script. Reads raw images/metadata, calls the respective embedders, and saves FAISS indices to `data/indexes/`.
*   `vector_store.py`: A wrapper class around the FAISS library to handle vector database creation, addition, saving, and searching.
*   `marqo_siglip_embedder.py`: Wrapper for the **Marqo-FashionSigLIP** (768-dim) model, loaded via `open_clip` with custom sentence-level text chunking.
*   `fashion_clip_embedder.py`: Wrapper for the **Fashion-CLIP** (512-dim) model, loaded via Hugging Face `transformers` with text chunking.
*   `siglip_embedder.py`: Wrapper for the vanilla **SigLIP-2** (768-dim) model.
*   `fashion_detector.py`: **[FALLBACK ONLY]** Contains a fine-tuned Conditional-DETR fashion object detector model. This script is **not used** in the main production pipeline due to significant performance limitations on complex outfits; it is kept only as a conceptual fallback for un-annotated raw data.

### Retrieval Engine (`part_b_retriever/`)
*   `query.py`: The command-line utility to run manual text searches against any of the indexed models.
*   `global_local_retriever.py`: Implements the **Global-Local Fusion (Hybrid)** retrieval logic, including dynamic query splitting, min-logic crop scores, and calibrated missing garment penalties.
*   `marqo_retriever.py`: Implements the single-stage **Marqo-FashionSigLIP** baseline search.
*   `fashion_clip_retriever.py`: Implements the single-stage **Fashion-CLIP** baseline search.
*   `dense_caption_retriever.py`: Implements the **VLM Dense Caption** text-to-text search.

### Ingestion & Data Preparation (`src/`)
*   `generate_dense_captions.py`: Connects to the Gemini-3.1-Flash-Lite API to generate rich, descriptive paragraphs for the images.
*   `download_dataset.py`: Script to download validation images from Hugging Face detection-datasets/fashionpedia.
*   `label_dataset.py`: Initial metadata labeling script.

### Evaluation Runner
*   `benchmark.py`: Evaluates all five search pipelines side-by-side on 700 images against 10 distinct queries, calculating Recall@5, Recall@10, mAP@5, MRR, and search Latency.

### Help Utility
*   `help.py`: A simple command-line guide showing all execution commands, steps, and options.

---

## 2. Setup & Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/sujalappa/Multimodal-Fashion-Retrieval-Context-Search.git
cd Multimodal-Fashion-Retrieval-Context-Search
```

### Step 2: Install Dependencies
Ensure you have PyTorch installed, then run:
```bash
pip install -r requirements.txt
```
*(Note: Key libraries installed include `open-clip-torch`, `sentence-transformers`, `faiss-cpu`, `google-generativeai`, and `pillow`)*.

### Step 3: Set Gemini API Key (Required for Ingestion)
```bash
# Windows CMD
set GEMINI_API_KEY=your_key_here

# Windows PowerShell
$env:GEMINI_API_KEY="your_key_here"
```

---

## 3. Execution Workflow

To run and test the complete pipeline, execute the following commands in order:

### Step A: Download & Prepare the Dataset
```bash
python src/download_dataset.py
```

### Step B: Generate VLM Dense Captions
Queries the Gemini API to describe images and populate `data/metadata.json`:
```bash
python src/generate_dense_captions.py
```

### Step C: Index the Models
Builds all the FAISS indices concurrently (saves them to `data/indexes/`):
```bash
python part_a_indexer/index_dataset.py --model all
```

### Step D: Run the Benchmark
Run the evaluator to generate the side-by-side performance table:
```bash
python benchmark.py
```

### Step E: Search Manually via the CLI
Test any search query on a specific model using `query.py`:
```bash
# Test the best baseline model (Marqo-FashionSigLIP)
python part_b_retriever/query.py --model marqo --query "A red tie and a white shirt in a formal setting"

# Test the Marqo Global-Local Fusion model
python part_b_retriever/query.py --model fusion_marqo --query "Someone wearing a blue denim shirt and brown pants sitting on a railing outdoors"
```
