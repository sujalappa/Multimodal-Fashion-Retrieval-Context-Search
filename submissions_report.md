# Multimodal Fashion & Context Retrieval: ML Technical Report

**Prepared for**: Glance ML Internship Assignment  
**Author**: Sujal  
**Focus**: Multimodal search engine benchmarking & Global-Local Semantic Fusion architecture.  

---

## 1. Project Overview
This project presents an intelligent search engine designed to retrieve fashion images based on natural language descriptions. The core challenge of fashion retrieval lies in understanding **compositionality** (e.g., distinguishing a *"red tie and a white shirt"* from a *"white tie and a red shirt"*), **attribute binding** (linking colors to specific garments), and **scene context** (e.g., distinguishing between an office interior and a park setting). 

We implemented a modular benchmarking system comparing four distinct approaches on a dataset of 700 fashion images to mathematically evaluate performance.

---

## 2. Comparative Analysis of Approaches

We evaluated four different architectures to identify the trade-offs in search accuracy, latency, and compositionality.

| Approach | Architecture Description | Pros | Cons | Best Suited For |
| :--- | :--- | :--- | :--- | :--- |
| **Approach A: Vanilla SigLIP-2** | Pairwise Sigmoid loss pre-trained dual-encoder (`google/siglip2-base-patch16-256`). | State-of-the-art zero-shot retrieval; excellent visual semantics; 768 dimensions. | Struggles with compositionality when queries have multiple garments/colors. | SOTA baseline for text-to-image queries without complex binding. |
| **Approach B: Fashion-CLIP** | OpenAI CLIP fine-tuned on fashion-text pairs (`patrickjohncyh/fashion-clip`). | Understands complex fashion terms (e.g., "monk straps", "tweed"). | Poor zero-shot performance on background settings (e.g., "park bench"). | Pure fashion retrieval without location/scene queries. |
| **Approach C: Global-Local Fusion** | **[Our Architecture]** Object detector crops garments for local embeddings; SigLIP-2 extracts scene/garment vectors. | **Solves compositionality**; binds colors directly to garments; context-aware. | Requires running an object detector during indexing (offline overhead). | **Fine-grained, compositional, and context-aware fashion retrieval.** |

---

## 3. Chosen Approach: Global-Local Semantic Fusion

### Architecture Blueprint
Our chosen architecture, **Global-Local Semantic Fusion**, decomposes the image search task into two semantic layers: **global environment (where)** and **local garments (what)**.

```mermaid
graph TD
    %% Indexing Pipeline %%
    subgraph Indexing Phase (Offline)
        Img[Input Image] --> GlobalSigLIP[SigLIP Global Encoder]
        GlobalSigLIP --> GlobalVec[Global Scene Vector]
        
        Img --> DETR[Fashion Object Detector]
        DETR --> Crops[Garment Crops: tops, bottoms, shoes]
        Crops --> LocalSigLIP[SigLIP Crop Encoder]
        LocalSigLIP --> CropVecs[Local Garment Vectors]
        
        GlobalVec --> GlobalStore[(Global FAISS Index)]
        CropVecs --> LocalStore[(Local FAISS Index)]
    end

    %% Retrieval Pipeline %%
    subgraph Retrieval Phase (Online)
        Query[User Query] --> Parse[Semantic Query Router]
        Parse -->|scene terms| GlobalQuery[Global Query Embedding]
        Parse -->|garment terms| LocalQueries[Local Query Embeddings]
        
        GlobalQuery --> SearchGlobal[Search Global Index]
        SearchGlobal --> Candidates[Top 150 Candidates]
        
        LocalQueries --> SearchLocal[Search Local Index]
        SearchLocal --> CandidateCrops[Crop Match Scores]
        
        Candidates --> Fusion[Weighted Fusion Module]
        CandidateCrops --> Fusion
        Fusion --> Rank[Top K Retrieved Images]
    end
    
    style Indexing Phase (Offline) fill:#f5f5f5,stroke:#333,stroke-width:2px;
    style Retrieval Phase (Online) fill:#e6f2ff,stroke:#333,stroke-width:2px;
```

### Technical Workflow

#### A. Offline Indexing
1. **Garment Detection**: For every image, we run a pre-trained `Conditional-DETR` model fine-tuned on fashion datasets. It outputs bounding boxes for 7 garment classes: `top`, `bottom`, `dress`, `hat`, `shoes`, `outer`, and `bag`.
2. **Local Feature Extraction**: We crop each detected bounding box and run it through `SigLIP` to generate a 768-dimensional local vector. These represent specific garment features (color, texture).
3. **Global Feature Extraction**: We run the full, uncropped image through `SigLIP` to capture the overall scene and background.
4. **Vector Storage**: Global and local vectors are stored in separate **FAISS** indexes.

#### B. Online Retrieval & Zero-Shot Query Parsing
When a user enters a query like *"Someone wearing a blue shirt sitting on a park bench"*:
1. **Semantic Query Routing**: We split the query into phrases using prepositions (`in`, `on`, `at`) and conjunctions (`and`). We run a zero-shot semantic classifier (using SigLIP cosine similarity against a reference "garment" vs "scene location" embedding) to route the phrases:
   * **Local garment queries**: `"blue shirt"`
   * **Global scene query**: `"park bench"`
2. **Coarse-to-Fine Search**:
   * **Step 1 (Coarse Filter)**: We query the global FAISS index with the global scene query to retrieve a candidate pool of the top 150 images.
   * **Step 2 (Local Alignment)**: For each candidate image, we compute the similarity of its local cropped garment vectors against `"blue shirt"`, taking the maximum score.
3. **Weighted Fusion**: We compute the final score using a weighted sum:
   $$\text{Score} = w_{\text{global}} \cdot S_{\text{global}} + w_{\text{local}} \cdot S_{\text{local}}$$
   We use $w_{\text{global}} = 0.4$ and $w_{\text{local}} = 0.6$. This ensures that both the "park bench" environment and the "blue shirt" are present, preventing false positives.

---

## 4. Future Work & Scalability

### A. Extending to Locations (Cities, Places) and Weather
To scale the retrieval system to understand location attributes (e.g. "Paris street style", "New York fashion") and weather context (e.g. "sunny day outfit", "rainy weather"):

1. **VLM Captioning Pipeline (Offline)**:
   * Run a high-performance vision-language model (VLM) like `SigLIP-2` or a lightweight `PaliGemma` over incoming images to generate dense description tags:
     * *Location features*: urban city streets, landmarks, architectural styles.
     * *Weather features*: wet pavement, sunny lens flare, overcast lighting, falling snow.
2. **Semantic Knowledge Graph**:
   * Map weather terms to expected fashion items (e.g., "rainy" $\rightarrow$ raincoat, umbrella; "sunny" $\rightarrow$ sunglasses, shorts). This allows the query router to perform semantic expansion when a user queries "rainy day city outfit".
3. **Hierarchical Metadata Fusion**:
   * Index location and weather tags as sparse categorical features alongside dense vision vectors in a database like **Qdrant**. This allows filtering candidates by place metadata (e.g., `location == "urban_street"`) before applying vector search, guaranteeing precision.

### B. Improving Retrieval Precision
To push retrieval metrics ($mAP$) closer to production-grade:

1. **Triplet Loss Fine-Tuning with Hard Negative Mining**:
   * Fine-tune the SigLIP backbone using triplet loss:
     * *Anchor*: "A red tie and white shirt in an office"
     * *Positive*: Image of red tie + white shirt in an office.
     * *Hard Negative*: Image of white tie + red shirt in an office (adversarial).
   * This forces the vector space to explicitly separate compositional swaps.
2. **Late Interaction Reranking (Cross-Encoder)**:
   * Use a two-stage retrieval pipeline:
     * *Stage 1*: Retrieve top 50 candidates using our fast Global-Local FAISS index.
     * *Stage 2*: Re-rank the top 50 candidates using a heavy multimodal **Cross-Encoder** (like BLIP-2 or PaliGemma). Cross-encoders process the image and text together in self-attention layers, yielding much higher matching precision at the cost of slight latency.

---

## 5. Codebase Structure & Execution

The codebase is organized into modular files:
* **`src/download_dataset.py`**: Streams 700 images from Fashionpedia.
* **`src/label_dataset.py`**: Programmatically enriches the database with environment and colors using `gemini-3.1-flash-lite` concurrently (15 threads).
* **`src/embedders.py`**: Model wrappers for feature extraction.
* **`src/detector.py`**: Bounding box garment extractor using Conditional-DETR.
* **`src/vector_store.py`**: Decoupled FAISS indexer with NumPy fallback.
* **`src/approach_*.py`**: Code for each baseline and fusion pipeline.
* **`benchmark.py`**: Runs evaluation queries and computes Recall, mAP, MRR, and Latency.

### How to Run:
1. Download data: `python src/download_dataset.py`
2. Label data (set your `GEMINI_API_KEY` first): `python src/label_dataset.py`
3. Run the benchmark suite: `python benchmark.py`
