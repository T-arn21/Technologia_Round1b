
# 🧠 Task 1B – Persona-Driven Document Intelligence

> **Adobe "Connecting the Dots" Challenge — Round 1B**  
> *Theme: "Connect What Matters — For the User Who Matters"*

---

## 🚀 Challenge Overview & Mission

In **Round 1B**, the objective is to build an intelligent, on-device document intelligence system that acts as an expert document analyst. Given a diverse collection of documents (3–10 PDFs) along with a user **persona** and their specific **job-to-be-done**, the system must extract, rank, and refine the most relevant sections and subsections.

### 🎯 Key Requirements
- **Generic Generalization:** The pipeline must generalize across arbitrary domains (e.g., Academic Research Papers, Business & Financial Reports, Educational Textbooks, Travel Guides, Legal/News Documents).
- **Persona & Task Alignment:** Deeply align document sections with the specific expertise and intent defined in the query:
  $$\text{Query} = \text{Persona} + \text{Job-to-be-Done}$$
- **Hierarchical Output:**
  1. **Document-Level Metadata:** Track input documents, flattened persona/task strings, and ISO-8601 UTC execution timestamps.
  2. **Extracted Sections (Top 5 Stack-Ranked):** Identify the most relevant sections with document name, page number, clean title, and importance rank.
  3. **Subsection Analysis:** Produce focused, 2-sentence extractive summaries (`refined_text`) grounded entirely in the source text.

---

## 🏆 Scoring Criteria (100 Points Total)

| Criteria | Max Points | Description |
|---|:---:|---|
| **Section Relevance** | **60** | Precision and recall of selected sections matching the persona and job-to-be-done with accurate stack ranking. |
| **Sub-Section Relevance** | **40** | Quality, conciseness, and factual grounding of granular subsection extraction and refined text analysis. |

---

## 🏗️ Architecture & Pipeline

```
Raw PDFs (3-10 files) ──► YOLOv10 Layout Detection (150 DPI)
                                  │
                                  ▼
                         Native PyMuPDF Text ──(Fallback)──► EasyOCR
                                  │
                         145 Section Chunks (Title + Body)
                                  │
                                  ▼
               Dense Semantic Embeddings (all-MiniLM-L6-v2)
                                  │
                         Cosine Similarity Ranking
                                  │
                                  ▼
                         Top 20 Candidates
                                  │
                                  ▼
               NLI Contradiction Guard (nli-deberta-v3-xsmall)
                                  │
               Penalty: Score * (1 - Contradiction)²
                                  │
                                  ▼
                         Top 5 Final Sections
                                  │
                                  ▼
               TF-IDF Extractive Summarization (refined_text)
                                  │
                                  ▼
                     challenge1b_output.json
```

### 1. 🧾 Document Layout & Native Text Parsing
- **YOLOv10 Layout Detection:** Identifies `title`, `list`, and wide `text` bounding boxes across all document pages.
- **Fast Native Vector Text:** Extracts digital text directly from bounding box coordinates in `< 0.1ms` using PyMuPDF (`fitz`), eliminating OCR typos and CPU bottlenecks.
- **Lazy EasyOCR Fallback:** EasyOCR is dynamically initialized only for scanned image regions without digital text layers.
- **Section Slicing:** Clips body text between consecutive detected headings across pages.

### 2. 🧠 Semantic Representation & Retrieval
- **Embedding Model:** `all-MiniLM-L6-v2` (Sentence Transformers, ~90MB, 22M parameters) computes dense 384-dimensional vector representations.
- **Query Formulation:** Rich composite query combining persona role and task description.
- **Fast Similarity Search:** Vectorized cosine similarity ranks all extracted sections across the document collection.

### 3. 🛡️ NLI Contradiction Filtering
- **Cross-Encoder Model:** Distilled `nli-deberta-v3-xsmall` (~280MB) evaluates pairwise premise-hypothesis entailment offline.
- **Targeted Candidate Pool:** Evaluates the **Top 20** semantic candidates (truncated to 200 words) to eliminate quadratic cross-attention overhead.
- **Contradiction Penalty:**
  $$\text{Final Score} = \text{Semantic Score} \times (1 - \text{Contradiction Score})^2$$

### 4. 🔍 Subsection Analysis & Grounded Refinement
- **TF-IDF Extractive Summarizer:** Scores sentences in each top section to extract a factual, 2-sentence summary.
- **Zero Hallucination:** Directly grounded in source PDF sentences.

---

## 🧪 Output Format Specification

```json
{
  "metadata": {
    "input_documents": [
      "South of France - Cities.pdf",
      "South of France - Cuisine.pdf",
      "South of France - History.pdf",
      "South of France - Restaurants and Hotels.pdf",
      "South of France - Things to Do.pdf",
      "South of France - Tips and Tricks.pdf",
      "South of France - Traditions and Culture.pdf"
    ],
    "persona": "Travel Planner",
    "job_to_be_done": "Plan a trip of 4 days for a group of 10 college friends.",
    "processing_timestamp": "2026-09-03T13:35:44.577004Z"
  },
  "extracted_sections": [
    {
      "document": "South of France - Tips and Tricks.pdf",
      "page_number": 8,
      "section_title": "Tips and Tricks for Packing",
      "importance_rank": 1
    },
    {
      "document": "South of France - Tips and Tricks.pdf",
      "page_number": 1,
      "section_title": "The Ultimate South of France Travel Companion: Your Comprehensive Guide to Packing, Planning, and Exploring",
      "importance_rank": 2
    }
  ],
  "subsection_analysis": [
    {
      "document": "South of France - Tips and Tricks.pdf",
      "page_number": 8,
      "refined_text": "Wear Bulky Items: Wear bulky items like coats or boots during travel to save suitcase space. Additional Tips: Pack a small travel umbrella, a reusable shopping bag, and a portable phone charger."
    }
  ]
}
```

---

## 🛠️ How to Build and Run

### 📦 Pre-setup

Before building the image:
- Unzip `models.zip` into the root directory so `models/` contains `all-MiniLM-L6-v2`, `nli-deberta-v3-xsmall`, and the YOLOv10 weights.

### 🐳 Build Docker Image

```bash
docker build --platform linux/amd64 -t mysolutionname:latest .
```

### ▶️ Run the Container

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  mysolutionname:latest
```

The container runs `miniLM_NLI.py` on `/app/input`. That folder contains input collections with `challenge1b_input.json` and a `PDFs/` (or `Pdfs/`) directory of PDFs. Output is written directly to `/app/output/challenge1b_output.json` (or `{collection_name}.json` for multiple collections).

---

## ⚙️ Constraints & Measured Benchmark Compliance

| Constraint | Requirement | Branch Status (`perf/fast-extraction`) | Notes |
|---|:---:|:---:|---|
| **Processing Time** | $\le 60\text{s}$ (3–5 PDFs) | ✅ **~45–48 s (5 PDFs)** / **66.81 s (7 dense PDFs)** | 11.4× faster than baseline (762s) |
| **Model Size** | $\le 1000\text{ MB}$ | ✅ **~500 MB total** | MiniLM (90MB) + DeBERTa (280MB) + YOLO (40MB) + EasyOCR (91MB) |
| **Compute** | CPU Only (amd64) | ✅ **100% CPU Compliant** | ProcessPoolExecutor + PyTorch CPU inference mode |
| **Network** | Offline Execution | ✅ **100% Offline** | `HF_HUB_OFFLINE=1`, no remote API calls |

---

## 🧩 Libraries and Models Used

| Tool / Library | Model / Version | Purpose |
|---|---|---|
| **DocLayout-YOLO** | YOLOv10 (~40MB) | High-speed document bounding box layout segmentation |
| **PyMuPDF (`fitz`)** | v1.26.x | Native digital PDF text extraction & bounding box clipping |
| **EasyOCR** | CRAFT + Latin (~91MB) | Fallback OCR for raster/scanned image headings |
| **Sentence-Transformers** | `all-MiniLM-L6-v2` (~90MB) | 384-dimensional dense semantic text embedding |
| **Cross-Encoder** | `nli-deberta-v3-xsmall` (~280MB) | Offline NLI contradiction verification & reranking |
| **sentencepiece, protobuf** | Tokenization & serialization | Required offline tokenizers for transformers |
| **scikit-learn** | TF-IDF Vectorizer | Unsupervised extractive summarization for `refined_text` |
| **concurrent.futures** | `ProcessPoolExecutor` | Multi-core parallel PDF extraction |
