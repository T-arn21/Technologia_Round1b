
# 🧠 Task 1B – Persona-Driven Document Intelligence

## 🚀 Challenge Overview

In **Round 1B** of the Adobe "Connecting the Dots" Challenge, our goal was to build an intelligent PDF analysis system that extracts and ranks **most relevant sections** from a collection of documents — based on a given **persona** and a **job-to-be-done**.

The system must:
- Understand the persona and task.
- Semantically analyze documents.
- Extract and refine relevant sections.
- Rank outputs by contextual alignment.

---

## 🏗️ Approach Summary

We extended our work from **Task 1A** by utilizing structured layout and OCR extraction. Then we introduced **semantic understanding**, **retrieval ranking**, and **NLI-based validation**.

### 1. 🧾 Document Parsing

We reused our Task 1A layout extractor:
- **YOLOv10**: For bounding box detection of text regions.
- **EasyOCR**: For multilingual text extraction from the identified boxes.
- Detected text was organized hierarchically using heading-level rules (H1/H2/H3).

### 2. 🧠 Semantic Understanding

To bridge user intent with document content:
- We used **MiniLM** (`all-MiniLM-L6-v2`) from Sentence Transformers to embed:
  - Extracted heading/paragraph texts
  - The concatenated persona + job description as the **query**

- For each section/subsection, we computed **cosine similarity** with the query.
- Top 50 high-similarity candidates were shortlisted for further analysis.

### 3. 🔍 Subtext Matching

- To improve fine-grained matching, we split longer sections into **subtexts** (3–5 sentences).
- Subtext embeddings were again matched with the query for deeper relevance checks.

### 4. ❗ Contradiction Filtering (NLI)

To ensure high alignment:
- We employed a distilled **Natural Language Inference (NLI)** model (`roberta-mnli`) offline.
- Initially, we applied contradiction checks to **all** candidates, but it increased latency.
- Final strategy: Apply NLI only on **Top 50** semantic matches.
  - If a contradiction is found → rank is penalized.
  - Entailments or neutral → higher rank retained.

This significantly reduced runtime while improving relevance precision.

---

## 🧪 Output Format

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "PhD Researcher in Computational Biology",
    "job_to_be_done": "Prepare a literature review on graph neural networks",
    "processed_at": "2025-07-28T14:05:23"
  },
  "extracted_sections": [
    {
      "document": "doc1.pdf",
      "page_number": 4,
      "section_title": "GNN Approaches for Molecule Property Prediction",
      "importance_rank": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "doc1.pdf",
      "refined_text": "We compare performance of GNNs on benchmark datasets like Tox21...",
      "page_number": 5
    }
  ]
}
```

---

## 🛠️ How to Build and Run

### 🐳 Build Docker Image

```bash
docker build --platform linux/amd64 -t mysolutionname:latest .
```

### ▶️ Run the Container

```bash
docker run --rm   -v $(pwd)/input:/app/input   -v $(pwd)/output:/app/output   --network none   mysolutionname:latest
```

The system will process all `.pdf` files in `/app/input` and generate corresponding `.json` outputs in `/app/output`.

---

## 🧩 Libraries and Models Used

| Tool/Library         | Purpose                             |
|----------------------|-------------------------------------|
| YOLOv10              | Document layout detection           |
| EasyOCR              | OCR text extraction (multilingual)  |
| Sentence Transformers| Semantic embeddings (MiniLM)        |
| HuggingFace Transformers | NLI Contradiction Checking (`roberta-mnli`) |
| NumPy, Pandas        | Data manipulation                   |
| concurrent.futures   | Thread-based parallel processing    |

---

## ⚙️ Constraints & Compliance

| Constraint            | Compliance Status |
|---------------------- |------------------|
| Processing Time ≤ 60s | ✅ (~35–45s for 5 PDFs) |
| Model Size ≤ 1GB      | ✅ (~750MB total) |
| CPU-only              | ✅ |
| Offline Execution     | ✅ (no internet dependency) |

---

## 💡 Design Decisions

- **Early Filtering** using MiniLM reduced overhead for NLI.
- Applying **NLI on Top 50** candidates preserved both quality and speed.
- Parallel processing via `ThreadPoolExecutor` accelerated OCR and embedding stages.
- Focused on **generic pipeline** suitable for research, education, and business contexts.

---

