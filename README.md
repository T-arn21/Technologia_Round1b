
# 🧠 Task 1B – Persona-Driven Document Intelligence

## 🚀 Challenge Overview

In **Round 1B** of the Adobe "Connecting the Dots" Challenge, our goal was to build an intelligent PDF analysis system that extracts and ranks **most relevant sections** from a collection of documents — based on a given **persona** and a **job-to-be-done**.

The system must:
- Read persona and job-to-be-done from the collection input.
- Semantically analyze documents against the job-to-be-done.
- Extract and refine relevant sections.
- Rank outputs by contextual alignment.

---

## 🏗️ Approach Summary

We extended our work from **Task 1A** by utilizing structured layout and OCR extraction. Then we introduced **semantic understanding**, **retrieval ranking**, and **NLI-based validation**. We also **extended our dataset** on a different field to test more results.

### 1. 🧾 Document Parsing

We reused our [Task 1A](https://github.com/T-arn21/Technologia_Round1a) layout extractor:
- **YOLOv10**: For bounding box detection of title, list, and wide text regions.
- **EasyOCR**: For English text extraction from the identified boxes (used as section titles).
- Section body text is taken from native PDF text (PyMuPDF) clipped between consecutive detected regions.

### 2. 🧠 Semantic Understanding

To bridge user intent with document content:
- We used **MiniLM** (`all-MiniLM-L6-v2`) from Sentence Transformers to embed:
  - Extracted section texts (title + body)
  - The job-to-be-done task as the **query**

- For each section, we computed **cosine similarity** with the query.
- Top 50 high-similarity candidates were shortlisted for further analysis.
- The top 5 NLI-reranked sections are written to the output.

### 3. 🔍 Refined Text

- For each of the top-ranked sections, we generate `refined_text` with a TF-IDF **extractive summary** (2 sentences from the section body).
- If the body is too short to summarize, the cleaned section title is used instead.

### 4. ❗ Contradiction Filtering (NLI)

To ensure high alignment:
- We employed a distilled **Natural Language Inference (NLI)** model (`nli-deberta-v3-xsmall`) offline.
- Initially, we applied contradiction checks to **all** candidates, but it increased latency.
- Final strategy: Apply NLI only on **Top 50** semantic matches.
  - Contradiction score is used as a penalty: `final_score = semantic_score * (1 - contradiction_score)²`.
  - Low-contradiction sections keep a higher rank.

This significantly reduced runtime while improving relevance precision.

---

## 🧪 Output Format

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "PhD Researcher in Computational Biology",
    "job_to_be_done": "Prepare a literature review on graph neural networks",
    "processing_timestamp": "2026-09-03T12:37:55.491385Z"
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
      "page_number": 5,
      "refined_text": "We compare performance of GNNs on benchmark datasets like Tox21..."
    }
  ]
}
```

---

## 🛠️ How to Build and Run

### 📦 Pre-setup

Before building the image:

- **Unzip** the provided `models.zip` into the root directory.
- Ensure the `models/` folder contains `all-MiniLM-L6-v2`, `nli-deberta-v3-xsmall`, and the YOLOv10 weights.

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

## 🧩 Libraries and Models Used

| Tool/Library         | Purpose                             |
|----------------------|-------------------------------------|
| YOLOv10              | Document layout detection           |
| EasyOCR              | OCR for section titles (English)    |
| PyMuPDF              | Native PDF text for section bodies  |
| Sentence Transformers| Semantic embeddings (MiniLM)        |
| Sentence Transformers CrossEncoder | NLI contradiction checking (`nli-deberta-v3-xsmall`) |
| sentencepiece, protobuf | Tokenization & serialization for transformers |
| scikit-learn         | TF-IDF extractive summaries         |
| NumPy, Pandas        | Data manipulation                   |
| concurrent.futures   | Process-based parallel PDF extraction |

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
- Parallel processing via `ProcessPoolExecutor` accelerated OCR/layout extraction (embeddings and NLI run in the main process).
- Focused on **generic pipeline** suitable for research, education, and business contexts.

---
