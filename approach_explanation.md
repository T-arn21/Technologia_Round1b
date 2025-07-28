# 🧠 Technologia Round 1B – Two-Stage Cognitive Pipeline for Document Intelligence

Our goal was to build more than just an information extractor. We envisioned a cognitive assistant — a system that mimics how a human expert would intelligently sift through, prioritize, and verify information. This led us to design a robust, multi-stage pipeline focused on accuracy, reliability, and performance within strict runtime constraints.

---

## 🔍 Stage 1: Layout-Aware Content Extraction

Building on our successful Round 1A foundation, the system begins by deconstructing PDFs to understand both content and context.

- We use a powerful **YOLOv10 document layout model** to identify structural elements:
  - Titles
  - Text blocks
  - Lists

- This **layout-aware approach** enables us to group text into meaningful, semantically coherent sections (e.g., a heading with its paragraph) rather than treating the document as a flat wall of text.

- Then, **EasyOCR** extracts the textual content from each detected block, ensuring clean and structured inputs for the ranking stage.

---

## 🧠 Stage 2: Hybrid Ranking with a "Contradiction Guard"

This is the core of our pipeline's intelligence. We realized that *true relevance* is a combination of:
- Semantic similarity
- Logical consistency

### ✅ Semantic Relevance Scoring
- We use **SentenceTransformer (all-MiniLM-L6-v2)** for fast and effective semantic matching.
- The persona's job description and each extracted document section are encoded into high-dimensional vectors.
- We compute **cosine similarity** between these vectors to get an initial ranking based on thematic alignment.

### 🛡️ NLI-Powered Reliability Check ("Contradiction Guard")
- Semantic similarity alone isn’t enough — some sections might be topically aligned but **contradict** the persona's goals.
- To prevent this, we introduced a **Natural Language Inference (NLI) step** using **nli-deberta-v3-xsmall**.
- The top 50 semantically matched sections are passed through this NLI model.
- Sections that contradict the persona’s objective are heavily penalized — ensuring the final output is not just relevant, but also **logically sound**.

---

## ⚙️ Engineered for Performance

We knew this deep processing had to complete in under **60 seconds on CPU**, so we optimized for speed and efficiency:

- We use Python’s **`ProcessPoolExecutor`** to parallelize the most intensive parts of the pipeline — processing multiple PDFs at once.
- Each worker lazily initializes its own instance of YOLO and EasyOCR models to avoid unnecessary memory load and latency.
- This architecture scales well with the number of documents and stays within tight compute limits.

---

## 🎯 Final Thoughts

Our system does more than match keywords — it **understands** content, **ranks** intelligently, and **filters contradictions**, just like a human expert would.

By combining semantic search with a contradiction filter and designing for high performance, we’ve built a document intelligence system that connects users with what truly matters.
