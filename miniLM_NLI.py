from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import easyocr
from doclayout_yolo.models.yolov10.model import YOLOv10
import sys
import os
import json
import torch
import re
import numpy as np
import pymupdf as fitz

import pandas as pd
import time
from datetime import datetime, timezone

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


def _first_existing_dir(candidates):
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


MODELS_DIR = _first_existing_dir([
    os.path.join(SCRIPT_DIR, "models"),
    "/models",
])
YOLO_MODEL_PATH = os.path.join(
    MODELS_DIR, "doclayout_yolo_docstructbench_imgsz1024.pt")
SBERT_MODEL_PATH = os.path.join(MODELS_DIR, "all-MiniLM-L6-v2")
NLI_MODEL_PATH = os.path.join(MODELS_DIR, "nli-deberta-v3-xsmall")
OUTPUT_DIR_BASE = "/app/output" if os.path.isdir("/app/output") else os.path.join(
    SCRIPT_DIR, "output")

worker_yolo_model = None
worker_ocr_reader = None


def flatten_persona_or_job(value, nested_key):
    if isinstance(value, dict):
        if nested_key in value:
            return value.get(nested_key, "") or ""
        return next(iter(value.values()), "") if value else ""
    if isinstance(value, str):
        return value
    return ""


def find_collections(root_path):
    json_name = "challenge1b_input.json"
    if os.path.isfile(os.path.join(root_path, json_name)):
        return [root_path]
    collections = []
    for name in sorted(os.listdir(root_path)):
        child = os.path.join(root_path, name)
        if os.path.isdir(child) and os.path.isfile(os.path.join(child, json_name)):
            collections.append(child)
    return collections


def find_pdf_files(collection_path, input_data):
    for folder in ("PDFs", "Pdfs", "pdfs"):
        path = os.path.join(collection_path, folder)
        if os.path.isdir(path):
            return sorted(
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.lower().endswith(".pdf")
            )
    files = []
    for doc in input_data.get("documents") or []:
        filename = doc.get("filename") if isinstance(doc, dict) else doc
        if not filename:
            continue
        candidate = os.path.join(collection_path, filename)
        if os.path.isfile(candidate):
            files.append(candidate)
    if files:
        return files
    return sorted(
        os.path.join(collection_path, f)
        for f in os.listdir(collection_path)
        if f.lower().endswith(".pdf")
    )


def clean_text(text):
    """Cleans text by removing bullet points, extra newlines, and whitespace."""
    text = re.sub(r'[\n\r]+', ' ', text)
    text = re.sub(r'[•*]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_sections_from_pdf(pdf_path):
    """Worker function: Extracts sections from a single PDF. Models are loaded here."""
    global worker_yolo_model, worker_ocr_reader

    if worker_yolo_model is None:
        worker_yolo_model = YOLOv10(YOLO_MODEL_PATH)

    doc = fitz.open(pdf_path)
    all_headings = []
    class_names = worker_yolo_model.names

    for i, page in enumerate(doc):
        page_num = i + 1
        page_width, page_height = page.rect.width, page.rect.height
        pix = page.get_pixmap(dpi=150)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        if img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        elif img_np.shape[2] == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

        results = worker_yolo_model.predict(img_np, conf=0.3, verbose=False)
        for box in results[0].boxes:
            label = class_names[int(box.cls)]
            if label in ["title", "list"] or (label == "text" and (box.xyxy[0][2] - box.xyxy[0][0]) > 100):
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                abs_bbox = ((x1/pix.width)*page_width, (y1/pix.height)*page_height,
                            (x2/pix.width)*page_width, (y2/pix.height)*page_height)
                
                # Fast path: extract native digital text from bounding box
                rect = fitz.Rect(abs_bbox)
                text = page.get_text("text", clip=rect).strip() if rect.is_valid else ""
                text = clean_text(text)

                # Fallback to EasyOCR if native text is empty or missing
                if not text or len(text.split()) < 1:
                    cropped_img = img_np[y1:y2, x1:x2]
                    if cropped_img.size > 0:
                        if worker_ocr_reader is None:
                            worker_ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                        ocr_result = worker_ocr_reader.readtext(
                            cropped_img, detail=0, paragraph=True)
                        if ocr_result:
                            text = clean_text(" ".join(ocr_result))

                if text and len(text.split()) >= 1:
                    all_headings.append(
                        {"text": text, "page": page_num, "y_coord": abs_bbox[1], "bbox": abs_bbox})

    if not all_headings:
        doc.close()
        return []

    df = pd.DataFrame(all_headings).sort_values(
        by=['page', 'y_coord']).reset_index(drop=True)
    sections = []
    for i in range(len(df)):
        current_heading = df.iloc[i]
        page_num = int(current_heading['page'])
        page = doc[page_num - 1]
        start_rect = fitz.Rect(current_heading['bbox'])
        next_heading_on_page = df[(df['page'] == page_num) & (df.index > i)]
        end_y = next_heading_on_page.iloc[0]['bbox'][1] - \
            5 if not next_heading_on_page.empty else page.rect.height
        clip_rect = fitz.Rect(start_rect.x0, start_rect.y0,
                              page.rect.width, end_y)
        section_text = page.get_text(
            "text", clip=clip_rect).strip() if clip_rect.is_valid else ""
        sections.append({"doc_name": os.path.basename(pdf_path), "page_number": page_num,
                        "section_title": current_heading['text'], "full_text": f"{current_heading['text']}\n{section_text}"})

    doc.close()
    return sections


def extractive_summary(text, num_sentences=2):
    """Generates a concise, extractive summary from body text."""
    if not text or len(text.split()) < 15:
        return text
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
    if len(sentences) <= num_sentences:
        return '. '.join(sentences) + '.' if sentences else ''
    try:
        vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_indices.sort()
        summary = '. '.join([sentences[i] for i in top_indices])
        return summary + '.' if not summary.endswith('.') else summary
    except ValueError:
        return text


def run_persona_analysis(collection_path, output_filename):
    """Main pipeline with multiprocessing for PDF extraction."""
    start_time = time.perf_counter()

    input_json_path = os.path.join(collection_path, "challenge1b_input.json")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    persona, job_to_be_done = input_data.get(
        'persona', {}), input_data.get('job_to_be_done', {})
    persona_str = flatten_persona_or_job(persona, "role")
    job_str = flatten_persona_or_job(job_to_be_done, "task")
    query_text = f"{persona_str}. {job_str}".strip(" .")

    pdf_files = find_pdf_files(collection_path, input_data)
    if not pdf_files:
        print("\n❌ No PDFs found. Cannot continue.")
        return

    print(f"📄 Extracting layout & text from {len(pdf_files)} PDF(s) in parallel...")
    num_workers = min(4, os.cpu_count() or 1)
    all_sections = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_pdf = {executor.submit(
            extract_sections_from_pdf, pdf): pdf for pdf in pdf_files}
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                if results := future.result():
                    all_sections.extend(results)
                    print(f"   ✓ Extracted {len(results)} sections from {os.path.basename(pdf_path)}")
            except Exception:
                print(
                    f"❌ Critical error processing {os.path.basename(pdf_path)}:")
                traceback.print_exc()

    if not all_sections:
        print("\n❌ No sections could be extracted. Cannot continue.")
        return

    print(f"📊 Total sections extracted across collection: {len(all_sections)}")
    print("🧠 Loading SBERT and NLI models for ranking...")
    sbert_model = SentenceTransformer(SBERT_MODEL_PATH)
    nli_model = CrossEncoder(NLI_MODEL_PATH)
    print("✅ Models loaded successfully.")

    with torch.inference_mode():
        print("🔍 Computing semantic similarity embeddings...")
        query_embedding = sbert_model.encode(
            query_text, convert_to_tensor=True, show_progress_bar=False)
        section_embeddings = sbert_model.encode(
            [sec['full_text'] for sec in all_sections], batch_size=32, show_progress_bar=False)
        semantic_scores = util.cos_sim(query_embedding, section_embeddings)[0]

        for i, section in enumerate(all_sections):
            section['semantic_score'] = semantic_scores[i].item()

        pre_ranked_sections = sorted(
            all_sections, key=lambda x: x['semantic_score'], reverse=True)

        nli_candidates = pre_ranked_sections[:20]

        if nli_candidates:
            print(f"🛡️ Running NLI contradiction guard on top {len(nli_candidates)} candidates...")
            
            def _truncate_text(text, max_words=200):
                words = text.split()
                return text if len(words) <= max_words else " ".join(words[:max_words])

            nli_pairs = [[query_text, _truncate_text(section['full_text'], 200)]
                         for section in nli_candidates]
            nli_scores = nli_model.predict(
                nli_pairs, batch_size=20, activation_fn=torch.nn.Softmax(dim=-1), show_progress_bar=False)

            for i, section in enumerate(nli_candidates):
                contradiction_score = float(nli_scores[i][0])
                original_score = section['semantic_score']
                final_score = original_score * ((1 - contradiction_score) ** 2)
                section['final_score'] = final_score
                section['contradiction_score'] = contradiction_score

    final_ranked_sections = sorted(
        nli_candidates, key=lambda x: x.get('final_score', 0), reverse=True)

    output_json = {
        "metadata": {
            "input_documents": [os.path.basename(p) for p in pdf_files],
            "persona": persona_str,
            "job_to_be_done": job_str,
            "processing_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        "extracted_sections": [],
        "subsection_analysis": [],
    }

    top_n = min(5, len(final_ranked_sections))
    for i, section in enumerate(final_ranked_sections[:top_n]):
        output_json["extracted_sections"].append(
            {"document": section['doc_name'], "page_number": section['page_number'], "section_title": section['section_title'], "importance_rank": i + 1})

        body_text = section['full_text'].replace(
            section['section_title'], '', 1).strip()
        summary = extractive_summary(body_text)
        cleaned_summary = clean_text(summary)

        if not cleaned_summary:
            cleaned_summary = clean_text(section['section_title'])

        output_json["subsection_analysis"].append(
            {"document": section['doc_name'], "page_number": section['page_number'], "refined_text": cleaned_summary})

    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    total_time = time.perf_counter() - start_time
    print(f"\n🎉 Analysis complete for '{os.path.basename(collection_path)}'!")
    print(f"   Output saved to: {output_filename}")
    print(f"   ⏱️   Total wall-clock time: {total_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Persona Analysis with Multiprocessing")
    parser.add_argument("collection_path", type=str,
                        help="Path to the input collection folder")
    args = parser.parse_args()
    if not os.path.isdir(args.collection_path):
        sys.exit(f"❌ Error: Path not found '{args.collection_path}'")
    collections = find_collections(args.collection_path)
    if not collections:
        sys.exit(
            f"❌ Error: No challenge1b_input.json under '{args.collection_path}'")
    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    if len(collections) == 1:
        output_path = os.path.join(OUTPUT_DIR_BASE, "challenge1b_output.json")
        run_persona_analysis(collections[0], output_path)
    else:
        for collection in collections:
            output_path = os.path.join(
                OUTPUT_DIR_BASE, f"{os.path.basename(collection)}.json")
            run_persona_analysis(collection, output_path)
