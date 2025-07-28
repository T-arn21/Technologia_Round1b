from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import easyocr
from doclayout_yolo.models.yolov10.model import YOLOv10
import sys
import os
import json
import torch
import re
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import time
from datetime import datetime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import cv2

# --- Add project root to sys.path for module imports ---
# Note: You may need to adjust this path based on your exact folder structure.
# This assumes the script is in a subfolder like '1b' and the project root is two levels up.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# --- ML/DL Imports ---

# --- Constants ---
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
YOLO_MODEL_PATH = os.path.join(
    MODELS_DIR, "doclayout_yolo_docstructbench_imgsz1024.pt")
SBERT_MODEL_PATH = os.path.join(MODELS_DIR, 'all-MiniLM-L6-v2')
NLI_MODEL_PATH = os.path.join(MODELS_DIR, 'nli-deberta-v3-xsmall')
OUTPUT_DIR_BASE = os.path.join(PROJECT_ROOT, "1b_outputs4")

# --- Global variables for worker processes ---
# These will be initialized once per worker process to avoid serialization errors.
worker_yolo_model = None
worker_ocr_reader = None

# --- Main Functions ---


def clean_text(text):
    """Cleans text by removing bullet points, extra newlines, and whitespace."""
    text = re.sub(r'[\n\r]+', ' ', text)
    text = re.sub(r'[•*]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_sections_from_pdf(pdf_path):
    """Worker function: Extracts sections from a single PDF. Models are loaded here."""
    global worker_yolo_model, worker_ocr_reader

    # Lazy Initialization: Load models if they haven't been loaded in this specific process yet
    if worker_yolo_model is None:
        # print(f"Process {os.getpid()}: Initializing YOLOv10 model...")
        worker_yolo_model = YOLOv10(YOLO_MODEL_PATH)
    if worker_ocr_reader is None:
        # print(f"Process {os.getpid()}: Initializing EasyOCR reader...")
        worker_ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    doc = fitz.open(pdf_path)
    all_headings = []
    class_names = worker_yolo_model.names

    for i, page in enumerate(doc):
        page_num = i + 1
        pix = page.get_pixmap(dpi=200)
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
                cropped_img = img_np[y1:y2, x1:x2]
                if cropped_img.size > 0:
                    ocr_result = worker_ocr_reader.readtext(
                        cropped_img, detail=0, paragraph=True)
                    if ocr_result and len(" ".join(ocr_result).split()) > 1:
                        text = " ".join(ocr_result).strip()
                        page_width, page_height = page.rect.width, page.rect.height
                        abs_bbox = ((x1/pix.width)*page_width, (y1/pix.height)*page_height,
                                    (x2/pix.width)*page_width, (y2/pix.height)*page_height)
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


def run_persona_analysis(collection_path):
    """Main pipeline with multiprocessing for PDF extraction."""
    start_time = time.perf_counter()

    input_json_path = os.path.join(collection_path, "challenge1b_input.json")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    persona, job_to_be_done = input_data.get(
        'persona', {}), input_data.get('job_to_be_done', {})
    job_text = job_to_be_done if isinstance(
        job_to_be_done, str) else job_to_be_done.get('task', '')

    print("🧠 Loading SBERT and NLI models in main process...")
    sbert_model = SentenceTransformer(SBERT_MODEL_PATH)
    nli_model = CrossEncoder(NLI_MODEL_PATH)
    print("✅ Main models loaded.")

    all_sections = []
    pdf_folder_path = os.path.join(collection_path, "Pdfs")
    pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(
        pdf_folder_path) if f.lower().endswith('.pdf')]

    # print(f"🚀 Starting parallel PDF processing for {len(pdf_files)} files...")
    # Using 'max_workers=2' as it was found to be optimal for a 16GB RAM system.
    # Adjust this value based on your system's available CPU cores and RAM.
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit the worker function without the model arguments
        future_to_pdf = {executor.submit(
            extract_sections_from_pdf, pdf): pdf for pdf in pdf_files}
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                if results := future.result():
                    all_sections.extend(results)
                    # print(f"✅ Finished processing: {os.path.basename(pdf_path)}")
            except Exception:
                print(
                    f"❌ Critical error processing {os.path.basename(pdf_path)}:")
                traceback.print_exc()

    if not all_sections:
        print("\n❌ No sections could be extracted. Cannot continue.")
        return

    # print(f"\n🔍 Stage 1: Calculating semantic scores for {len(all_sections)} sections...")
    query_embedding = sbert_model.encode(job_text, convert_to_tensor=True)
    section_embeddings = sbert_model.encode(
        [sec['full_text'] for sec in all_sections])
    semantic_scores = util.cos_sim(query_embedding, section_embeddings)[0]

    for i, section in enumerate(all_sections):
        section['semantic_score'] = semantic_scores[i].item()

    pre_ranked_sections = sorted(
        all_sections, key=lambda x: x['semantic_score'], reverse=True)

    nli_candidates = pre_ranked_sections[:50]

    # print(f"\n🧐 Stage 2: Applying NLI contradiction guard to top {len(nli_candidates)} candidates...")
    if nli_candidates:
        nli_pairs = [[job_text, section['full_text']]
                     for section in nli_candidates]
        # New Code
        nli_scores = nli_model.predict(
            nli_pairs, activation_fn=torch.nn.Softmax(dim=-1))

        for i, section in enumerate(nli_candidates):
            contradiction_score = nli_scores[i][0]
            original_score = section['semantic_score']
            final_score = original_score * ((1 - contradiction_score) ** 2)
            section['final_score'] = final_score
            section['contradiction_score'] = contradiction_score

    final_ranked_sections = sorted(
        nli_candidates, key=lambda x: x.get('final_score', 0), reverse=True)

    output_json = {"metadata": {"input_documents": [os.path.basename(p) for p in pdf_files], "persona": persona, "job_to_be_done": job_to_be_done, "processing_timestamp": datetime.utcnow(
    ).isoformat() + "Z"}, "extracted_sections": [], "sub_section_analysis": []}

    # print("\n📝 Generating final output...")
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

        output_json["sub_section_analysis"].append(
            {"document": section['doc_name'], "page_number": section['page_number'], "refined_text": cleaned_summary})

    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    output_filename = os.path.join(
        OUTPUT_DIR_BASE, f"{os.path.basename(collection_path)}_output.json")
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
    run_persona_analysis(args.collection_path)
