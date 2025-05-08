import os
import re
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
import spacy
from pdf2image import convert_from_path
from langdetect import detect, LangDetectException
from dotenv import load_dotenv
from typing import List, Dict
import logging
import json

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Tesseract path config
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH", "C:\Program Files\Tesseract-OCR\tesseract.exe")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    raise RuntimeError("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")

# --- Utility functions ---

def extract_text_fitz(page) -> str:
    return page.get_text("text").strip()

def extract_text_plumber(pdf_path: str, page_num: int) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num < len(pdf.pages):
                return pdf.pages[page_num].extract_text() or ""
    except Exception as e:
        logging.error(f"❌ pdfplumber failed on page {page_num+1}: {e}")
    return ""

def extract_text_ocr(image) -> str:
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        logging.error(f"❌ OCR failed: {e}")
        return ""

def is_image_only(page) -> bool:
    return len(page.get_text("text").strip()) < 10

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("�", "").replace("\u00a0", " ")).strip()

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def extract_headings(text: str) -> List[str]:
    lines = text.splitlines()
    headings = []
    for line in lines:
        if line.strip().isupper() and 5 < len(line.strip()) < 100:
            headings.append(f"[{line.strip()}]")
        elif re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$', line.strip()):
            headings.append(f"[{line.strip()}]")
    return sorted(set(headings))

def extract_entities(text: str) -> List[str]:
    doc = nlp(text)
    entities = [f"{ent.label_}: {ent.text}" for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
    return sorted(set(entities))

def infer_year(text: str, filename: str) -> str:
    match = re.search(r"(20\d{2})", text)
    if match:
        return match.group(1)
    match = re.search(r"(20\d{2})", filename)
    if match:
        return match.group(1)
    return "unknown"

def infer_document_type(text: str) -> str:
    lowered = text.lower()
    if "balance sheet" in lowered:
        return "Financial Statements"
    if "board of directors" in lowered:
        return "Governance"
    if "ceo" in lowered or "chairman" in lowered:
        return "Leadership"
    if "sustainability" in lowered:
        return "Sustainability"
    return "General"

# --- Main extraction logic ---

def extract_from_pdf(file_path: str, output_txt: bool = True) -> List[Dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"📁 File not found: {file_path}")

    doc = fitz.open(file_path)
    filename = os.path.basename(file_path)
    results = []

    logging.info(f"📄 Starting extraction: {file_path} | Pages: {len(doc)}")

    for i, page in enumerate(doc):
        logging.info(f"🔍 Page {i + 1}/{len(doc)}")

        raw_text = extract_text_fitz(page)

        if not raw_text.strip():
            raw_text = extract_text_plumber(file_path, i)

        ocr_used = False
        if not raw_text.strip() or is_image_only(page):
            logging.info("📸 Image page detected — running OCR")
            images = convert_from_path(file_path, first_page=i + 1, last_page=i + 1)
            if images:
                raw_text = extract_text_ocr(images[0])
                ocr_used = True

        cleaned = clean_text(raw_text)
        lang = detect_language(cleaned)
        year = infer_year(cleaned, filename)
        doc_type = infer_document_type(cleaned)
        headings = extract_headings(cleaned)
        entities = extract_entities(cleaned)

        results.append({
            "filename": filename,
            "page": i + 1,
            "text": cleaned,
            "language": lang,
            "year": year,
            "type": doc_type,
            "headings": headings,
            "entities": entities,
            "ocr_used": ocr_used
        })

    logging.info(f"✅ Extraction complete: {len(results)} pages processed.")

    if output_txt:
        txt_path = file_path.replace(".pdf", "_extracted.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for p in results:
                f.write(f"\n--- Page {p['page']} | {p['year']} | {p['type']} ---\n")
                for h in p["headings"]:
                    f.write(f"🔹 {h}\n")
                for e in p["entities"]:
                    f.write(f"🔸 {e}\n")
                f.write(f"\nText:\n{p['text']}\n\n{'-'*40}\n")
        logging.info(f"💾 Saved: {txt_path}")

    return results


# CLI test
if __name__ == "__main__":
    test_file = "resources/people-s-insurance-plc-ar2024.pdf"
    pages = extract_from_pdf(test_file)
    for p in pages:
        print(f"\n--- Page {p['page']} | {p['year']} | {p['type']} ---")
        for h in p["headings"]:
            print(f"🔹 {h}")
        for e in p["entities"]:
            print(f"🔸 {e}")
