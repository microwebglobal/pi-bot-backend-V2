import os
import logging
from dotenv import load_dotenv
from ingestion.extractor import extract_from_pdf
from ingestion.chunker import chunk_text
from ingestion.vectorstore import VectorStore
from ingestion.embeddings import EmbeddingService

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize components
embedder = EmbeddingService()
vectorstore = VectorStore()

# Folder for OCR-based PDFs (use a separate one from regular documents)
OCR_PDF_FOLDER = "resources/ocr_docs"

def process_ocr_document(file_path: str):
    file_name = os.path.basename(file_path)
    logging.info(f"\n📄 OCR Processing: {file_name}")

    # Step 1: Extract text from scanned PDF (uses Tesseract internally if needed)
    extracted_pages = extract_from_pdf(file_path)
    full_text = ""

    for page in extracted_pages:
        section_tags = [f"[{h}]" for h in page.get("headings", [])] or ["[Full Page]"]
        for tag in section_tags:
            full_text += f"\n{tag}\n{page.get('text', '')}\n"

    # Step 2: Chunk into clean semantic blocks
    chunks = chunk_text(full_text, doc_name=file_name)

    # Step 3: Embed each chunk
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.get_embeddings(texts)

    for i in range(len(chunks)):
        chunks[i]["embedding"] = embeddings[i]

    # Step 4: Add to vector database
    vectorstore.add_chunks(chunks)
    logging.info(f"✅ OCR ingestion complete for: {file_name}")


def process_all_ocr_documents():
    if not os.path.exists(OCR_PDF_FOLDER):
        logging.warning(f"🚫 OCR folder not found: {OCR_PDF_FOLDER}")
        return

    for file in os.listdir(OCR_PDF_FOLDER):
        if file.lower().endswith(".pdf"):
            process_ocr_document(os.path.join(OCR_PDF_FOLDER, file))


if __name__ == "__main__":
    process_all_ocr_documents()
