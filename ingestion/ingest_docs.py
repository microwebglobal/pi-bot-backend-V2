import os
import logging
from dotenv import load_dotenv

from ingestion.extractor import extract_from_pdf
from ingestion.chunker import chunk_text
from ingestion.embeddings import EmbeddingService
from ingestion.vectorstore import VectorStore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
PDF_FOLDER = "resources/docs"

# Initialize services
embedder = EmbeddingService()
vectorstore = VectorStore()


def process_document(file_path: str):
    try:
        file_name = os.path.basename(file_path)
        logging.info(f"\n==============================\n📄 Processing: {file_name}\n==============================")

        # Step 1: Extract pages with text and headings
        pages = extract_from_pdf(file_path, output_txt=True)
        if not pages:
            logging.warning(f"⚠️ No pages extracted from {file_name}")
            return

        all_text = ""
        for page in pages:
            section_tags = [f"[{h}]" for h in page["headings"]] if page["headings"] else ["[Full Page]"]
            for tag in section_tags:
                all_text += f"\n{tag}\n{page['text']}\n"

        # Step 2: Chunk into digestible segments
        chunks = chunk_text(all_text, doc_name=file_name)
        if not chunks:
            logging.warning(f"⚠️ No valid chunks created for {file_name}")
            return

        # Step 3: Embed chunks
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.get_embeddings(texts)
        for i, embedding in enumerate(embeddings):
            chunks[i]["embedding"] = embedding

        # Step 4: Add to vector store
        vectorstore.add_chunks(chunks)
        logging.info(f"✅ {file_name} successfully processed and added to Pinecone index.")

    except Exception as e:
        logging.error(f"❌ Error processing {file_path}: {e}")


def process_all_documents():
    if not os.path.exists(PDF_FOLDER):
        logging.error(f"❌ PDF folder does not exist: {PDF_FOLDER}")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logging.warning(f"⚠️ No PDF files found in {PDF_FOLDER}")
        return

    for filename in pdf_files:
        full_path = os.path.join(PDF_FOLDER, filename)
        process_document(full_path)


if __name__ == "__main__":
    process_all_documents()
