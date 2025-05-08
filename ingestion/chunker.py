import os
import re
import tiktoken
import spacy
import logging
from typing import List, Dict
from langdetect import detect
from hashlib import md5
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from statistics import mean

# Load env vars
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set in the environment variables.")

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Tokenizer + NLP
tokenizer = tiktoken.get_encoding("cl100k_base")
nlp = spacy.load("en_core_web_sm")

# Deduplication memory
_seen_hashes = set()

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def break_into_adaptive_chunks(text: str, max_tokens: int = 400) -> List[str]:
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if sentence_tokens > max_tokens:
            chunks.extend(break_long_sentence(sentence, max_tokens))
            continue

        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk += " " + sentence
            current_tokens += sentence_tokens
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = sentence_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def break_long_sentence(sentence: str, max_tokens: int = 400) -> List[str]:
    words = sentence.split()
    sub_chunks = []
    current = []

    for word in words:
        current.append(word)
        if count_tokens(" ".join(current)) >= max_tokens:
            sub_chunks.append(" ".join(current))
            current = []

    if current:
        sub_chunks.append(" ".join(current))

    return sub_chunks

def clean_and_filter_chunk(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.replace("�", "").replace("\x00", "")).strip()
    if len(cleaned) < 20:
        return None
    try:
        if detect(cleaned) != "en":
            return None
    except Exception:
        return None

    symbol_ratio = sum(1 for c in cleaned if not c.isalnum() and not c.isspace()) / max(len(cleaned), 1)
    if symbol_ratio > 0.2:
        return None

    return cleaned

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def split_text_with_retry(text: str) -> List[str]:
    return break_into_adaptive_chunks(text)

def auto_detect_sections(text: str) -> List[tuple]:
    matches = re.findall(r'(\n[A-Z][A-Z\s\d,:\-&]{6,}\n)', text)
    if not matches:
        return [("Full Document", text, "")]

    sections = []
    start = 0
    for match in matches:
        heading_start = text.find(match, start)
        if heading_start == -1:
            continue
        if sections:
            prev_heading, prev_start = sections[-1]
            body = text[prev_start:heading_start].strip()
            sections[-1] = (prev_heading, body, "")
        sections.append((match.strip(), heading_start))
        start = heading_start + len(match)

    if sections:
        last_heading, last_start = sections[-1]
        sections[-1] = (last_heading, text[last_start:].strip(), "")

    return [(h, b, "") for h, b, _ in sections]

def _process_section(section_title: str, section_body: str, doc_name: str) -> List[Dict]:
    chunk_dicts = []
    try:
        chunks = split_text_with_retry(section_body)
    except Exception as e:
        logging.error(f"❌ Chunking failed for section '{section_title}': {e}")
        return chunk_dicts

    for chunk in chunks:
        cleaned = clean_and_filter_chunk(chunk)
        if not cleaned:
            continue

        chunk_hash = md5(cleaned.encode("utf-8")).hexdigest()
        if chunk_hash in _seen_hashes:
            continue
        _seen_hashes.add(chunk_hash)

        ents = [ent.text for ent in nlp(cleaned).ents if ent.label_ in {"ORG", "PERSON", "DATE", "GPE"}]

        chunk_dicts.append({
            "text": cleaned,
            "doc_name": doc_name,
            "section": section_title,
            "entities": ents
        })

    return chunk_dicts

def chunk_text(text: str, doc_name: str = "unknown") -> List[Dict]:
    logging.info(f"🧠 Chunking text from: {doc_name}")

    section_pattern = r'\[(.*?)\]\s*(.+?)(?=(\[\w.*?\])|\Z)'
    matches = re.findall(section_pattern, text, flags=re.DOTALL)

    if not matches:
        logging.warning("No bracketed section titles found — applying auto-detection.")
        matches = auto_detect_sections(text)

    chunk_list = []
    all_tokens = []
    lang_counts = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for section_title, section_body, _ in matches:
            section_title = section_title.strip()
            section_body = section_body.strip()
            if not section_body:
                continue
            futures.append(executor.submit(_process_section, section_title, section_body, doc_name))

        for future in futures:
            section_chunks = future.result()
            for ch in section_chunks:
                token_count = count_tokens(ch['text'])
                lang = detect(ch['text'])
                all_tokens.append(token_count)
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            chunk_list.extend(section_chunks)

    logging.info(f"✅ Chunking complete. Total chunks: {len(chunk_list)} | Avg tokens: {mean(all_tokens):.2f} | Languages: {lang_counts}")
    return chunk_list
