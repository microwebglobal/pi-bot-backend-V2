import os
import uuid
import logging
import re
import langdetect
import redis
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from ingestion.vectorstore import VectorStore

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Redis setup
cache = redis.Redis.from_url(REDIS_URL)

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Vector store and LLM setup
vector_store = VectorStore()
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)

# Load strict system prompt
try:
    with open("prompts/system_prompt.txt", "r") as f:
        SYSTEM_PROMPT = f.read()
        logger.info("✅ Loaded system prompt from file")
except Exception as e:
    SYSTEM_PROMPT = "You are a helpful assistant. Only respond based on the documents provided."
    logger.warning(f"⚠️ Failed to load custom system prompt: {e}")

# Prompt Template
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

# Sanitize input
def sanitize_question(question: str) -> str:
    question = re.sub(r"[^\w\s.,?!]", "", question)
    return question.strip()

# Detect and annotate language if needed
def auto_translate_if_needed(question: str) -> str:
    try:
        lang = langdetect.detect(question)
        if lang != "en":
            logger.info(f"Translating from {lang} to English (not implemented, only annotated).")
            return f"[Original Language: {lang}] {question}"
    except:
        logger.warning("Language detection failed.")
    return question

# Retrieves top-k relevant documents using hybrid logic
def retrieve_chunks(question: str, k: int = 20) -> List[Document]:
    results = vector_store.hybrid_search(question, k=k)
    logger.info(f"[RAG] Retrieved {len(results)} document chunks")
    return [doc for doc, _ in results]

# Full RAG pipeline with caching and fallback
def run_rag_pipeline(question: str) -> str:
    sanitized = sanitize_question(question)
    translated = auto_translate_if_needed(sanitized)
    query_id = str(uuid.uuid4())
    logger.info(f"[RAG] Query ID: {query_id} | Question: {translated}")

    # Caching
    cache_key = f"rag_cache:{hash(translated)}"
    cached = cache.get(cache_key)
    if cached:
        logger.info("[RAG] Cache hit")
        return cached.decode("utf-8")

    # Document retrieval
    docs = retrieve_chunks(translated)
    if not docs:
        logger.warning("[RAG] No documents found for query.")
        return "The documents do not contain any relevant information to answer this question."

    context = "\n\n".join(doc.page_content for doc in docs)

    # Prompt construction
    prompt = RAG_PROMPT.invoke({"context": context, "question": translated})

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        logger.error(f"[RAG] GPT-4o failed: {e} — falling back to GPT-3.5")

        fallback_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
        fallback_prompt = RAG_PROMPT.invoke({"context": context, "question": translated})
        response = fallback_llm.invoke(fallback_prompt)
        answer = response.content.strip()

    # Cache result
    cache.setex(cache_key, 86400, answer)  # 1 day
    logger.info(f"[RAG] Query ID: {query_id} completed.")
    return answer
