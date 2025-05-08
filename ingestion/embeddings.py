import os
import time
import random
import logging
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

# Validate required keys
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables.")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=OPENAI_EMBED_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
        self.max_retries = 5
        self.retry_base_delay = 30  # seconds
        self.request_interval = 10  # seconds between calls
        self.consecutive_failures = 0
        logger.info(f"🧠 EmbeddingService initialized with model: {OPENAI_EMBED_MODEL}")

    @retry(wait=wait_random_exponential(min=1, max=15), stop=stop_after_attempt(5))
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        time.sleep(self.request_interval)
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=15), stop=stop_after_attempt(5))
    def get_query_embedding(self, query: str) -> List[float]:
        time.sleep(self.request_interval)
        try:
            return self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"❌ Query embedding failed: {e}")
            raise

    def embed_documents(self, documents: List[Union[Dict[str, Any], Any]]) -> List[Union[Dict[str, Any], Any]]:
        total = len(documents)
        logger.info(f"🚀 Starting embedding for {total} document(s)")
        processed = []

        for idx, doc in enumerate(tqdm(documents, desc="🔗 Embedding documents")):
            success, retries = False, 0

            while not success and retries < self.max_retries:
                try:
                    content = doc.page_content if hasattr(doc, "page_content") else doc["text"]
                    embedding = self.get_embeddings([content])[0]

                    if hasattr(doc, "metadata"):
                        doc.metadata["embedding"] = embedding
                    else:
                        doc["embedding"] = embedding

                    processed.append(doc)
                    success = True
                    self.consecutive_failures = 0
                    logger.info(f"✅ Embedded doc {idx + 1}/{total}")
                except Exception as e:
                    retries += 1
                    self.consecutive_failures += 1

                    backoff = self.retry_base_delay * (2 ** (retries - 1)) * (1.5 ** self.consecutive_failures)
                    jitter = random.uniform(0.8, 1.2)
                    wait = min(backoff * jitter, 1800)

                    logger.warning(f"⚠️ Retry {retries}/{self.max_retries} after error: {e}")
                    logger.warning(f"⏳ Waiting {wait:.2f}s before retrying...")
                    time.sleep(wait)

            if not success:
                logger.error(f"❌ Skipped doc {idx + 1}/{total} after {retries} failed attempts.")

        logger.info(f"🏁 Embedding complete: {len(processed)} / {total} documents embedded.")
        return processed
