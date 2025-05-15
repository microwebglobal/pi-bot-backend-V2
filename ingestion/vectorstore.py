import os
import re
import uuid
import logging
from typing import List, Tuple, Dict
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from rank_bm25 import BM25Okapi
import openai

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

if not all([PINECONE_API_KEY, PINECONE_INDEX_NAME, OPENAI_API_KEY]):
    raise EnvironmentError("Missing one or more required environment variables.")

openai.api_key = OPENAI_API_KEY

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.embedding_service = OpenAIEmbeddings()
        self.pinecone = Pinecone(api_key=PINECONE_API_KEY)
        self.chat_threads = {}  # key: thread_id, value: list of messages

        index_list = self.pinecone.list_indexes()
        if not any(idx.name == PINECONE_INDEX_NAME for idx in index_list.indexes):
            logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
            self.pinecone.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
            )

        self.index = self.pinecone.Index(PINECONE_INDEX_NAME)
        Settings.llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-4", temperature=0.1)
        vector_store = PineconeVectorStore(pinecone_index=self.index)
        self.llama_index = VectorStoreIndex.from_vector_store(vector_store, settings=Settings)

        self.bm25_corpus = []
        self.bm25_metadata = []
        self.bm25_engine = None
        logger.info("✅ Vector store initialized and LlamaIndex configured")

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def add_chunks(self, chunks: List[Dict[str, str]]):
        try:
            vectors = []
            for chunk in chunks:
                embedding = self.embedding_service.embed_documents([chunk["text"]])[0]
                unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, chunk["text"])
                vector_id = f"{chunk['doc_name']}_{unique_id}"

                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk["text"],
                        "doc_name": chunk["doc_name"],
                        "section": chunk.get("section", ""),
                        "year": chunk.get("year", ""),
                        "type": chunk.get("type", "")
                    }
                })

                self.bm25_corpus.append(chunk["text"])
                self.bm25_metadata.append(chunk)

            for i in range(0, len(vectors), 100):
                batch = vectors[i:i + 100]
                self.index.upsert(vectors=batch)
                # Write only the vector IDs to the log file
                upsert_log_file = "logs/upserted_chunks.txt"
                with open(upsert_log_file, "a") as f:
                    for vector in batch:
                        f.write(vector["id"] + "\n")
                logger.info(f"📦 Upserted batch {i // 100 + 1} with {len(batch)} vectors")

            self.bm25_engine = BM25Okapi([self._tokenize(text) for text in self.bm25_corpus])
            logger.info(f"✅ Added {len(vectors)} vectors and initialized BM25 corpus")

        except Exception as e:
            logger.error(f"❌ Failed to add chunks: {e}")

    def hybrid_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        try:
            query_vector = self.embedding_service.embed_query(query)
            semantic_results = self.index.query(
                vector=query_vector,
                top_k=k,
                include_metadata=True
            ).get("matches", [])

            semantic_docs = [
                (Document(page_content=m["metadata"]["text"], metadata=m["metadata"]), m["score"])
                for m in semantic_results
            ]

            tokenized_query = self._tokenize(query)
            bm25_scores = self.bm25_engine.get_scores(tokenized_query) if self.bm25_engine else []
            bm25_results = sorted(
                zip(self.bm25_metadata, bm25_scores),
                key=lambda x: x[1],
                reverse=True
            )[:k]

            lexical_docs = [
                (Document(page_content=md["text"], metadata=md), score)
                for md, score in bm25_results
            ]

            combined_docs = semantic_docs + lexical_docs

            def is_relevant(doc_text):
                lower = doc_text.lower()
                if "unearned reinsurance premium" in lower and "reinsurance" not in query.lower():
                    return False
                return True

            query_year = "2024" if "2024" in query else "2023" if "2023" in query else None

            filtered_docs = [
                (doc, score) for doc, score in combined_docs
                if is_relevant(doc.page_content) and
                   (not query_year or query_year in doc.page_content)
            ]

            deduped = {doc.page_content: (doc, score) for doc, score in filtered_docs}
            sorted_combined = sorted(deduped.values(), key=lambda x: x[1], reverse=True)

            logger.info(f"🔍 Hybrid search returned {len(sorted_combined)} filtered results")
            return sorted_combined[:k]

        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            return []

    def gpt_answer_from_chunks(
        self,
        query: str,
        docs: List[Tuple[Document, float]],
        thread_id: str = None,
        system_prompt: str = "You are a strict document-compliant assistant."
    ) -> str:
        # Build the document context string
        context = "\n\n".join(
            f"{doc.metadata.get('doc_name', f'Document {i+1}')} (Score: {score:.3f}):\n{doc.page_content}"
            for i, (doc, score) in enumerate(docs)
        )

        try:
            # Compose system message including documents context and instructions
            system_message = {
                "role": "system",
                "content": f"{system_prompt}\n\n--- DOCUMENTS ---\n{context}\n\nAnswer strictly based on the documents above."
            }

            # Prepare the messages list starting with system message
            messages = [system_message]

            # Add previous chat history (raw Q&A) if available
            if thread_id:
                if thread_id not in self.chat_threads:
                    self.chat_threads[thread_id] = []
                messages.extend(self.chat_threads[thread_id])

            # Add current user query as a separate user message
            current_user_message = {"role": "user", "content": query}
            messages.append(current_user_message)

            # Call OpenAI Chat Completion API
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=800,
            )
            print(messages);

            assistant_message = {"role": "assistant", "content": response.choices[0].message.content.strip()}

            # Save the new exchange to chat history if thread_id is provided
            if thread_id:
                self.chat_threads[thread_id].append(current_user_message)
                self.chat_threads[thread_id].append(assistant_message)

            return assistant_message["content"]

        except Exception as e:
            logger.error(f"❌ GPT generation failed: {e}")
            return "Sorry, I couldn’t generate a response at the moment."
