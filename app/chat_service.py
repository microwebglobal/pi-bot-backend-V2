import os
import logging
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from ingestion.vectorstore import VectorStore
import uuid

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate config
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set in the environment variables.")

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load system prompt from external file
SYSTEM_PROMPT_PATH = "prompts/system_prompt.txt"
try:
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read().strip()
        logger.info("✅ Loaded system prompt from file")
except Exception as e:
    logger.error(f"❌ Failed to load system prompt: {e}")
    SYSTEM_PROMPT = "You are a helpful assistant based only on official uploaded documents."

# Initialize core components
vector_store = VectorStore()
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)

# Define LangGraph state structure
class ChatState(TypedDict):
    question: str
    context: str
    answer: str
    thread_id: str

class ChatService:
    def __init__(self):
        def generate_answer(state: ChatState) -> ChatState:
            question = state["question"].strip()
            thread_id = state.get("thread_id", "default_thread")
            logger.info(f"🔍 Received question: {question}")

            # Step 1: Retrieve context using hybrid search
            try:
                results = vector_store.hybrid_search(question, k=6)
                context = "\n\n".join(
                    f"{i+1}. {doc.page_content}" for i, (doc, _) in enumerate(results)
                )
                logger.info(f"📚 Retrieved {len(results)} relevant chunks")
            except Exception as e:
                logger.error(f"❌ Retrieval error: {e}")
                return {"question": question, "context": "", "answer": "Failed to retrieve context."}

            # Step 2: Answer with GPT using system instructions + context
            try:
                answer = vector_store.gpt_answer_from_chunks(question, results, system_prompt=SYSTEM_PROMPT,thread_id=thread_id)
            except Exception as e:
                logger.error(f"❌ GPT generation error: {e}")
                answer = "Failed to generate a response at the moment."

            return {
                "question": question,
                "context": context,
                "answer": answer,
                "thread_id": thread_id
            }

        # LangGraph setup
        builder = StateGraph(ChatState)
        builder.add_node("generate", generate_answer)
        builder.set_entry_point("generate")
        builder.set_finish_point("generate")
        self.graph = builder.compile()

        logger.info("✅ ChatService graph initialized successfully")

    def ask(self, question: str, thread_id: str = None) -> str:
        # First message: Generate a new thread_id if it's the first message
        if thread_id is None:
            # Generate a unique thread_id for the first message using uuid
            print("There is no thread id");
            thread_id = str(uuid.uuid4())  # Use UUID to generate a unique thread ID
            logger.info(f"💡 Created new thread ID: {thread_id}")

            # For the first message, pass thread_id as generated
            result = self.graph.invoke({"question": question, "context": "", "thread_id": thread_id})
        else:
            # For subsequent messages, use the provided thread_id
            result = self.graph.invoke({"question": question, "context": "", "thread_id": thread_id})
        
        return {
    "answer": result.get("answer", "No answer returned."),
    "thread_id": result.get("thread_id", thread_id)
    }

