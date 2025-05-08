import logging
from langgraph_chat.chat_service import ChatService  # Adjust path if needed

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize chat engine
chat_engine = ChatService()


class ChatController:
    """
    ChatController handles user-facing chat logic, including
    input validation, logging, and LangGraph invocation.
    """

    @staticmethod
    def ask_question(user_input: str) -> dict:
        """
        Takes raw user input and returns structured chatbot response.

        Args:
            user_input (str): The question from the user.

        Returns:
            dict: {
                "question": original input,
                "answer": response from LangGraph,
                "status": success/fail
            }
        """
        logger.info(f"🎯 Incoming question: {user_input}")
        if not user_input or not user_input.strip():
            logger.warning("⚠️ Empty input received.")
            return {
                "question": user_input,
                "answer": "Please provide a valid question.",
                "status": "fail"
            }

        try:
            response = chat_engine.ask(user_input.strip())
            return {
                "question": user_input,
                "answer": response,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"❌ Error while answering: {e}")
            return {
                "question": user_input,
                "answer": "Something went wrong while generating a response.",
                "status": "error"
            }
