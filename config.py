import os
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # OpenAI settings
    OPENAI_API_KEY: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
    OPENAI_EMBEDDING_MODEL: str = Field(default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"))
    OPENAI_CHAT_MODEL: str = Field(default=os.getenv("OPENAI_CHAT_MODEL", "gpt-4"))
    
    # Pinecone settings
    PINECONE_API_KEY: str = Field(default=os.getenv("PINECONE_API_KEY", ""))
    PINECONE_ENVIRONMENT: str = Field(default=os.getenv("PINECONE_ENVIRONMENT", ""))
    PINECONE_INDEX_NAME: str = Field(default=os.getenv("PINECONE_INDEX_NAME", "insurance-knowledge"))
    
    # Document settings
    DOCUMENTS_DIR: str = Field(default=os.getenv("DOCUMENTS_DIR", "./data/documents"))
    CHUNK_SIZE: int = Field(default=int(os.getenv("CHUNK_SIZE", "1000")))
    CHUNK_OVERLAP: int = Field(default=int(os.getenv("CHUNK_OVERLAP", "200")))
    
    # Application settings
    MAX_CONTEXT_DOCUMENTS: int = Field(default=int(os.getenv("MAX_CONTEXT_DOCUMENTS", "5")))
    RELEVANCE_THRESHOLD: float = Field(default=float(os.getenv("RELEVANCE_THRESHOLD", "0.75")))

    class Config:
        env_file = ".env.example"

settings = Settings()