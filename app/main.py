import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router as chat_routes
from ingestion.ingest_docs import process_all_documents  # Adjust if the path differs

# Load environment variables
load_dotenv()

# Initialize FastAPI app with Swagger metadata
app = FastAPI(
    title="People's Insurance Chatbot API",
    description="An AI-powered API that allows users to ask questions based on uploaded financial and policy documents using hybrid search (Pinecone + BM25) and GPT-4o.",
    version="1.0.0",
    contact={
        "name": "People's Insurance AI Team",
        "email": "support@peoplesinsurance.lk",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# Enable CORS (Allow all origins during dev, restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include chat routes
app.include_router(chat_routes)

# Health check route
@app.get("/", tags=["Health Check"])
async def health_check():
    return {"status": "ok", "message": "Chatbot API is running. Visit /docs for Swagger UI."}

# Optional startup ingestion
@app.on_event("startup")
async def startup_ingestion():
    try:
        print("🚀 Running startup document ingestion...")
        # process_all_documents()
        print("✅ Ingestion completed successfully at startup.")
    except Exception as e:
        print(f"❌ Failed to ingest documents on startup: {e}")
