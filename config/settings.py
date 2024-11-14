import os
from pathlib import Path

class Settings:
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    VECTOR_STORE_DIR = DATA_DIR / "vectorstore"
    
    # Ollama Configuration
    OLLAMA_BASE_URL = "http://localhost:11434"
    EMBEDDING_MODEL = "nomic-embed-text"
    LLM_MODEL = "llama3.2"
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # Model Configuration
    MAX_TOKENS = 4096
    TEMPERATURE = 0.7
    
    # Processing Configuration
    BATCH_SIZE = 32
    MAX_WORKERS = 4
    CHUNK_SIZE = 512
    OVERLAP_SIZE = 50

    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)