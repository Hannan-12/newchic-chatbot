from typing import List
import logging
from tqdm import tqdm
from config.settings import Settings
from api.ollama_client import OllamaClient

class EmbeddingGenerator:
    def __init__(self, ollama_client: OllamaClient):
        self.logger = logging.getLogger(__name__)
        self.client = ollama_client

    def generate(self, text: str) -> List[float]:
        """Generate embedding for a single text using Ollama API."""
        try:
            embedding = self.client.generate_embedding(text)
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * 384

    def batch_generate(self, texts: List[str], batch_size: int = Settings.BATCH_SIZE) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        embeddings = []
        
        try:
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.client.batch_generate_embeddings(batch)
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating batch embeddings: {str(e)}")
            return [[0.0] * 384 for _ in range(len(texts))]