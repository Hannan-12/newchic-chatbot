import requests
import json
import logging
from typing import Dict, List, Optional
from config.settings import Settings

class OllamaClient:
    def __init__(self):
        self.base_url = Settings.OLLAMA_BASE_URL
        self.embedding_model = Settings.EMBEDDING_MODEL
        self.llm_model = Settings.LLM_MODEL
        self.logger = logging.getLogger(__name__)

    def generate_response(self, 
                         prompt: str, 
                         temperature: float = Settings.TEMPERATURE) -> str:
        """Generate a response using the Ollama API."""
        try:
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.llm_model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '')
            
        except Exception as e:
            self.logger.error(f"Error generating response from Ollama: {str(e)}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using nomic-embed-text model."""
        try:
            url = f"{self.base_url}/api/embeddings"
            
            payload = {
                "model": self.embedding_model,
                "prompt": text,
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result.get('embedding', [])
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * 384  # Default embedding size

    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.error(f"Error in batch embedding generation: {str(e)}")
                embeddings.append([0.0] * 384)
        return embeddings

    def get_model_info(self) -> Dict:
        """Get information about the models."""
        return {
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "base_url": self.base_url
        }