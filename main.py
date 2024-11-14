import os
import logging
import logging.config
from pathlib import Path
from typing import Dict, Optional

from database.embeddings import EmbeddingGenerator
from database.vector_store import VectorStore
from api.ollama_client import OllamaClient
from agents.schema_analyzer import SchemaAnalyzer
from agents.data_processor import DataProcessor
from agents.query_agent import QueryAgent
from ui.gradio_app import ProductCatalogUI
from config.settings import Settings
from dotenv import load_dotenv

class ProductCatalogSystem:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Create necessary directories
        Settings.create_directories()
        
        # Initialize basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
        try:
            # Check if Ollama is running
            self._check_ollama_status()
            
            # Initialize components
            self.initialize_components()
            self.logger.info("System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing system: {str(e)}")
            raise

    def _check_ollama_status(self):
        """Check if Ollama server is running and models are available."""
        try:
            ollama_client = OllamaClient()
            # Try to ping the server
            ollama_client.get_model_info()
            self.logger.info("Successfully connected to Ollama server")
        except Exception as e:
            self.logger.error(f"Error connecting to Ollama server: {str(e)}")
            self.logger.error("""
                Please ensure:
                1. Ollama is installed and running
                2. Ollama server is accessible at http://localhost:11434
                3. Required models are pulled:
                   - ollama pull nomic-embed-text
                   - ollama pull llama2
            """)
            raise

    def initialize_components(self):
        """Initialize all system components."""
        # Initialize Ollama client
        self.ollama_client = OllamaClient()
        self.logger.info(f"Initialized Ollama client with models: {self.ollama_client.get_model_info()}")
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(
            ollama_client=self.ollama_client
        )
        self.logger.info("Initialized embedding generator")
        
        # Initialize vector store
        self.vector_store = VectorStore(
            embedding_generator=self.embedding_generator,
            persist_directory=str(Settings.VECTOR_STORE_DIR)
        )
        self.logger.info("Initialized vector store")
        
        # Initialize agents
        self.schema_analyzer = SchemaAnalyzer(self.ollama_client)
        self.data_processor = DataProcessor(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator
        )
        self.query_agent = QueryAgent(
            vector_store=self.vector_store,
            ollama_client=self.ollama_client
        )
        self.logger.info("Initialized agents")
        
        # Initialize UI
        self.ui = ProductCatalogUI(
            schema_analyzer=self.schema_analyzer,
            data_processor=self.data_processor,
            query_agent=self.query_agent
        )
        self.logger.info("Initialized UI")

    def process_initial_data(self, directory_path: Optional[str] = None) -> None:
        """Process initial data if provided."""
        if directory_path:
            try:
                directory = Path(directory_path)
                if not directory.exists():
                    self.logger.error(f"Directory not found: {directory_path}")
                    raise FileNotFoundError(f"Directory not found: {directory_path}")
                
                self.logger.info(f"Processing initial data from {directory_path}")
                results = self.data_processor.process_directory(directory_path)
                self.logger.info(f"Processed {len(results)} files from initial data")
                
                # Log processing summary
                total_rows = sum(result.get('rows_processed', 0) for result in results)
                self.logger.info(f"Total rows processed: {total_rows}")
                
            except Exception as e:
                self.logger.error(f"Error processing initial data: {str(e)}")
                raise

    def start_ui(self, share: bool = False) -> None:
        """Start the Gradio interface."""
        try:
            self.logger.info("Starting Gradio interface")
            interface = self.ui.create_interface()
            
            # Get vector store stats
            stats = self.vector_store.get_collection_stats()
            self.logger.info(f"Vector store stats: {stats}")
            
            # Launch the interface
            interface.launch(
                server_name=Settings.API_HOST,
                server_port=Settings.API_PORT,
                share=share,
                show_error=True
            )
            
        except Exception as e:
            self.logger.error(f"Error starting UI: {str(e)}")
            raise

    def run(self, initial_data_dir: Optional[str] = None, share_ui: bool = False) -> None:
        """Run the complete system."""
        try:
            self.logger.info("Starting Product Catalog System")
            
            # Process initial data if provided
            if initial_data_dir:
                self.process_initial_data(initial_data_dir)
            
            # Start the UI
            self.start_ui(share=share_ui)
            
        except Exception as e:
            self.logger.error(f"Error running system: {str(e)}")
            raise

def main():
    try:
        # Initialize system
        system = ProductCatalogSystem()
        
        # Get initial data directory from environment
        initial_data_dir = os.getenv("INITIAL_DATA_DIR")
        
        if initial_data_dir:
            logging.info(f"Found initial data directory: {initial_data_dir}")
        
        # Run system
        system.run(
            initial_data_dir=initial_data_dir,
            share_ui=True
        )
        
    except Exception as e:
        logging.error(f"Application failed to start: {str(e)}")
        raise

if __name__ == "__main__":
    main()