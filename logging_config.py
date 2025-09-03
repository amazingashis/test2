"""
Logging configuration for cleaner output with progress indicators
"""
import logging
import sys

def setup_clean_logging():
    """Setup logging configuration for cleaner output"""
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Reduce verbosity for specific loggers
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('chromadb.telemetry').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    # Keep important loggers at INFO level
    logging.getLogger('src.rag.semantic_rag').setLevel(logging.INFO)
    logging.getLogger('embeddings.embedder').setLevel(logging.WARNING)  # Reduce embedding verbosity
    logging.getLogger('vectordb.vectordb_client').setLevel(logging.WARNING)

def setup_quiet_logging():
    """Setup logging configuration for minimal output"""
    
    # Configure root logger for minimal output
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s:%(name)s:%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Set all loggers to WARNING or higher
    for logger_name in ['chromadb', 'embeddings', 'vectordb', 'src.rag', 'urllib3', 'requests']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

# Apply clean logging by default when imported
setup_clean_logging()
