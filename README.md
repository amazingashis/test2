# Semantic RAG System with Relationship Graphs

A comprehensive semantic RAG (Retrieval-Augmented Generation) system that builds relationship graphs and stores embeddings in a vector database for intelligent document analysis and question answering.

## Features

- **Multi-format File Processing**: Parse CSV, PDF, and Python files
- **Semantic Embeddings**: Generate embeddings using BGE Large EN v1.5 model
- **Relationship Graphs**: Build semantic relationship graphs between documents and fields
- **Vector Database**: Store embeddings in ChromaDB for efficient similarity search
- **Advanced Q&A**: Answer questions using Claude Sonnet 4 with context from graphs and embeddings
- **Complex Analysis**: Relationship analysis using Meta Llama 3 70B Instruct

## Architecture

```
Semantic RAG System
├── File Parsers (CSV, PDF, Python)
├── Embedding Generator (BGE Large EN v1.5)
├── Relationship Graph Builder (NetworkX)
├── Vector Database (ChromaDB)
├── Model Orchestrator
│   ├── BGE Large EN v1.5 (Embeddings)
│   ├── Meta Llama 3 70B (Relationship Analysis)
│   └── Claude Sonnet 4 (Question Answering)
└── Semantic RAG Engine
```

## Installation

### � Setup Instructions
```bash
# 1. Clone repository
git clone <repository-url>
cd semantic-rag

# 2. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create necessary directories
mkdir -p data/chroma_db data/demo_chroma_db exports logs files

# 5. Setup configuration
cp config_enhanced.env .env
# Edit .env with your API credentials
```

### 🔑 API Configuration
Edit the `.env` file with your Databricks credentials:
```bash
DATABRICKS_WORKSPACE_URL=your_workspace_url
DATABRICKS_TOKEN=your_token
BGE_EMBEDDING_ENDPOINT=databricks-bge-large-en
LLAMA_RELATIONSHIP_ENDPOINT=databricks-meta-llama-3-3-70b-instruct
CLAUDE_QA_ENDPOINT=databricks-claude-3-5-sonnet
```

## Configuration

### Required API Keys

- **Databricks API Key**: For BGE Large EN v1.5, Meta Llama 3 70B, and Claude Sonnet 4 models

### Environment Variables

Set your Databricks API key as an environment variable:

```bash
export DATABRICKS_API_KEY=your_databricks_key
```

Or create a `.env` file with:
```bash
DATABRICKS_API_KEY=your_databricks_key
```

## Usage

### Command Line Interface

1. **Process files and build the system**:
   ```bash
   python main.py --process-files files/
   ```

2. **Query the system**:
   ```bash
   python main.py --query "What are the main claims fields in the data?"
   ```

3. **Export system state**:
   ```bash
   python main.py --export-state exports/
   ```

4. **Create graph visualization**:
   ```bash
   python main.py --visualize-graph graph.html
   ```

5. **Interactive mode**:
   ```bash
   python main.py --interactive
   ```

### File Structure

The system expects files in the following structure:

```
files/
└── [layout_name]/
    └── [data_source]/
        └── [client_name]/
            ├── mapping_code.py
            ├── mapping_document_source_to_stage.csv
            └── data_dictionary.pdf
```

For example:
```
files/
└── claim_data/
    └── cms/
        └── facebook/
            ├── mappign_code.py
            ├── claims_mapping document.csv
            └── CCLF IP V36-64-85.pdf
```

### Python API

```python
from src.rag.semantic_rag import SemanticRAG

# Initialize system
rag = SemanticRAG(
    collection_name="my_rag",
    persist_directory="./data/chroma_db"
)

# Process files
results = rag.process_files("files/")

# Generate embeddings
rag.generate_embeddings(use_local=True)

# Build relationship graph
rag.build_relationship_graph()

# Query the system
response = rag.query("What fields are available in the claims data?")
print(response['answer'])
```