# Semantic RAG System - Complete Implementation Overview

## What Was Built

A comprehensive **Semantic RAG (Retrieval-Augmented Generation)** system with relationship graphs and vector database integration for intelligent document analysis and question answering.

## 🎯 Project Summary

This system processes multiple file formats (CSV, PDF, Python), generates semantic embeddings, builds relationship graphs, and provides intelligent question answering using state-of-the-art AI models.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 Semantic RAG System                         │
├─────────────────────────────────────────────────────────────┤
│  Input Layer: CSV, PDF, Python Files                       │
│     ↓                                                       │
│  File Parsers (Extract & Structure Content)                │
│     ↓                                                       │
│  Embedding Generator (BGE Large EN v1.5)                   │
│     ↓                                                       │
│  Vector Database (ChromaDB) + Relationship Graph (NetworkX)│
│     ↓                                                       │
│  Query Engine + Context Aggregation                        │
│     ↓                                                       │
│  Answer Generation (Claude Sonnet 4)                       │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
/workspaces/test2/
├── main.py                    # Main CLI interface
├── requirements.txt           # Python dependencies
├── setup.sh                  # Setup script
├── .env.example              # Environment configuration template
├── .gitignore               # Git ignore patterns
├── README.md                # Project documentation
├── 
├── src/                     # Source code
│   ├── embeddings/
│   │   └── embedder.py      # BGE embedding generation & semantic matching
│   ├── graph/
│   │   └── relationship_graph.py  # NetworkX relationship graphs
│   ├── models/
│   │   └── databricks_models.py   # AI model integrations
│   ├── rag/
│   │   └── semantic_rag.py  # Main RAG engine
│   ├── utils/
│   │   └── file_parsers.py  # CSV, PDF, Python parsers
│   └── vectordb/
│       └── vectordb_client.py     # ChromaDB client
│
├── files/                   # Input data
│   └── claim_data/
│       └── cms/
│           └── facebook/
│               ├── mappign_code.py
│               ├── claims_mapping document.csv
│               └── CCLF IP V36-64-85.pdf
│
└── data/                    # Generated data (ignored by git)
    ├── chroma_db/          # Vector database
    ├── exports/            # System exports
    └── logs/               # Log files
```

## 🧠 AI Models Integration

### 1. **BGE Large EN v1.5** (Databricks)
- **Purpose**: Semantic embeddings generation
- **Use Cases**: 
  - Field similarity detection
  - Content semantic matching
  - Document similarity scoring
- **Integration**: Databricks hosted endpoint with local fallback

### 2. **Meta Llama 3 70B Instruct** (Databricks)
- **Purpose**: Complex relationship analysis
- **Use Cases**:
  - Cross-document relationship identification
  - Field mapping generation
  - Schema analysis
- **Integration**: Databricks hosted endpoint

### 3. **Claude Sonnet 4** (Anthropic)
- **Purpose**: Question answering and summarization
- **Use Cases**:
  - Final answer generation
  - Document summarization
  - Entity extraction
- **Integration**: Anthropic API

## 🔧 Core Components

### 1. File Parsers (`src/utils/file_parsers.py`)
**What it does**: Extracts and structures content from different file formats

**Features**:
- **CSV Parser**: Row-by-row extraction with field mapping
- **PDF Parser**: Text extraction with PyPDF2, chunking for large documents
- **Python Parser**: AST analysis for functions, classes, imports, variables
- **Metadata Extraction**: File information, creation dates, structure analysis

**Example Usage**:
```python
from src.utils.file_parsers import FileParser

parser = FileParser()
csv_data = parser.parse_csv("data.csv")
pdf_content = parser.parse_pdf("document.pdf")
python_info = parser.parse_python_file("script.py")
```

### 2. Embedding Generator (`src/embeddings/embedder.py`)
**What it does**: Converts text into semantic vector representations

**Features**:
- **BGE Integration**: Primary embedding model via Databricks API
- **Production-Ready**: Databricks BGE API integration
- **Semantic Matching**: Field and content similarity detection
- **Batch Processing**: Efficient bulk embedding generation

**Example Usage**:
```python
from src.embeddings.embedder import EmbeddingGenerator

embedder = EmbeddingGenerator()
embedding = embedder.get_embedding("Sample text")
similarities = embedder.find_field_similarities(csv_data)
```

### 3. Relationship Graph (`src/graph/relationship_graph.py`)
**What it does**: Builds and analyzes semantic relationships between documents

**Features**:
- **NetworkX Integration**: Graph construction and analysis
- **Multiple Relationship Types**: Similarity, containment, field mapping
- **Graph Analytics**: Centrality analysis, clustering, path finding
- **Visualization**: Interactive Plotly charts and static matplotlib

**Relationship Types**:
- `semantic_similarity`: Documents with similar content
- `field_similarity`: CSV fields with similar meanings
- `contains`: Document chunks within parent documents
- `source_of`: Documents from same source file
- `mapping`: Field mappings between schemas

**Example Usage**:
```python
from src.graph.relationship_graph import RelationshipGraph

graph = RelationshipGraph()
graph.add_node("doc1", node_type="document")
graph.add_relationship("doc1", "doc2", "semantic_similarity", weight=0.85)
```

### 4. Vector Database (`src/vectordb/vectordb_client.py`)
**What it does**: Persistent storage and efficient similarity search

**Features**:
- **ChromaDB Integration**: Vector storage with metadata
- **Similarity Search**: Cosine similarity queries
- **Batch Operations**: Efficient bulk insertions
- **Export/Import**: System state persistence

**Example Usage**:
```python
from src.vectordb.vectordb_client import VectorDBClient

vectordb = VectorDBClient("my_collection")
vectordb.add_embedding("doc1", embedding, "Document content", metadata)
results = vectordb.query(query_embedding, n_results=5)
```

### 5. Model Orchestrator (`src/models/databricks_models.py`)
**What it does**: Coordinates multiple AI models for comprehensive analysis

**Features**:
- **BGE Client**: Embedding generation and similarity
- **Llama Client**: Relationship analysis and mapping
- **Claude Client**: Question answering and summarization
- **Orchestrator**: Comprehensive multi-model analysis

### 6. Semantic RAG Engine (`src/rag/semantic_rag.py`)
**What it does**: Main system orchestration and query processing

**Features**:
- **File Processing Pipeline**: End-to-end document processing
- **Embedding Generation**: Batch processing with progress tracking
- **Graph Construction**: Automatic relationship discovery
- **Query Processing**: Multi-source context aggregation
- **System Management**: Status monitoring and state export

## 🚀 How It Works

### 1. **File Processing**
```python
# Process all files in a directory
results = semantic_rag.process_files("files/")

# What happens:
# 1. Discovers all CSV, PDF, Python files
# 2. Extracts content and metadata
# 3. Structures data for further processing
```

### 2. **Embedding Generation**
```python
# Generate embeddings for all documents
embedding_results = semantic_rag.generate_embeddings(use_local=True)

# What happens:
# 1. Batches documents for efficient processing
# 2. Calls BGE Large EN v1.5 for embeddings
# 3. Stores in ChromaDB with metadata
```

### 3. **Relationship Graph Building**
```python
# Build semantic relationships
graph_results = semantic_rag.build_relationship_graph()

# What happens:
# 1. Analyzes semantic similarities between documents
# 2. Identifies field relationships in CSV data
# 3. Creates hierarchical relationships (chunks to documents)
# 4. Builds graph with NetworkX
```

### 4. **Query Processing**
```python
# Query the system
response = semantic_rag.query("What are the main claims fields?")

# What happens:
# 1. Generates embedding for query
# 2. Searches vector database for similar content
# 3. Traverses relationship graph for related context
# 4. Aggregates context from multiple sources
# 5. Uses Claude to generate final answer
```

## 🎮 Usage Examples

### Command Line Interface

```bash
# 1. Process files and build system
python main.py --process-files files/

# 2. Query the system
python main.py --query "What are the main claims fields in the data?"

# 3. Export system state
python main.py --export-state exports/

# 4. Create interactive graph visualization
python main.py --visualize-graph graph.html

# 5. Interactive mode
python main.py --interactive
```

### Python API

```python
from src.rag.semantic_rag import SemanticRAG

# Initialize system
rag = SemanticRAG(
    collection_name="claims_analysis",
    persist_directory="./data/chroma_db"
)

# Process files
rag.process_files("files/claim_data/")

# Generate embeddings
rag.generate_embeddings(use_local=True)

# Build relationships
rag.build_relationship_graph()

# Query system
response = rag.query("How do source fields map to target schema?")
print(response['answer'])

# Export results
rag.export_system_state("exports/")
```

## 📊 What The System Discovered

Based on the current files in `files/claim_data/cms/facebook/`:

### Documents Processed:
1. **`mappign_code.py`** - Python mapping code
2. **`claims_mapping document.csv`** - Field mapping data
3. **`CCLF IP V36-64-85.pdf`** - Claims documentation

### Relationships Identified:
- **Semantic Similarities**: Between CSV rows and PDF content
- **Field Mappings**: Source to target field relationships
- **Code Dependencies**: Python functions and data structures
- **Documentation Links**: PDF concepts to CSV fields

### Embeddings Generated:
- Document-level embeddings for each file
- Chunk-level embeddings for PDF pages
- Row-level embeddings for CSV data
- Function-level embeddings for Python code

## 🔍 System Capabilities

### 1. **Multi-Format Understanding**
- Parse and understand CSV structure and field meanings
- Extract and chunk PDF documentation
- Analyze Python code structure and dependencies

### 2. **Semantic Relationships**
- Identify similar fields across different data sources
- Connect documentation concepts to actual data fields
- Discover implicit relationships between documents

### 3. **Intelligent Querying**
- Natural language questions about the data
- Context-aware responses using multiple information sources
- Graph-traversal for related information discovery

### 4. **Visual Analytics**
- Interactive relationship graphs
- Document similarity networks
- Field mapping visualizations

## 📈 Performance & Scalability

### Current Configuration:
- **Batch Size**: 16-32 documents for embedding generation
- **Vector Storage**: ChromaDB persistent storage
- **Memory Usage**: Optimized for local development
- **API Calls**: Batched to minimize costs

### Scalability Features:
- **Chunking**: Large documents split for processing
- **Persistent Storage**: No re-processing on restart
- **Local Fallbacks**: Reduced API dependencies
- **Batch Processing**: Efficient bulk operations

## 🛠️ Setup & Configuration

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your API keys:
# - DATABRICKS_API_KEY
# - ANTHROPIC_API_KEY
```

### 3. **Run Setup**
```bash
chmod +x setup.sh
./setup.sh
```

### 4. **Process Files**
```bash
python main.py --process-files files/
```

## 🎯 Key Achievements

### ✅ **Implemented Features**
1. **Multi-format file parsing** (CSV, PDF, Python)
2. **Semantic embedding generation** (BGE Large EN v1.5)
3. **Relationship graph construction** (NetworkX)
4. **Vector database storage** (ChromaDB)
5. **AI model integration** (Databricks + Anthropic)
6. **Interactive CLI interface**
7. **Graph visualization** (Plotly + Matplotlib)
8. **System state export/import**
9. **Comprehensive error handling**
10. **Local fallback mechanisms**

### 🔧 **Technical Excellence**
- **Modular Architecture**: Each component is independent and reusable
- **Error Handling**: Graceful fallbacks and error recovery
- **Performance Optimization**: Batch processing and caching
- **Documentation**: Comprehensive code documentation
- **Testing Ready**: Structured for easy testing
- **Git Integration**: Proper version control setup

## 🚀 Next Steps & Extensions

### Potential Enhancements:
1. **Advanced Analytics**: More sophisticated relationship analysis
2. **Real-time Processing**: Streaming document ingestion
3. **Multi-language Support**: Documents in different languages
4. **Advanced Visualization**: 3D graph rendering
5. **API Endpoints**: REST API for integration
6. **Distributed Processing**: Scale to larger datasets
7. **Fine-tuned Models**: Domain-specific model training

## 📋 System Status

The system is **fully functional** and ready for:
- ✅ File processing and analysis
- ✅ Semantic embedding generation
- ✅ Relationship graph construction
- ✅ Vector database operations
- ✅ Query processing and response generation
- ✅ Visualization and export

## 🎉 Conclusion

This Semantic RAG system represents a comprehensive solution for intelligent document analysis. It successfully:

1. **Processes diverse file formats** with robust parsing
2. **Generates semantic embeddings** using state-of-the-art models
3. **Builds meaningful relationships** between documents and concepts
4. **Provides intelligent querying** with context-aware responses
5. **Offers visualization** and analysis capabilities
6. **Maintains scalability** and performance optimization

The system is production-ready for analyzing claim data, documentation, and code repositories, with the flexibility to extend to other domains and use cases.
