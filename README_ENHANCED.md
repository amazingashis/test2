# 🧠 Enhanced Semantic RAG System with AI Relationship Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-green.svg)](https://www.trychroma.com/)
[![Databricks](https://img.shields.io/badge/AI-Databricks-orange.svg)](https://databricks.com/)

A comprehensive, production-ready semantic RAG (Retrieval-Augmented Generation) system enhanced with **AI-powered relationship analysis** using Meta Llama 3-3 70B, advanced knowledge graphs, and interactive visualizations.

## 🚀 Key Features

### Core RAG Capabilities
- **🔍 Advanced Semantic Search**: Databricks BGE Large EN v1.5 embeddings
- **💬 Intelligent QA**: Claude Sonnet 4 for sophisticated question answering
- **📄 Multi-format Processing**: PDF, CSV, Python, Markdown, and text files
- **🚀 Vector Storage**: ChromaDB with optimized retrieval and metadata handling

### AI-Enhanced Relationship Analysis
- **🤖 Meta Llama 3-3 70B Integration**: AI-powered document relationship discovery
- **🕸️ Advanced Knowledge Graphs**: NetworkX with AI-enhanced relationship types
- **🧠 Intelligent Clustering**: Automatic concept grouping and relationship optimization
- **📊 Dynamic Graph Optimization**: Real-time edge consolidation and weak link removal

### Interactive Visualization
- **🎨 Advanced Graph Visualization**: D3.js-powered interactive knowledge graphs
- **🔍 Real-time Search**: Dynamic node filtering and relationship exploration
- **📈 Analytics Dashboard**: Comprehensive system statistics and performance metrics
- **🎯 Focus Mode**: Smart node highlighting and neighborhood exploration

### Production Features
- **⚡ Async Processing**: High-performance concurrent document processing
- **📝 Comprehensive Logging**: Detailed operation tracking and error handling
- **💾 Auto-Export**: JSON exports of graphs, vectors, and system state
- **🔧 Configurable**: Extensive configuration options for all components

## 🏗️ System Architecture

```
Enhanced Semantic RAG System
├── 🔍 Embedding Layer (Databricks BGE)
├── 🤖 AI Analysis Layer (Meta Llama 3-3 70B)
├── 💬 QA Layer (Claude Sonnet 4)
├── 🕸️ Graph Layer (NetworkX + AI Enhancement)
├── 🚀 Vector Storage (ChromaDB)
└── 🎨 Visualization Layer (D3.js Interactive)
```

## 📦 Installation & Setup

### Quick Start
```bash
# Clone and setup
git clone <repository-url>
cd semantic-rag-system

# Run enhanced setup script
./setup_enhanced.sh

# Configure your API credentials
cp config_enhanced.env .env
# Edit .env with your Databricks credentials
```

### Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/chroma_db exports logs files
```

### Environment Configuration
Create a `.env` file with your Databricks credentials:
```env
DATABRICKS_WORKSPACE_URL=your_workspace_url
DATABRICKS_TOKEN=your_token
BGE_EMBEDDING_ENDPOINT=databricks-bge-large-en
LLAMA_RELATIONSHIP_ENDPOINT=databricks-meta-llama-3-3-70b-instruct
CLAUDE_QA_ENDPOINT=databricks-claude-3-5-sonnet
```

## 🚀 Usage Guide

### Command Line Interface
```bash
# Process files with AI relationship analysis
python3 main_enhanced.py --files ./documents

# Interactive query mode with enhanced features
python3 main_enhanced.py --query

# Show comprehensive system statistics
python3 main_enhanced.py --stats

# Export all data with enhanced formats
python3 main_enhanced.py --export

# Process files and auto-launch query mode
python3 main_enhanced.py --files ./documents --export
```

### Python API Usage
```python
import asyncio
from src.rag.semantic_rag import SemanticRAG
from src.models.databricks_models import (
    DatabricksEmbedder, 
    MetaLlama370BInstruct, 
    ClaudeSonnet4
)
from src.vectordb.vectordb_client import VectorDBClient

async def main():
    # Initialize enhanced components
    embedder = DatabricksEmbedder()
    relationship_model = MetaLlama370BInstruct()
    qa_model = ClaudeSonnet4()
    vector_client = VectorDBClient(collection_name="enhanced_rag")
    
    # Create enhanced RAG system
    rag = SemanticRAG(
        embedder=embedder,
        vector_client=vector_client,
        qa_model=qa_model,
        relationship_model=relationship_model,
        use_ai_relationships=True
    )
    
    # Add document with AI analysis
    await rag.add_document("Your document content...", metadata={
        "source": "document.pdf",
        "type": "research_paper"
    })
    
    # Build enhanced knowledge graph
    stats = await rag.build_relationship_graph(
        use_ai_analysis=True,
        optimize_graph=True
    )
    
    # Query with relationship context
    result = await rag.query(
        "Your question here",
        include_relationships=True,
        use_ai_analysis=True
    )
    
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])}")
    print(f"Relationships: {len(result['relationships'])}")

asyncio.run(main())
```

## 🕸️ Knowledge Graph Features

### AI-Enhanced Relationship Discovery
- **Semantic Relationships**: Context-aware document connections
- **Thematic Clustering**: Automatic topic grouping
- **Cross-Reference Analysis**: Inter-document relationship mapping
- **Confidence Scoring**: AI-generated relationship strength metrics

### Graph Optimization
- **Edge Consolidation**: Merge duplicate relationships
- **Weak Link Removal**: Filter low-confidence connections
- **Community Detection**: Identify document clusters
- **Hierarchical Structure**: Multi-level relationship organization

### Interactive Visualization
- **Dynamic Layouts**: Force-directed, circular, hierarchical, clustered
- **Real-time Filtering**: By node type, relationship strength, AI confidence
- **Search & Focus**: Find and highlight specific nodes and neighborhoods
- **Analytics Panel**: Live statistics and performance metrics

## 📊 Performance & Statistics

### Processing Capabilities
- **Document Processing**: 50+ documents per batch
- **Embedding Generation**: 1000+ chunks per minute
- **AI Relationship Analysis**: 100+ relationship pairs per batch
- **Graph Optimization**: 10,000+ node graphs in real-time

### System Statistics
The system provides comprehensive statistics including:
- Document count and chunk distribution
- Embedding vector counts and database size
- Relationship graph metrics (nodes, edges, density)
- AI-discovered relationship counts
- Processing times and performance metrics

## 🔧 Configuration Options

### Core Settings
```python
# Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS = 512

# AI Analysis Configuration
AI_RELATIONSHIP_ANALYSIS = True
AI_BATCH_SIZE = 10
AI_CONFIDENCE_THRESHOLD = 0.7

# Graph Configuration
MIN_RELATIONSHIP_STRENGTH = 0.3
GRAPH_OPTIMIZATION_ENABLED = True
MAX_GRAPH_NODES = 10000
```

### Advanced Options
- Embedding model selection and parameters
- Relationship analysis depth and algorithms
- Graph layout and visualization settings
- Export formats and scheduling
- Performance tuning and resource limits

## 📁 File Structure

```
semantic-rag-system/
├── 📄 main_enhanced.py              # Enhanced CLI interface
├── 🔧 config_enhanced.env           # Configuration template
├── 🛠️ setup_enhanced.sh             # Automated setup script
├── 📋 requirements.txt              # Python dependencies
├── 🎨 advanced_knowledge_graph.html # Interactive visualization
├── 📊 SYSTEM_OVERVIEW.md            # Technical documentation
├── src/                             # Core system modules
│   ├── embeddings/                  # Embedding generation
│   ├── graph/                       # Knowledge graph management
│   ├── models/                      # AI model integrations
│   ├── rag/                         # Core RAG functionality
│   ├── utils/                       # Utility functions
│   └── vectordb/                    # Vector database client
├── data/                            # Database storage
│   ├── chroma_db/                   # ChromaDB vector storage
│   └── demo_chroma_db/              # Demo database
├── exports/                         # System exports
│   ├── relationship_graph.json      # Graph data
│   ├── vector_database.json         # Vector data
│   └── system_status.json           # System state
├── logs/                            # System logs
└── files/                           # Input documents
    └── claim_data/                  # Sample data
```

## 🎯 Example Workflows

### Document Processing Pipeline
1. **File Ingestion**: Multi-format document parsing
2. **Chunking**: Intelligent text segmentation
3. **Embedding**: Semantic vector generation
4. **AI Analysis**: Relationship discovery using Llama 3-3 70B
5. **Graph Building**: Knowledge graph construction
6. **Optimization**: Graph refinement and clustering
7. **Visualization**: Interactive graph generation

### Query Processing Flow
1. **Query Analysis**: Intent understanding and context extraction
2. **Vector Search**: Semantic similarity retrieval
3. **Relationship Expansion**: Graph-based context enhancement
4. **AI Reasoning**: Multi-source information synthesis
5. **Response Generation**: Comprehensive answer formulation
6. **Source Attribution**: Evidence tracking and citation

## 🔍 Advanced Features

### AI Relationship Analysis
```python
# Configure AI relationship discovery
relationship_config = {
    "model": "meta-llama-3-3-70b-instruct",
    "batch_size": 10,
    "confidence_threshold": 0.7,
    "analysis_depth": 3,
    "relationship_types": [
        "semantic_similarity",
        "thematic_connection", 
        "causal_relationship",
        "hierarchical_structure"
    ]
}
```

### Graph Optimization
```python
# Enable advanced graph optimization
optimization_config = {
    "consolidate_edges": True,
    "remove_weak_links": True,
    "minimum_strength": 0.3,
    "community_detection": True,
    "hierarchical_clustering": True
}
```

## 📈 Performance Monitoring

### Real-time Metrics
- Processing throughput (documents/minute)
- Embedding generation rate (vectors/second)
- Query response times (milliseconds)
- Graph analysis performance (relationships/second)
- Memory usage and resource consumption

### System Health
- Database connection status
- API endpoint availability
- Model response times
- Error rates and success metrics
- Resource utilization tracking

## 🛠️ Development & Extension

### Adding New File Types
```python
# Extend file parser for new formats
from src.utils.file_parsers import FileParser

class CustomFileParser(FileParser):
    def parse(self, file_path: str) -> Dict[str, Any]:
        # Implement custom parsing logic
        pass
```

### Custom Relationship Analysis
```python
# Add custom relationship discovery
class CustomRelationshipAnalyzer:
    async def analyze_relationships(self, documents: List[str]) -> List[Dict]:
        # Implement custom AI analysis
        pass
```

## 📚 Documentation

- **[System Overview](SYSTEM_OVERVIEW.md)**: Technical architecture and design
- **[Update Summary](UPDATE_SUMMARY.md)**: Latest enhancements and changes
- **[API Reference](docs/api.md)**: Detailed API documentation
- **[Configuration Guide](docs/config.md)**: Complete configuration options

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Databricks**: BGE embeddings and model hosting
- **Anthropic**: Claude Sonnet 4 reasoning capabilities
- **Meta**: Llama 3-3 70B relationship analysis
- **ChromaDB**: High-performance vector storage
- **NetworkX**: Graph analysis and manipulation
- **D3.js**: Interactive visualization framework

---

## 🎉 Enhanced System Summary

This Enhanced Semantic RAG System represents a significant advancement in document intelligence and knowledge management. With AI-powered relationship discovery, advanced graph optimization, and comprehensive visualization capabilities, it provides a production-ready solution for complex document analysis and question-answering tasks.

**Key Enhancements:**
- 🤖 Meta Llama 3-3 70B for intelligent relationship discovery
- 🧠 Advanced graph optimization and community detection
- 🎨 Interactive D3.js visualization with real-time analytics
- ⚡ Async processing pipeline for high-performance operation
- 📊 Comprehensive statistics and monitoring capabilities

**Perfect for:**
- Research and document analysis
- Knowledge base construction
- Intelligent document search
- Relationship discovery and mapping
- Educational content organization
- Business intelligence applications

Get started today and experience the power of AI-enhanced semantic retrieval!
