#!/bin/bash

# Enhanced Semantic RAG System Setup Script
# Comprehensive setup with AI relationship analysis capabilities

set -e

echo "🚀 Setting up Enhanced Semantic RAG System..."
echo "============================================"

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data/chroma_db
mkdir -p data/demo_chroma_db
mkdir -p exports
mkdir -p logs
mkdir -p files/claim_data
mkdir -p src/embeddings/__pycache__
mkdir -p src/graph/__pycache__
mkdir -p src/models/__pycache__
mkdir -p src/rag/__pycache__
mkdir -p src/utils/__pycache__
mkdir -p src/vectordb/__pycache__

echo "✅ Directory structure created!"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📋 Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found. Installing core dependencies..."
    pip install \
        chromadb \
        networkx \
        matplotlib \
        numpy \
        requests \
        python-dotenv \
        asyncio \
        pathlib \
        aiofiles
fi

echo "✅ Dependencies installed!"

# Setup environment configuration
echo "⚙️ Setting up environment configuration..."
if [ ! -f ".env" ]; then
    if [ -f "config_enhanced.env" ]; then
        cp config_enhanced.env .env
        echo "📋 Copied enhanced configuration to .env"
        echo "⚠️ Please edit .env with your actual API keys and endpoints"
    else
        echo "# Enhanced Semantic RAG Configuration" > .env
        echo "DATABRICKS_WORKSPACE_URL=your_workspace_url" >> .env
        echo "DATABRICKS_TOKEN=your_token" >> .env
        echo "🔧 Created basic .env file - please configure with your credentials"
    fi
else
    echo "✅ .env file already exists"
fi

# Setup logging
echo "📝 Setting up logging..."
if [ ! -f "logs/semantic_rag.log" ]; then
    touch logs/semantic_rag.log
    echo "✅ Log file created"
fi

# Test Python imports
echo "🧪 Testing Python module imports..."
python3 -c "
import sys
import os
sys.path.append('.')

try:
    from src.embeddings.embedder import *
    print('✅ Embeddings module imported successfully')
except ImportError as e:
    print(f'❌ Embeddings import failed: {e}')

try:
    from src.graph.relationship_graph import *
    print('✅ Graph module imported successfully')
except ImportError as e:
    print(f'❌ Graph import failed: {e}')

try:
    from src.models.databricks_models import *
    print('✅ Models module imported successfully')
except ImportError as e:
    print(f'❌ Models import failed: {e}')

try:
    from src.rag.semantic_rag import *
    print('✅ RAG module imported successfully')
except ImportError as e:
    print(f'❌ RAG import failed: {e}')

try:
    from src.utils.file_parsers import *
    print('✅ Utils module imported successfully')
except ImportError as e:
    print(f'❌ Utils import failed: {e}')

try:
    from src.vectordb.vectordb_client import *
    print('✅ VectorDB module imported successfully')
except ImportError as e:
    print(f'❌ VectorDB import failed: {e}')
"

# Check if main files exist
echo "📋 Checking main system files..."
files=(
    "src/embeddings/embedder.py"
    "src/graph/relationship_graph.py"
    "src/models/databricks_models.py"
    "src/rag/semantic_rag.py"
    "src/utils/file_parsers.py"
    "src/vectordb/vectordb_client.py"
    "main.py"
    "main_enhanced.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
    fi
done

# Check visualization files
echo "🎨 Checking visualization files..."
if [ -f "knowledge_graph.html" ]; then
    echo "✅ Basic knowledge graph visualization available"
fi

if [ -f "advanced_knowledge_graph.html" ]; then
    echo "✅ Advanced knowledge graph visualization available"
fi

# Create demo script
echo "📝 Creating demo script..."
cat > run_demo.sh << 'EOF'
#!/bin/bash

echo "🚀 Running Enhanced Semantic RAG Demo..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if files directory exists and has content
if [ -d "files" ] && [ "$(ls -A files)" ]; then
    echo "📁 Processing files from files/ directory..."
    python3 main_enhanced.py --files files --export
else
    echo "⚠️ No files found in files/ directory"
    echo "Please add some documents to process"
    echo ""
    echo "Example usage:"
    echo "  mkdir -p files"
    echo "  # Add your documents to files/"
    echo "  python3 main_enhanced.py --files files"
fi
EOF

chmod +x run_demo.sh

# Create test script
echo "🧪 Creating test script..."
cat > test_system.sh << 'EOF'
#!/bin/bash

echo "🧪 Testing Enhanced Semantic RAG System..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Test imports
echo "🔍 Testing module imports..."
python3 -c "
import sys
sys.path.append('.')

modules = [
    'src.embeddings.embedder',
    'src.graph.relationship_graph', 
    'src.models.databricks_models',
    'src.rag.semantic_rag',
    'src.utils.file_parsers',
    'src.vectordb.vectordb_client'
]

success = True
for module in modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except Exception as e:
        print(f'❌ {module}: {e}')
        success = False

if success:
    print('✅ All modules imported successfully!')
else:
    print('❌ Some modules failed to import')
"

# Test database initialization
echo "💾 Testing database initialization..."
python3 -c "
import sys
sys.path.append('.')
from src.vectordb.vectordb_client import VectorDBClient

try:
    client = VectorDBClient(collection_name='test_collection')
    print('✅ Vector database initialized successfully')
except Exception as e:
    print(f'❌ Database initialization failed: {e}')
"

echo "✅ System test completed!"
EOF

chmod +x test_system.sh

# Final setup verification
echo ""
echo "🎉 Enhanced Semantic RAG System Setup Complete!"
echo "=============================================="
echo ""
echo "📋 Next Steps:"
echo "1. Configure your API credentials in .env file"
echo "2. Add documents to the files/ directory"
echo "3. Run: ./run_demo.sh to test the system"
echo "4. Or run: python3 main_enhanced.py --help for options"
echo ""
echo "🛠️ Available Commands:"
echo "  ./run_demo.sh                    # Run demo with file processing"
echo "  ./test_system.sh                 # Test system components"
echo "  python3 main_enhanced.py --help  # Show all options"
echo ""
echo "📊 System Features:"
echo "  🔍 Databricks BGE embeddings"
echo "  🤖 Meta Llama 3-3 70B AI analysis"
echo "  💬 Claude Sonnet 4 QA"
echo "  🕸️ NetworkX knowledge graphs"
echo "  🚀 ChromaDB vector storage"
echo "  🎨 Interactive graph visualization"
echo ""
echo "✅ Setup completed successfully!"

# Deactivate virtual environment
deactivate 2>/dev/null || true
