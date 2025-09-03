#!/usr/bin/env python3
"""
Semantic RAG System Main Entry Point

This script implements a comprehensive semantic RAG (Retrieval-Augmented Generation) system
with relationship graphs and vector database storage.

Features:
- File parsing (CSV, PDF, Python)
- Semantic embedding generation using BGE Large EN v1.5
- Relationship graph construction
- Vector database storage with ChromaDB
- Question answering with Claude Sonnet 4
- Complex relationship analysis with Meta Llama 3 70B

Usage:
    python main.py --process-files files/
    python main.py --query "What are the main claims fields?"
    python main.py --export-state exports/
    python main.py --visualize-graph graph.html
"""

import os
import sys
import argparse
import json
from datetime import datetime
import logging
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag.semantic_rag import SemanticRAG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('semantic_rag.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables and API keys"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.info("python-dotenv not installed, using environment variables directly")
    
    # Check for required API keys
    databricks_key = os.getenv('DATABRICKS_TOKEN')
    
    if not databricks_key:
        logger.warning("DATABRICKS_API_KEY not found. Some features may be limited.")
    
    return databricks_key

def initialize_system(databricks_key=None):
    """Initialize the Semantic RAG system"""
    logger.info("Initializing Semantic RAG system...")
    
    try:
        # Set Databricks endpoints directly
        os.environ['BGE_API_URL'] = 'https://dbc-3735add4-1cb6.cloud.databricks.com/serving-endpoints/bge_large_en_v1_5/invocations'
        os.environ['LLAMA_API_URL'] = 'https://dbc-3735add4-1cb6.cloud.databricks.com/serving-endpoints/databricks-meta-llama-3-3-70b-instruct/invocations'
        os.environ['CLAUDE_API_URL'] = 'https://dbc-3735add4-1cb6.cloud.databricks.com/serving-endpoints/databricks-claude-sonnet-4/invocations'
        os.environ['DATABRICKS_BASE_URL'] = 'https://dbc-3735add4-1cb6.cloud.databricks.com'
        
        semantic_rag = SemanticRAG(
            collection_name="semantic_rag_main",
            persist_directory="./data/chroma_db",
            databricks_api_key=databricks_key
        )
        
        logger.info("System initialized successfully")
        return semantic_rag
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        raise

def process_files_command(semantic_rag, files_directory, generate_embeddings=True, build_graph=True):
    """Process files and build the semantic RAG system"""
    logger.info(f"Processing files from: {files_directory}")
    
    if not os.path.exists(files_directory):
        logger.error(f"Directory not found: {files_directory}")
        return False
    
    try:
        # Process files
        print("🔍 Processing files...")
        processing_results = semantic_rag.process_files(files_directory)
        
        print(f"✅ Processed {processing_results['total_documents']} documents")
        print(f"   Files found: {sum(len(files) for files in processing_results['files_found'].values())}")
        
        if processing_results['errors']:
            print(f"⚠️  Errors encountered: {len(processing_results['errors'])}")
            for error in processing_results['errors'][:3]:  # Show first 3 errors
                print(f"   - {error}")
        
        # Generate embeddings
        if generate_embeddings:
            print("\n🧠 Generating embeddings...")
            embedding_results = semantic_rag.generate_embeddings(batch_size=16, use_local=True)
            
            if embedding_results['success']:
                print(f"✅ Generated embeddings for {embedding_results['documents_added']} documents")
                if embedding_results['failed_embeddings'] > 0:
                    print(f"⚠️  Failed to generate {embedding_results['failed_embeddings']} embeddings")
            else:
                print(f"❌ Embedding generation failed: {embedding_results.get('message', 'Unknown error')}")
        
        # Build relationship graph
        if build_graph:
            print("\n🕸️  Building relationship graph...")
            graph_results = semantic_rag.build_relationship_graph(similarity_threshold=0.7)
            
            if graph_results['success']:
                stats = graph_results['graph_statistics']
                print(f"✅ Built relationship graph:")
                print(f"   - Nodes: {stats['num_nodes']}")
                print(f"   - Edges: {stats['num_edges']}")
                print(f"   - Density: {stats['density']:.3f}")
                print(f"   - Content relationships: {graph_results['content_relationships']}")
            else:
                print("❌ Graph building failed")
        
        # Show system status
        print("\n📊 System Status:")
        status = semantic_rag.get_system_status()
        print(f"   - Documents processed: {status['processed_documents']}")
        print(f"   - Vector DB documents: {status['vector_database']['total_documents']}")
        print(f"   - Graph nodes: {status['relationship_graph']['num_nodes']}")
        print(f"   - Graph edges: {status['relationship_graph']['num_edges']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        print(f"❌ Error processing files: {str(e)}")
        return False

def query_command(semantic_rag, query_text, n_results=5):
    """Execute a query against the semantic RAG system"""
    logger.info(f"Executing query: {query_text}")
    
    try:
        print(f"\n🔎 Query: {query_text}")
        print("=" * 50)
        
        results = semantic_rag.query(
            query_text=query_text,
            n_results=n_results,
            include_graph_context=True
        )
        
        if results['success']:
            print(f"\n💡 Answer:")
            print(f"{results['answer']}")
            
            print(f"\n📄 Context Documents ({len(results['context_documents'])}):")
            for i, doc in enumerate(results['context_documents'], 1):
                print(f"\n{i}. Document: {doc['id']}")
                print(f"   Similarity: {doc['similarity']:.3f}")
                print(f"   Source: {doc['metadata'].get('source_file', 'Unknown')}")
                print(f"   Content: {doc['content'][:200]}...")
            
            if results['graph_context']:
                print(f"\n🕸️  Related from Graph ({len(results['graph_context'])}):")
                for i, doc in enumerate(results['graph_context'], 1):
                    print(f"\n{i}. Related: {doc['id']} ({doc['type']})")
                    print(f"   Source: {doc['source']}")
                    print(f"   Content: {doc['content'][:150]}...")
            
            print(f"\n📈 Statistics:")
            print(f"   - Total context length: {results['total_context_length']} characters")
            print(f"   - Search results: {results['search_results_count']}")
            
        else:
            print(f"❌ Query failed: {results.get('message', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        print(f"❌ Error executing query: {str(e)}")

def export_command(semantic_rag, output_directory):
    """Export system state"""
    logger.info(f"Exporting system state to: {output_directory}")
    
    try:
        print(f"\n💾 Exporting system state to: {output_directory}")
        
        export_paths = semantic_rag.export_system_state(output_directory)
        
        print("✅ Export completed:")
        for component, path in export_paths.items():
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"   - {component}: {path} ({file_size:.1f} MB)")
        
    except Exception as e:
        logger.error(f"Error exporting system state: {str(e)}")
        print(f"❌ Error exporting system state: {str(e)}")

def visualize_command(semantic_rag, output_file, interactive=True):
    """Create graph visualization"""
    logger.info(f"Creating graph visualization: {output_file}")
    
    try:
        print(f"\n📊 Creating graph visualization: {output_file}")
        
        # Check if current graph has data
        current_nodes = semantic_rag.relationship_graph.graph.number_of_nodes()
        current_edges = semantic_rag.relationship_graph.graph.number_of_edges()
        
        if current_nodes == 0:
            # Try to load existing graph data first
            graph_file = "exports/relationship_graph.json"
            if os.path.exists(graph_file):
                print(f"📈 Loading existing graph data from {graph_file}")
                semantic_rag.relationship_graph.load_graph(graph_file)
                current_nodes = semantic_rag.relationship_graph.graph.number_of_nodes()
                current_edges = semantic_rag.relationship_graph.graph.number_of_edges()
                logger.info(f"Loaded graph with {current_nodes} nodes and {current_edges} edges")
            
            # If still empty, inform user to run processing
            if current_nodes == 0:
                print("⚠️  No graph data available for visualization.")
                print("📝 Please run the full processing pipeline first:")
                print("   python main.py --process-files files/")
                print("")
                print("This will:")
                print("  • Process your files (CSV, PDF, Python)")
                print("  • Generate embeddings via Databricks BGE")
                print("  • Build relationship graphs")
                print("  • Save graph data for visualization")
                return
        else:
            print(f"📈 Using current graph data: {current_nodes} nodes, {current_edges} edges")
        
        semantic_rag.visualize_graph(
            output_file=output_file,
            interactive=interactive
        )
        
        print(f"✅ Visualization saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        print(f"❌ Error creating visualization: {str(e)}")

def interactive_mode(semantic_rag):
    """Run in interactive mode"""
    print("\n🎯 Interactive Mode")
    print("Commands:")
    print("  query <text>     - Ask a question")
    print("  status          - Show system status")
    print("  export <dir>    - Export system state")
    print("  visualize <file> - Create graph visualization")
    print("  quit            - Exit")
    print()
    
    while True:
        try:
            command = input("semantic_rag> ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            
            parts = command.split(' ', 1)
            cmd = parts[0].lower()
            
            if cmd == 'query' and len(parts) > 1:
                query_command(semantic_rag, parts[1])
            
            elif cmd == 'status':
                status = semantic_rag.get_system_status()
                print(json.dumps(status, indent=2, default=str))
            
            elif cmd == 'export' and len(parts) > 1:
                export_command(semantic_rag, parts[1])
            
            elif cmd == 'visualize' and len(parts) > 1:
                visualize_command(semantic_rag, parts[1])
            
            else:
                print("Invalid command. Type 'quit' to exit.")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Semantic RAG System with Relationship Graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process files and build system
  python main.py --process-files files/

  # Query the system
  python main.py --query "What are the main claims fields?"

  # Export system state
  python main.py --export-state exports/

  # Create visualization
  python main.py --visualize-graph graph.html

  # Interactive mode
  python main.py --interactive
        """
    )
    
    parser.add_argument('--process-files', metavar='DIR',
                        help='Process files from directory')
    parser.add_argument('--query', metavar='TEXT',
                        help='Query the system')
    parser.add_argument('--export-state', metavar='DIR',
                        help='Export system state to directory')
    parser.add_argument('--visualize-graph', metavar='FILE',
                        help='Create graph visualization')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--no-embeddings', action='store_true',
                        help='Skip embedding generation when processing files')
    parser.add_argument('--no-graph', action='store_true',
                        help='Skip graph building when processing files')
    parser.add_argument('--n-results', type=int, default=5,
                        help='Number of results for queries (default: 5)')
    
    args = parser.parse_args()
    
    if not any([args.process_files, args.query, args.export_state, 
                args.visualize_graph, args.interactive]):
        parser.print_help()
        return
    
    # Load environment
    databricks_key = load_environment()
    
    # Initialize system
    try:
        semantic_rag = initialize_system(databricks_key)
    except Exception as e:
        print(f"❌ Failed to initialize system: {str(e)}")
        return
    
    # Execute commands
    if args.process_files:
        success = process_files_command(
            semantic_rag,
            args.process_files,
            generate_embeddings=not args.no_embeddings,
            build_graph=not args.no_graph
        )
        if not success:
            return
    
    if args.query:
        query_command(semantic_rag, args.query, args.n_results)
    
    if args.export_state:
        export_command(semantic_rag, args.export_state)
    
    if args.visualize_graph:
        visualize_command(semantic_rag, args.visualize_graph)
    
    if args.interactive:
        interactive_mode(semantic_rag)

if __name__ == "__main__":
    main()
