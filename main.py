#!/usr/bin/env python3
"""
Semantic RAG System Main Entry Point

This script implements a comprehensive semantic RAG (Retrieval-Augmented        # Still initialize nodes even if skipping graph building
        semantic_rag._initialize_graph_nodes()
        
        # Auto-export processed documents for relationship building
        print("\n💾 Auto-exporting processed documents...")
        os.makedirs("exports", exist_ok=True)
        export_paths = semantic_rag.export_system_state("exports")
        print(f"✅ Documents exported to exports/ for relationship building")
        
        # Show system statuseration) system
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

# Setup clean logging first
from logging_config import setup_clean_logging
setup_clean_logging()

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
            # Only build basic node structure, skip old semantic relationships
            semantic_rag._initialize_graph_nodes()
            print("✅ Graph nodes initialized (use --build-relationships for field mappings)")
            
            # Show basic stats
            status = semantic_rag.get_system_status()
            print(f"   - Graph nodes: {status['relationship_graph']['num_nodes']}")
            print(f"   - Graph edges: {status['relationship_graph']['num_edges']}")
        else:
            # Still initialize nodes even if skipping graph building
            semantic_rag._initialize_graph_nodes()
        
        # Auto-export processed documents for relationship building
        print("\n💾 Auto-exporting processed documents...")
        os.makedirs("exports", exist_ok=True)
        export_paths = semantic_rag.export_system_state("exports")
        print(f"✅ Documents exported to exports/ for relationship building")
        
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

def build_relationships_command(semantic_rag):
    """Build relationships between PDF key terms and mapping source columns"""
    logger.info("Building relationships between PDF key terms and mapping source columns")
    
    try:
        print("\n🔗 Building PDF-to-Mapping Relationships...")
        print("=" * 50)
        
        # Load existing processed documents if available
        if not semantic_rag.processed_documents:
            print("📂 Loading previously processed documents...")
            if semantic_rag.load_processed_documents():
                print(f"✅ Loaded {len(semantic_rag.processed_documents)} processed documents")
            else:
                print("⚠️  No processed documents found. Please run --process-files first.")
                return
        
        # Extract PDF chunks with key terms
        pdf_chunks = []
        mapping_rows = []
        
        for doc_id, doc in semantic_rag.processed_documents.items():
            if doc.get('type') == 'pdf_page':
                llm_analysis = doc.get('llm_analysis', {})
                
                # Try multiple sources for key terms
                key_terms = []
                
                # 1. From LLM analysis key_concepts
                if llm_analysis.get('key_concepts'):
                    key_terms.extend(llm_analysis['key_concepts'])
                
                # 2. From LLM analysis fields (field names are key terms)
                if llm_analysis.get('fields'):
                    for field in llm_analysis['fields']:
                        if field.get('field_name'):
                            key_terms.append(field['field_name'])
                
                # 3. From metadata directly
                metadata = doc.get('metadata', {})
                if metadata.get('key_concepts'):
                    key_terms.extend(metadata['key_concepts'])
                if metadata.get('key_terms'):
                    key_terms.extend(metadata['key_terms'])
                
                # 4. Fallback to table name and functional area
                if llm_analysis.get('table_name'):
                    key_terms.append(llm_analysis['table_name'])
                if llm_analysis.get('functional_area'):
                    key_terms.append(llm_analysis['functional_area'])
                
                # Remove duplicates and empty terms
                key_terms = list(set([term.strip() for term in key_terms if term and term.strip()]))
                
                if key_terms:
                    pdf_chunks.append({
                        'id': doc_id,
                        'key_terms': key_terms,
                        'source_file': doc.get('source_file', 'unknown'),
                        'page_number': doc.get('metadata', {}).get('page_number', 'unknown'),
                        'llm_analysis': llm_analysis
                    })
            
            elif doc.get('type') == 'csv_row' and doc.get('metadata', {}).get('_mapping_type') == 'data_mapping':
                # Try multiple possible source column field names
                metadata = doc.get('metadata', {})
                source_column = None
                
                # Try different variations of source column names
                possible_source_keys = [
                    'source_column', 'Source Column', 'source_field', 'Source Field',
                    '\ufeffSource\xa0Field',  # Handle BOM and non-breaking space
                    'Source\xa0Field',       # Handle non-breaking space
                    '\ufeffSource Field',    # Handle BOM with regular space
                    'Source Field'           # Regular space
                ]
                
                for key in possible_source_keys:
                    if key in metadata and metadata[key]:
                        source_column = metadata[key]
                        break
                
                if source_column:
                    mapping_rows.append({
                        'id': doc_id,
                        'source_column': source_column,
                        'source_table': metadata.get('source_table', ''),
                        'target_field': metadata.get('target_field', ''),
                        'source_file': doc.get('source_file', 'unknown')
                    })
        
        print(f"📄 Found {len(pdf_chunks)} PDF chunks with key terms")
        print(f"📊 Found {len(mapping_rows)} mapping rows with source columns")
        
        # Debug information
        if len(pdf_chunks) > 0:
            sample_chunk = pdf_chunks[0]
            print(f"🔍 Sample PDF key terms: {sample_chunk['key_terms'][:10]}")  # Show first 10
        if len(mapping_rows) > 0:
            sample_mapping = mapping_rows[0]
            print(f"🔍 Sample mapping columns: {[row['source_column'] for row in mapping_rows[:5]]}")
        
        if not pdf_chunks:
            print("⚠️  No PDF chunks with key terms found. Make sure files have been processed.")
            
            # Show debug info about what PDF documents we have
            pdf_docs = [doc for doc_id, doc in semantic_rag.processed_documents.items() 
                       if doc.get('type') in ['pdf_page', 'pdf']]
            print(f"🔍 Debug: Found {len(pdf_docs)} PDF documents in processed_documents")
            
            if pdf_docs:
                sample_pdf = pdf_docs[0]
                print(f"🔍 Debug: Sample PDF doc type: {sample_pdf.get('type')}")
                print(f"🔍 Debug: Sample PDF metadata keys: {list(sample_pdf.get('metadata', {}).keys())}")
                if sample_pdf.get('llm_analysis'):
                    print(f"🔍 Debug: Sample LLM analysis keys: {list(sample_pdf.get('llm_analysis', {}).keys())}")
                else:
                    print("🔍 Debug: No llm_analysis found in sample PDF")
            return
        
        if not mapping_rows:
            print("⚠️  No mapping rows found. Make sure CSV files have been processed.")
            return
        
        # Build relationships using the new method
        relationships_built = 0
        
        for pdf_chunk in pdf_chunks:
            key_terms = pdf_chunk.get('key_terms', [])
            pdf_id = pdf_chunk.get('id')
            
            for mapping_row in mapping_rows:
                source_col = mapping_row.get('source_column')
                mapping_id = mapping_row.get('id')
                
                if not source_col or not pdf_id or not mapping_id:
                    continue
                
                # Check for exact match (case-insensitive)
                for term in key_terms:
                    if term and source_col and term.strip().lower() == source_col.strip().lower():
                        # High confidence (exact match)
                        semantic_rag.relationship_graph.add_relationship(
                            mapping_id, pdf_id,
                            relationship_type='field_to_pdf_keyterm',
                            weight=1.0,
                            source_column=source_col,
                            source_table=mapping_row.get('source_table', ''),
                            target_field=mapping_row.get('target_field', ''),
                            matched_term=term,
                            confidence=1.0,
                            description=f"Mapping source column '{source_col}' matches PDF key term '{term}'"
                        )
                        relationships_built += 1
                        print(f"   ✅ {source_col} ↔ {pdf_chunk.get('source_file', 'unknown')} (page {pdf_chunk.get('page_number', 'unknown')})")
                        break  # Only create one relationship per mapping-PDF pair
        
        print(f"\n🎉 Relationship building completed!")
        print(f"   📈 Built {relationships_built} high-confidence relationships")
        print(f"   🔗 Total nodes in graph: {semantic_rag.relationship_graph.graph.number_of_nodes()}")
        print(f"   🔗 Total edges in graph: {semantic_rag.relationship_graph.graph.number_of_edges()}")
        
        if relationships_built == 0:
            print("\n💡 No exact matches found between PDF key terms and mapping source columns.")
            print("   This could mean:")
            print("   • PDF key terms don't match source column names exactly")
            print("   • Different naming conventions are used")
            print("   • Consider reviewing the key terms extraction logic")
            
            # Show some examples for debugging
            if pdf_chunks and mapping_rows:
                print(f"\n🔍 Sample PDF key terms: {pdf_chunks[0]['key_terms'][:5]}")
                print(f"🔍 Sample mapping columns: {[row['source_column'] for row in mapping_rows[:5]]}")
        
    except Exception as e:
        logger.error(f"Error building relationships: {str(e)}")
        print(f"❌ Error building relationships: {str(e)}")

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

  # Build relationships between PDF terms and mapping columns
  python main.py --build-relationships

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
    parser.add_argument('--build-relationships', action='store_true',
                        help='Build relationships between PDF key terms and mapping source columns')
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
                args.visualize_graph, args.interactive, args.build_relationships]):
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
    
    if args.build_relationships:
        build_relationships_command(semantic_rag)
    
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
