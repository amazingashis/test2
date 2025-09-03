#!/usr/bin/env python3
"""
Enhanced Semantic RAG System with AI-Powered Relationship Analysis
Advanced main CLI interface with comprehensive features
"""

import os
import sys
import time
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime

# Import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.semantic_rag import SemanticRAG
from src.vectordb.vectordb_client import VectorDBClient
from src.models.databricks_models import DatabricksEmbedder, MetaLlama370BInstruct, ClaudeSonnet4
from src.utils.file_parsers import process_files_from_directory

def setup_logging():
    """Setup enhanced logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'semantic_rag_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def print_banner():
    """Print enhanced system banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║              🧠 ENHANCED SEMANTIC RAG SYSTEM 🧠              ║
    ║                  with AI Relationship Analysis               ║
    ║──────────────────────────────────────────────────────────────║
    ║  🔍 Databricks BGE Embeddings                               ║
    ║  🤖 Meta Llama 3-3 70B AI Analysis                          ║
    ║  💬 Claude Sonnet 4 QA Engine                               ║
    ║  🕸️ NetworkX Knowledge Graphs                               ║
    ║  🚀 ChromaDB Vector Storage                                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_stats(stats: Dict[str, Any]):
    """Print enhanced system statistics"""
    print("\n📊 System Statistics:")
    print("═" * 50)
    print(f"📄 Documents Processed: {stats.get('total_documents', 0)}")
    print(f"🧩 Total Chunks: {stats.get('total_chunks', 0)}")
    print(f"🔗 Relationships Found: {stats.get('total_relationships', 0)}")
    print(f"🤖 AI Relationships: {stats.get('ai_relationships', 0)}")
    print(f"📐 Embeddings Generated: {stats.get('total_embeddings', 0)}")
    print(f"💾 Database Size: {stats.get('db_size_mb', 0):.2f} MB")
    if stats.get('processing_time'):
        print(f"⏱️ Processing Time: {stats['processing_time']:.2f}s")
    print("═" * 50)

class EnhancedSemanticRAGCLI:
    """Enhanced CLI interface for Semantic RAG System"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.rag_system = None
        self.start_time = None
        
    async def initialize_system(self, collection_name: str = "semantic_rag_enhanced") -> bool:
        """Initialize the enhanced RAG system"""
        try:
            print("🚀 Initializing Enhanced Semantic RAG System...")
            self.start_time = time.time()
            
            # Initialize components
            embedder = DatabricksEmbedder()
            qa_model = ClaudeSonnet4()
            relationship_model = MetaLlama370BInstruct()
            
            # Initialize vector database
            vector_client = VectorDBClient(collection_name=collection_name)
            
            # Initialize RAG system with AI relationship analysis
            self.rag_system = SemanticRAG(
                embedder=embedder,
                vector_client=vector_client,
                qa_model=qa_model,
                relationship_model=relationship_model,
                use_ai_relationships=True
            )
            
            print("✅ System initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            print(f"❌ Initialization failed: {e}")
            return False
    
    async def process_files_interactive(self, file_path: str) -> Dict[str, Any]:
        """Process files with enhanced progress tracking"""
        try:
            print(f"\n📁 Processing files from: {file_path}")
            print("🔄 Starting enhanced processing pipeline...")
            
            start_time = time.time()
            
            # Process files
            processed_files = process_files_from_directory(file_path)
            
            if not processed_files:
                print("❌ No files found to process")
                return {"success": False, "message": "No files found"}
            
            print(f"📄 Found {len(processed_files)} files to process")
            
            # Add documents with progress tracking
            results = []
            for i, (file_path, content, metadata) in enumerate(processed_files, 1):
                print(f"🔄 Processing file {i}/{len(processed_files)}: {Path(file_path).name}")
                
                try:
                    result = await self.rag_system.add_document(content, metadata)
                    results.append(result)
                    print(f"  ✅ Processed: {result.get('chunks_created', 0)} chunks created")
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    self.logger.error(f"Error processing {file_path}: {e}")
            
            # Build enhanced relationship graph with AI analysis
            print("\n🕸️ Building enhanced relationship graph with AI analysis...")
            relationship_stats = await self.rag_system.build_relationship_graph(
                use_ai_analysis=True,
                optimize_graph=True
            )
            
            processing_time = time.time() - start_time
            
            # Generate comprehensive statistics
            stats = await self.get_system_stats()
            stats.update({
                'processing_time': processing_time,
                'files_processed': len(processed_files),
                'ai_relationships': relationship_stats.get('ai_relationships_added', 0),
                'optimization_applied': relationship_stats.get('edges_consolidated', 0)
            })
            
            print_stats(stats)
            
            return {
                "success": True,
                "stats": stats,
                "relationship_stats": relationship_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error in processing: {e}")
            print(f"❌ Processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def query_interactive(self):
        """Enhanced interactive query interface"""
        print("\n💬 Enhanced Interactive Query Mode")
        print("Type 'exit' to quit, 'stats' for statistics, 'graph' for graph info")
        print("=" * 60)
        
        while True:
            try:
                query = input("\n🔍 Enter your question: ").strip()
                
                if query.lower() == 'exit':
                    break
                elif query.lower() == 'stats':
                    stats = await self.get_system_stats()
                    print_stats(stats)
                    continue
                elif query.lower() == 'graph':
                    await self.show_graph_info()
                    continue
                elif not query:
                    continue
                
                print("🤔 Thinking...")
                start_time = time.time()
                
                # Enhanced query with metadata
                result = await self.rag_system.query(
                    query, 
                    top_k=5,
                    include_relationships=True,
                    use_ai_analysis=True
                )
                
                query_time = time.time() - start_time
                
                print("\n" + "="*60)
                print("🎯 Answer:")
                print("-" * 30)
                print(result['answer'])
                
                if result.get('sources'):
                    print(f"\n📚 Sources ({len(result['sources'])}):")
                    for i, source in enumerate(result['sources'][:3], 1):
                        source_name = source.get('source_file', 'Unknown').split('/')[-1]
                        print(f"  {i}. {source_name}")
                        if source.get('content'):
                            preview = source['content'][:100] + "..." if len(source['content']) > 100 else source['content']
                            print(f"     Preview: {preview}")
                
                if result.get('relationships'):
                    print(f"\n🔗 Related Concepts ({len(result['relationships'])}):")
                    for rel in result['relationships'][:3]:
                        print(f"  • {rel}")
                
                print(f"\n⏱️ Response time: {query_time:.2f}s")
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                self.logger.error(f"Error in query: {e}")
                print(f"❌ Query failed: {e}")
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            if not self.rag_system:
                return {}
            
            # Get vector database stats
            collection_stats = self.rag_system.vector_client.get_collection_stats()
            
            # Get relationship graph stats
            graph_stats = {}
            if hasattr(self.rag_system, 'relationship_graph') and self.rag_system.relationship_graph:
                graph = self.rag_system.relationship_graph.graph
                graph_stats = {
                    'total_nodes': graph.number_of_nodes(),
                    'total_relationships': graph.number_of_edges(),
                    'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
                    'density': (2.0 * graph.number_of_edges()) / (graph.number_of_nodes() * (graph.number_of_nodes() - 1)) if graph.number_of_nodes() > 1 else 0
                }
            
            # Calculate database size
            db_path = Path("data/chroma_db")
            db_size_mb = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file()) / (1024 * 1024) if db_path.exists() else 0
            
            return {
                'total_documents': collection_stats.get('count', 0),
                'total_chunks': collection_stats.get('count', 0),
                'total_embeddings': collection_stats.get('count', 0),
                'db_size_mb': db_size_mb,
                **graph_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {}
    
    async def show_graph_info(self):
        """Display enhanced graph information"""
        try:
            if not self.rag_system or not self.rag_system.relationship_graph:
                print("❌ No relationship graph available")
                return
            
            graph = self.rag_system.relationship_graph.graph
            
            print("\n🕸️ Knowledge Graph Information:")
            print("=" * 40)
            print(f"📊 Nodes: {graph.number_of_nodes()}")
            print(f"🔗 Edges: {graph.number_of_edges()}")
            
            # Node type distribution
            node_types = {}
            for node, data in graph.nodes(data=True):
                node_type = data.get('node_type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            print("\n📈 Node Types:")
            for node_type, count in sorted(node_types.items()):
                print(f"  • {node_type}: {count}")
            
            # AI relationships
            ai_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('relationship_type', '').startswith('ai_')]
            print(f"\n🤖 AI-Discovered Relationships: {len(ai_edges)}")
            
            # Show visualization file info
            viz_file = Path("advanced_knowledge_graph.html")
            if viz_file.exists():
                print(f"\n🎨 Visualization available: {viz_file.name}")
                print("   Open in browser to explore the graph interactively")
            
        except Exception as e:
            self.logger.error(f"Error showing graph info: {e}")
            print(f"❌ Error displaying graph info: {e}")
    
    async def export_data(self, export_dir: str = "exports"):
        """Export all system data with enhanced formats"""
        try:
            export_path = Path(export_dir)
            export_path.mkdir(exist_ok=True)
            
            print(f"\n💾 Exporting system data to: {export_path}")
            
            # Export vector database
            print("🔄 Exporting vector database...")
            vector_data = self.rag_system.vector_client.export_data()
            
            with open(export_path / "vector_database.json", 'w') as f:
                json.dump(vector_data, f, indent=2, default=str)
            
            # Export relationship graph
            if self.rag_system.relationship_graph:
                print("🔄 Exporting relationship graph...")
                graph_data = self.rag_system.relationship_graph.export_graph()
                
                with open(export_path / "relationship_graph.json", 'w') as f:
                    json.dump(graph_data, f, indent=2, default=str)
            
            # Export system status
            print("🔄 Exporting system status...")
            stats = await self.get_system_stats()
            system_status = {
                'timestamp': datetime.now().isoformat(),
                'system_stats': stats,
                'processing_time': time.time() - self.start_time if self.start_time else 0,
                'version': '2.0.0-enhanced'
            }
            
            with open(export_path / "system_status.json", 'w') as f:
                json.dump(system_status, f, indent=2, default=str)
            
            print(f"✅ Export completed successfully!")
            print(f"   📁 Files saved to: {export_path.absolute()}")
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            print(f"❌ Export failed: {e}")

async def main():
    """Enhanced main function with comprehensive CLI"""
    parser = argparse.ArgumentParser(
        description="Enhanced Semantic RAG System with AI Relationship Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --files ./files                    # Process files and build graph
  %(prog)s --query                           # Interactive query mode
  %(prog)s --files ./files --export          # Process and export
  %(prog)s --stats                           # Show system statistics
        """
    )
    
    parser.add_argument('--files', '-f', type=str, help='Directory containing files to process')
    parser.add_argument('--query', '-q', action='store_true', help='Start interactive query mode')
    parser.add_argument('--export', '-e', action='store_true', help='Export system data')
    parser.add_argument('--stats', '-s', action='store_true', help='Show system statistics')
    parser.add_argument('--collection', '-c', type=str, default='semantic_rag_enhanced', 
                        help='Vector database collection name')
    parser.add_argument('--graph-only', action='store_true', help='Show graph information only')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Initialize CLI
    cli = EnhancedSemanticRAGCLI()
    
    # Initialize system
    if not await cli.initialize_system(args.collection):
        sys.exit(1)
    
    try:
        # Handle different modes
        if args.files:
            # Process files
            result = await cli.process_files_interactive(args.files)
            
            if result.get('success'):
                print("\n🎉 Processing completed successfully!")
                
                # Auto-export if requested
                if args.export:
                    await cli.export_data()
                
                # Auto-query mode if no other options
                if not args.export and not args.stats:
                    print("\n🚀 Launching interactive query mode...")
                    await cli.query_interactive()
            
        elif args.query:
            # Interactive query mode
            await cli.query_interactive()
            
        elif args.stats:
            # Show statistics
            stats = await cli.get_system_stats()
            print_stats(stats)
            
        elif args.graph_only:
            # Show graph info
            await cli.show_graph_info()
            
        elif args.export:
            # Export data
            await cli.export_data()
            
        else:
            # Default: show help and basic stats
            parser.print_help()
            print("\n📊 Current System Status:")
            stats = await cli.get_system_stats()
            if stats:
                print_stats(stats)
            else:
                print("No data found. Use --files to process documents first.")
        
    except KeyboardInterrupt:
        print("\n\n👋 System shutdown requested. Goodbye!")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
