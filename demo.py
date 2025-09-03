#!/usr/bin/env python3
"""
Demo script to showcase the Semantic RAG system functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag.semantic_rag import SemanticRAG

def demo():
    print("🚀 Semantic RAG System Demo")
    print("=" * 50)
    
    # Initialize the system
    print("\n1. Initializing Semantic RAG System...")
    rag = SemanticRAG(
        collection_name="demo_rag",
        persist_directory="./data/demo_chroma_db"
    )
    print("✅ System initialized")
    
    # Process files
    print("\n2. Processing files...")
    results = rag.process_files("files/")
    print(f"✅ Processed {results['total_documents']} documents")
    print(f"   - CSV files: {len(results['files_found']['csv'])}")
    print(f"   - PDF files: {len(results['files_found']['pdf'])}")
    print(f"   - Python files: {len(results['files_found']['python'])}")
    
    # Generate embeddings with local model
    print("\n3. Generating embeddings...")
    emb_results = rag.generate_embeddings(batch_size=8, use_local=True)
    if emb_results['success']:
        print(f"✅ Generated embeddings for {emb_results['documents_added']} documents")
    
    # Build relationship graph
    print("\n4. Building relationship graph...")
    graph_results = rag.build_relationship_graph(similarity_threshold=0.6)
    if graph_results['success']:
        stats = graph_results['graph_statistics']
        print(f"✅ Built graph with {stats['num_nodes']} nodes and {stats['num_edges']} edges")
        print(f"   - Graph density: {stats['density']:.3f}")
        print(f"   - Content relationships: {graph_results['content_relationships']}")
    
    # Demo queries without Claude API
    print("\n5. Testing semantic search...")
    test_queries = [
        "claim payment amount",
        "source field mapping",
        "total charge amount"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n5.{i} Query: '{query}'")
        
        # Get embedding for query
        query_embedding = rag.embedding_generator.get_embedding(query, use_local=True)
        if query_embedding:
            # Search vector database
            results = rag.vector_db.query(
                query_embedding=query_embedding,
                n_results=3,
                include_distances=True
            )
            
            print(f"   Found {len(results['ids'])} relevant documents:")
            for j, (doc_id, doc, distance) in enumerate(zip(results['ids'], results['documents'], results.get('distances', []))):
                similarity = 1 - distance if distance else 0
                print(f"   {j+1}. {doc_id} (similarity: {similarity:.3f})")
                print(f"      {doc[:100]}...")
    
    # System statistics
    print("\n6. System Statistics:")
    status = rag.get_system_status()
    print(f"   - Total documents: {status['processed_documents']}")
    print(f"   - Vector DB documents: {status['vector_database']['total_documents']}")
    print(f"   - Graph nodes: {status['relationship_graph']['num_nodes']}")
    print(f"   - Graph edges: {status['relationship_graph']['num_edges']}")
    
    # Show available fields from CSV
    print("\n7. Claims Data Fields Found:")
    csv_docs = [doc for doc in rag.processed_documents.values() if doc['type'] == 'csv_row']
    if csv_docs:
        # Get field names from first CSV row
        first_row = csv_docs[0].get('metadata', {})
        fields = [key for key in first_row.keys() if not key.startswith('_')]
        print(f"   Available fields ({len(fields)}):")
        for field in fields[:10]:  # Show first 10 fields
            print(f"   - {field}")
        if len(fields) > 10:
            print(f"   ... and {len(fields) - 10} more fields")
    
    print(f"\n🎉 Demo completed successfully!")
    print("\nKey Features Demonstrated:")
    print("✅ Multi-format file parsing (CSV, PDF, Python)")
    print("✅ Semantic embedding generation")
    print("✅ Relationship graph construction")  
    print("✅ Vector database storage and search")
    print("✅ Semantic similarity matching")
    print("\nTo test with API models, configure your .env file with:")
    print("- DATABRICKS_API_KEY (for BGE and Llama models)")
    print("- ANTHROPIC_API_KEY (for Claude question answering)")

if __name__ == "__main__":
    demo()
