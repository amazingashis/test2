import os
import sys
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.file_parsers import FileParser
from embeddings.embedder import EmbeddingGenerator, SemanticMatcher
from graph.relationship_graph import RelationshipGraph
from vectordb.vectordb_client import VectorDBClient
from models.databricks_models import ModelOrchestrator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SemanticRAG:
    """Main Semantic RAG system with relationship graphs and vector database"""
    
    def __init__(self, collection_name: str = "semantic_rag", 
                 persist_directory: str = "./chroma_db",
                 databricks_api_key: str = None):
        """
        Initialize Semantic RAG system.
        
        Args:
            collection_name (str): Name for vector database collection
            persist_directory (str): Directory to persist vector database
            databricks_api_key (str, optional): Databricks API key
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize components
        self.file_parser = FileParser()
        self.embedding_generator = EmbeddingGenerator()
        self.semantic_matcher = SemanticMatcher(self.embedding_generator)
        self.relationship_graph = RelationshipGraph()
        self.vector_db = VectorDBClient(collection_name, persist_directory)
        self.model_orchestrator = ModelOrchestrator(databricks_api_key)
        
        # Data storage
        self.processed_documents = {}
        self.field_mappings = {}
        self.relationships = []
        
        logger.info("Semantic RAG system initialized")
    
    def process_files(self, root_directory: str) -> Dict[str, Any]:
        """
        Process all files in the directory structure.
        
        Args:
            root_directory (str): Root directory to process
            
        Returns:
            Dict[str, Any]: Processing results summary
        """
        logger.info(f"Starting file processing from: {root_directory}")
        
        # Find all files
        found_files = self.file_parser.find_files(root_directory)
        
        processing_results = {
            "start_time": datetime.now().isoformat(),
            "root_directory": root_directory,
            "files_found": found_files,
            "processed_documents": {},
            "relationships_discovered": 0,
            "embeddings_generated": 0,
            "errors": []
        }
        
        # Process each file type
        document_id = 0
        
        # Process CSV files
        for csv_file in found_files['csv']:
            try:
                csv_data = self.file_parser.parse_csv(csv_file)
                doc_id = f"csv_{document_id}"
                
                # Process each row as a separate document
                for i, row in enumerate(csv_data):
                    row_doc_id = f"{doc_id}_row_{i}"
                    text_content = " ".join(f"{k}: {v}" for k, v in row.items() if v)
                    
                    self.processed_documents[row_doc_id] = {
                        'type': 'csv_row',
                        'source_file': csv_file,
                        'content': text_content,
                        'metadata': row,
                        'row_number': i
                    }
                
                # Also create a summary document for the entire CSV
                csv_summary = f"CSV file with {len(csv_data)} rows. Fields: {', '.join(csv_data[0].keys()) if csv_data else 'No data'}"
                
                self.processed_documents[doc_id] = {
                    'type': 'csv_summary',
                    'source_file': csv_file,
                    'content': csv_summary,
                    'metadata': {
                        'num_rows': len(csv_data),
                        'fields': list(csv_data[0].keys()) if csv_data else []
                    }
                }
                
                processing_results["processed_documents"][doc_id] = {
                    'type': 'csv',
                    'rows': len(csv_data),
                    'fields': list(csv_data[0].keys()) if csv_data else []
                }
                
                document_id += 1
                
            except Exception as e:
                error_msg = f"Error processing CSV file {csv_file}: {str(e)}"
                logger.error(error_msg)
                processing_results["errors"].append(error_msg)
        
        # Process PDF files
        for pdf_file in found_files['pdf']:
            try:
                pdf_data = self.file_parser.parse_pdf(pdf_file)
                doc_id = f"pdf_{document_id}"
                
                self.processed_documents[doc_id] = {
                    'type': 'pdf',
                    'source_file': pdf_file,
                    'content': pdf_data['text'],
                    'metadata': pdf_data['metadata'],
                    'chunks': pdf_data['chunks']
                }
                
                # Also process chunks as separate documents
                for i, chunk in enumerate(pdf_data['chunks']):
                    chunk_doc_id = f"{doc_id}_chunk_{i}"
                    self.processed_documents[chunk_doc_id] = {
                        'type': 'pdf_chunk',
                        'source_file': pdf_file,
                        'content': chunk,
                        'metadata': {**pdf_data['metadata'], 'chunk_index': i},
                        'parent_document': doc_id
                    }
                
                processing_results["processed_documents"][doc_id] = {
                    'type': 'pdf',
                    'pages': pdf_data['metadata'].get('num_pages', 0),
                    'chunks': len(pdf_data['chunks'])
                }
                
                document_id += 1
                
            except Exception as e:
                error_msg = f"Error processing PDF file {pdf_file}: {str(e)}"
                logger.error(error_msg)
                processing_results["errors"].append(error_msg)
        
        # Process Python files
        for py_file in found_files['python']:
            try:
                py_data = self.file_parser.parse_python_file(py_file)
                doc_id = f"python_{document_id}"
                
                self.processed_documents[doc_id] = {
                    'type': 'python',
                    'source_file': py_file,
                    'content': py_data['content'],
                    'metadata': {
                        'functions': py_data['functions'],
                        'classes': py_data['classes'],
                        'imports': py_data['imports'],
                        'variables': py_data['variables']
                    }
                }
                
                processing_results["processed_documents"][doc_id] = {
                    'type': 'python',
                    'functions': len(py_data['functions']),
                    'classes': len(py_data['classes']),
                    'imports': len(py_data['imports'])
                }
                
                document_id += 1
                
            except Exception as e:
                error_msg = f"Error processing Python file {py_file}: {str(e)}"
                logger.error(error_msg)
                processing_results["errors"].append(error_msg)
        
        processing_results["end_time"] = datetime.now().isoformat()
        processing_results["total_documents"] = len(self.processed_documents)
        
        logger.info(f"File processing completed. {len(self.processed_documents)} documents processed.")
        return processing_results
    
    def generate_embeddings(self, batch_size: int = 32, use_local: bool = False) -> Dict[str, Any]:
        """
        Generate embeddings for all processed documents.
        
        Args:
            batch_size (int): Batch size for embedding generation
            use_local (bool): Whether to use local embedding model
            
        Returns:
            Dict[str, Any]: Embedding generation results
        """
        logger.info("Starting embedding generation")
        
        # Prepare texts and document IDs
        texts = []
        doc_ids = []
        
        for doc_id, doc_data in self.processed_documents.items():
            content = doc_data.get('content', '')
            if content and len(content.strip()) > 0:
                texts.append(content)
                doc_ids.append(doc_id)
        
        if not texts:
            logger.warning("No texts found for embedding generation")
            return {"success": False, "message": "No texts to process"}
        
        # Generate embeddings
        embeddings = self.embedding_generator.get_batch_embeddings(
            texts, batch_size=batch_size, use_local=use_local
        )
        
        # Store embeddings in vector database
        documents_added = 0
        failed_embeddings = 0
        
        for doc_id, text, embedding in zip(doc_ids, texts, embeddings):
            if embedding is not None:
                doc_data = self.processed_documents[doc_id]
                metadata = {
                    'type': doc_data['type'],
                    'source_file': doc_data['source_file'],
                    **doc_data.get('metadata', {})
                }
                
                success = self.vector_db.add_embedding(doc_id, embedding, text, metadata)
                if success:
                    documents_added += 1
                else:
                    failed_embeddings += 1
            else:
                failed_embeddings += 1
        
        results = {
            "success": True,
            "total_texts": len(texts),
            "documents_added": documents_added,
            "failed_embeddings": failed_embeddings,
            "embedding_source": "databricks_bge_api"
        }
        
        logger.info(f"Embedding generation completed: {documents_added} documents added to vector DB")
        return results
    
    async def _analyze_semantic_relationships(self) -> None:
        """
        Use Meta Llama 3-3 70B to analyze deeper semantic relationships.
        """
        logger.info("Starting advanced relationship analysis with Meta Llama 3-3 70B")
        
        try:
            # Prepare context for analysis
            document_summaries = []
            doc_ids = list(self.processed_documents.keys())
            
            for doc_id in doc_ids[:10]:  # Analyze first 10 documents to avoid token limits
                doc = self.processed_documents[doc_id]
                summary = f"ID: {doc_id}\nType: {doc['type']}\nContent: {doc['content'][:200]}..."
                document_summaries.append(summary)
            
            # Create analysis prompt
            prompt = f"""
            Analyze the following documents and identify semantic relationships between them.
            Focus on conceptual connections, data flow relationships, and domain-specific associations.
            
            Documents:
            {chr(10).join(document_summaries)}
            
            For each relationship you identify, provide:
            1. Source document ID
            2. Target document ID  
            3. Relationship type (e.g., "maps_to", "contains", "references", "transforms")
            4. Relationship strength (0.0 to 1.0)
            5. Brief explanation
            
            Format your response as JSON array:
            [
                {{
                    "source": "doc_id_1",
                    "target": "doc_id_2", 
                    "type": "relationship_type",
                    "strength": 0.8,
                    "explanation": "brief explanation"
                }}
            ]
            """
            
            # Get analysis from Meta Llama 3-3 70B
            analysis_result = self.model_orchestrator.llama_model.generate_text(prompt)
            
            if analysis_result and analysis_result.get('success'):
                try:
                    # Parse the JSON response
                    import json
                    relationships = json.loads(analysis_result['response'])
                    
                    # Add AI-discovered relationships to graph
                    for rel in relationships:
                        if (rel['source'] in self.processed_documents and 
                            rel['target'] in self.processed_documents):
                            
                            self.relationship_graph.add_relationship(
                                rel['source'], rel['target'],
                                relationship_type=f"ai_{rel['type']}",
                                weight=rel['strength'],
                                metadata={
                                    'explanation': rel['explanation'],
                                    'ai_generated': True,
                                    'confidence': rel['strength']
                                }
                            )
                    
                    logger.info(f"Added {len(relationships)} AI-discovered relationships")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse AI relationship analysis: {e}")
                    
        except Exception as e:
            logger.warning(f"AI relationship analysis failed: {e}")
    
    async def _analyze_semantic_relationships(self) -> List[Dict[str, Any]]:
        """
        Advanced AI-powered relationship analysis using Meta Llama 3-3 70B.
        
        Returns:
            List[Dict[str, Any]]: AI-identified relationships
        """
        logger.info("Performing AI-powered relationship analysis")
        ai_relationships = []
        
        try:
            # Get document pairs for analysis
            doc_items = list(self.processed_documents.items())
            analysis_pairs = []
            
            # Sample pairs for AI analysis (to avoid token limits)
            import random
            if len(doc_items) > 50:
                # For large datasets, sample strategically
                for i in range(min(100, len(doc_items) * len(doc_items) // 20)):
                    doc1_id, doc1_data = random.choice(doc_items)
                    doc2_id, doc2_data = random.choice(doc_items)
                    if doc1_id != doc2_id:
                        analysis_pairs.append((doc1_id, doc1_data, doc2_id, doc2_data))
            else:
                # For smaller datasets, analyze all pairs
                for i, (doc1_id, doc1_data) in enumerate(doc_items):
                    for j, (doc2_id, doc2_data) in enumerate(doc_items[i+1:], i+1):
                        analysis_pairs.append((doc1_id, doc1_data, doc2_id, doc2_data))
            
            # Batch process pairs
            batch_size = 5
            for i in range(0, len(analysis_pairs), batch_size):
                batch = analysis_pairs[i:i+batch_size]
                batch_relationships = await self._analyze_relationship_batch(batch)
                ai_relationships.extend(batch_relationships)
                
                # Prevent rate limiting
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"AI relationship analysis failed: {e}")
            
        logger.info(f"AI identified {len(ai_relationships)} advanced relationships")
        return ai_relationships
    
    async def _analyze_relationship_batch(self, batch: List[tuple]) -> List[Dict[str, Any]]:
        """
        Analyze a batch of document pairs for relationships using AI.
        
        Args:
            batch: List of (doc1_id, doc1_data, doc2_id, doc2_data) tuples
            
        Returns:
            List[Dict[str, Any]]: Identified relationships
        """
        relationships = []
        
        for doc1_id, doc1_data, doc2_id, doc2_data in batch:
            try:
                # Prepare content for analysis
                content1 = doc1_data.get('content', '')[:500]  # Limit content length
                content2 = doc2_data.get('content', '')[:500]
                
                # Create analysis prompt
                analysis_prompt = f"""
                Analyze the relationship between these two documents and identify any meaningful connections:

                Document 1 ({doc1_data.get('type', 'unknown')}):
                {content1}

                Document 2 ({doc2_data.get('type', 'unknown')}):
                {content2}

                Please identify if there are any of the following relationship types:
                1. Conceptual similarity (shared concepts, topics, or themes)
                2. Complementary information (one extends or complements the other)
                3. Sequential relationship (one follows from the other)
                4. Hierarchical relationship (one is a subset/superset of the other)
                5. Causal relationship (one causes or influences the other)

                Respond with ONLY a JSON object in this format:
                {{
                    "has_relationship": true/false,
                    "relationship_type": "conceptual|complementary|sequential|hierarchical|causal",
                    "confidence": 0.0-1.0,
                    "explanation": "brief explanation"
                }}
                """
                
                # Get AI analysis
                response = await self.model_orchestrator.generate_with_model(
                    "llama_model", analysis_prompt
                )
                
                # Parse AI response
                if response and isinstance(response, str):
                    try:
                        import json
                        import re
                        
                        # Extract JSON from response
                        json_match = re.search(r'\{.*\}', response, re.DOTALL)
                        if json_match:
                            analysis = json.loads(json_match.group())
                            
                            if (analysis.get('has_relationship', False) and 
                                analysis.get('confidence', 0) > 0.6):
                                
                                relationships.append({
                                    'source': doc1_id,
                                    'target': doc2_id,
                                    'type': f"ai_{analysis.get('relationship_type', 'unknown')}",
                                    'confidence': analysis.get('confidence', 0),
                                    'explanation': analysis.get('explanation', ''),
                                    'weight': analysis.get('confidence', 0)
                                })
                                
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.debug(f"Failed to parse AI response: {e}")
                        
            except Exception as e:
                logger.debug(f"Failed to analyze pair {doc1_id}-{doc2_id}: {e}")
                
        return relationships
    
    def build_relationship_graph(self, similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Build relationship graph based on semantic similarities and content analysis.
        
        Args:
            similarity_threshold (float): Threshold for similarity relationships
            
        Returns:
            Dict[str, Any]: Graph building results
        """
        logger.info("Building relationship graph")
        
        # Enhanced relationship analysis using Meta Llama 3-3 70B
        import asyncio
        try:
            asyncio.run(self._analyze_semantic_relationships())
        except Exception as e:
            logger.warning(f"AI relationship analysis failed, continuing with basic analysis: {e}")
        
        # Add documents as nodes
        for doc_id, doc_data in self.processed_documents.items():
            # Prepare node attributes, avoiding conflicts
            node_attrs = {
                'node_type': doc_data['type'],
                'source_file': doc_data['source_file'],
                'content_length': len(doc_data.get('content', ''))
            }
            
            # Add metadata, avoiding 'source_file' conflicts
            metadata = doc_data.get('metadata', {})
            for key, value in metadata.items():
                if key not in node_attrs:  # Avoid overwriting existing attributes
                    node_attrs[key] = value
            
            self.relationship_graph.add_node(doc_id, **node_attrs)
        
        # Find semantic similarities
        doc_list = list(self.processed_documents.values())
        content_relationships = self.semantic_matcher.find_content_relationships(
            doc_list, similarity_threshold
        )
        
        # Add similarity relationships to graph
        for rel in content_relationships:
            doc1_id = list(self.processed_documents.keys())[rel['doc1_id']]
            doc2_id = list(self.processed_documents.keys())[rel['doc2_id']]
            
            self.relationship_graph.add_relationship(
                doc1_id, doc2_id,
                relationship_type='semantic_similarity',
                weight=rel['similarity'],
                similarity_score=rel['similarity']
            )
        
        # Find field similarities for CSV documents
        csv_documents = [
            (doc_id, doc_data) for doc_id, doc_data in self.processed_documents.items()
            if doc_data['type'] in ['csv_row', 'csv_summary']
        ]
        
        if csv_documents:
            # Group CSV rows by source file
            csv_by_file = {}
            for doc_id, doc_data in csv_documents:
                source_file = doc_data['source_file']
                if source_file not in csv_by_file:
                    csv_by_file[source_file] = []
                csv_by_file[source_file].append((doc_id, doc_data))
            
            # Find field similarities within each CSV file
            for source_file, file_docs in csv_by_file.items():
                csv_rows = [doc_data.get('metadata', {}) for _, doc_data in file_docs 
                           if doc_data['type'] == 'csv_row']
                
                if csv_rows:
                    field_similarities = self.semantic_matcher.find_field_similarities(
                        csv_rows, similarity_threshold
                    )
                    
                    # Create field nodes and relationships
                    for sim in field_similarities:
                        field1_id = f"field_{source_file}_{sim['field1']}"
                        field2_id = f"field_{source_file}_{sim['field2']}"
                        
                        # Add field nodes if they don't exist
                        if field1_id not in self.relationship_graph.graph.nodes:
                            self.relationship_graph.add_node(
                                field1_id,
                                node_type='field',
                                field_name=sim['field1'],
                                source_file=source_file
                            )
                        
                        if field2_id not in self.relationship_graph.graph.nodes:
                            self.relationship_graph.add_node(
                                field2_id,
                                node_type='field',
                                field_name=sim['field2'],
                                source_file=source_file
                            )
                        
                        # Add similarity relationship
                        self.relationship_graph.add_relationship(
                            field1_id, field2_id,
                            relationship_type='field_similarity',
                            weight=sim['similarity'],
                            similarity_score=sim['similarity']
                        )
        
        # Add hierarchical relationships (chunks to parent documents)
        for doc_id, doc_data in self.processed_documents.items():
            if 'parent_document' in doc_data:
                parent_id = doc_data['parent_document']
                self.relationship_graph.add_relationship(
                    parent_id, doc_id,
                    relationship_type='contains',
                    weight=1.0
                )
        
        # Add file-based relationships (documents from same source)
        docs_by_source = {}
        for doc_id, doc_data in self.processed_documents.items():
            source = doc_data['source_file']
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc_id)
        
        for source, doc_ids in docs_by_source.items():
            if len(doc_ids) > 1:
                # Create source file node
                source_node_id = f"source_{hash(source)}"
                self.relationship_graph.add_node(
                    source_node_id,
                    node_type='source_file',
                    file_path=source,
                    document_count=len(doc_ids)
                )
                
                # Connect all documents to source
                for doc_id in doc_ids:
                    self.relationship_graph.add_relationship(
                        source_node_id, doc_id,
                        relationship_type='source_of',
                        weight=1.0
                    )
        
        # Enhanced AI relationship analysis with Meta Llama 3-3 70B
        if self.model_orchestrator:
            logger.info("Starting AI-powered relationship analysis...")
            doc_list = list(self.processed_documents.values())
            
            # Sample documents for AI analysis (to manage API costs)
            sample_size = min(15, len(doc_list))
            doc_sample = doc_list[:sample_size]
            
            # Analyze relationships with AI
            ai_relationships = self._analyze_ai_relationships(doc_sample)
            
            # Add AI-discovered relationships to graph
            if ai_relationships:
                self._add_ai_relationships(ai_relationships)
                logger.info(f"Added {len(ai_relationships)} AI-discovered relationships")
        
        # Optimize the graph
        self.optimize_graph()
        
        # Get graph statistics
        graph_stats = self.relationship_graph.get_statistics()
        
        # Auto-save the graph for visualization
        try:
            import os
            os.makedirs("exports", exist_ok=True)
            self.relationship_graph.save_graph("exports/relationship_graph.json")
            logger.info("Relationship graph auto-saved to exports/relationship_graph.json")
        except Exception as e:
            logger.warning(f"Failed to auto-save graph: {e}")
        
        results = {
            "success": True,
            "graph_statistics": graph_stats,
            "content_relationships": len(content_relationships),
            "similarity_threshold": similarity_threshold
        }
        
        # Auto-save graph after building
        try:
            os.makedirs("exports", exist_ok=True)
            graph_file = "exports/relationship_graph.json"
            self.relationship_graph.save_graph(graph_file, format='json')
            logger.info(f"Graph automatically saved to {graph_file}")
        except Exception as e:
            logger.warning(f"Failed to auto-save graph: {e}")
        
        logger.info(f"Relationship graph built: {graph_stats['num_nodes']} nodes, {graph_stats['num_edges']} edges")
        return results

    def _analyze_ai_relationships(self, documents_sample: List[Dict]) -> List[Dict[str, Any]]:
        """
        Use Meta Llama 3-3 70B to analyze semantic relationships between documents.
        
        Args:
            documents_sample: Sample of documents to analyze
            
        Returns:
            List of discovered relationships
        """
        if not self.model_orchestrator or len(documents_sample) < 2:
            return []
            
        try:
            # Prepare document summary for AI analysis
            doc_summaries = []
            for doc in documents_sample[:10]:  # Limit to 10 docs for efficiency
                summary = {
                    'id': doc['doc_id'],
                    'type': doc.get('type', 'unknown'),
                    'content_preview': doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                    'source': doc.get('source_file', 'unknown')
                }
                doc_summaries.append(summary)
            
            # Create analysis prompt
            analysis_prompt = f"""Analyze the following {len(doc_summaries)} documents and identify semantic relationships between them.

Documents:
{json.dumps(doc_summaries, indent=2)}

Please identify:
1. Conceptual relationships (e.g., "claims processing" relates to "payment amounts")
2. Data flow relationships (e.g., "patient_id maps to beneficiary_id") 
3. Hierarchical relationships (e.g., "claim header contains claim details")
4. Temporal relationships (e.g., "admission_date precedes discharge_date")
5. Functional relationships (e.g., "diagnosis codes determine treatment codes")

Return your analysis as a JSON array of relationships in this format:
[
  {{
    "source_doc": "doc_id_1",
    "target_doc": "doc_id_2", 
    "relationship_type": "conceptual|dataflow|hierarchical|temporal|functional",
    "relationship_name": "descriptive name",
    "strength": 0.0-1.0,
    "explanation": "brief explanation"
  }}
]

Focus on meaningful, actionable relationships for healthcare claims data processing."""
            
            # Call Meta Llama 3-3 70B for relationship analysis
            logger.info("Analyzing document relationships with Meta Llama 3-3 70B...")
            ai_response = self.model_orchestrator.llama_model.analyze_relationships(
                analysis_prompt,
                max_tokens=2000
            )
            
            # Parse AI response
            try:
                relationships = json.loads(ai_response)
                if isinstance(relationships, list):
                    logger.info(f"AI discovered {len(relationships)} relationships")
                    return relationships
                else:
                    logger.warning("AI response was not a valid list of relationships")
                    return []
            except json.JSONDecodeError:
                logger.warning("Could not parse AI relationship analysis response")
                return []
                
        except Exception as e:
            logger.error(f"Error in AI relationship analysis: {str(e)}")
            return []
    
    def _add_ai_relationships(self, ai_relationships: List[Dict[str, Any]]):
        """
        Add AI-discovered relationships to the graph.
        
        Args:
            ai_relationships: List of relationships from AI analysis
        """
        for rel in ai_relationships:
            try:
                source_id = rel.get('source_doc')
                target_id = rel.get('target_doc')
                rel_type = rel.get('relationship_type', 'semantic')
                rel_name = rel.get('relationship_name', 'related_to')
                strength = float(rel.get('strength', 0.5))
                explanation = rel.get('explanation', '')
                
                if source_id and target_id and source_id in self.processed_documents and target_id in self.processed_documents:
                    self.relationship_graph.add_relationship(
                        source_id, target_id,
                        relationship_type=f"ai_{rel_type}",
                        weight=strength,
                        relationship_name=rel_name,
                        explanation=explanation,
                        discovered_by='meta_llama_3_70b'
                    )
                    
            except Exception as e:
                logger.warning(f"Error adding AI relationship: {str(e)}")
                continue
    
    def optimize_graph(self):
        """
        Optimize the relationship graph for better performance and visualization.
        """
        try:
            logger.info("Optimizing relationship graph...")
            
            # Remove redundant relationships
            removed_count = 0
            edges_to_remove = []
            
            for edge in self.relationship_graph.graph.edges(data=True):
                source, target, attrs = edge
                weight = attrs.get('weight', 0.0)
                
                # Remove weak relationships
                if weight < 0.1:
                    edges_to_remove.append((source, target))
                    removed_count += 1
            
            for source, target in edges_to_remove:
                if self.relationship_graph.graph.has_edge(source, target):
                    self.relationship_graph.graph.remove_edge(source, target)
            
            # Consolidate similar relationships
            consolidated_count = self._consolidate_relationships()
            
            logger.info(f"Graph optimization complete: removed {removed_count} weak edges, consolidated {consolidated_count} relationships")
            
        except Exception as e:
            logger.error(f"Error optimizing graph: {str(e)}")
    
    def _consolidate_relationships(self) -> int:
        """
        Consolidate similar relationships between the same nodes.
        
        Returns:
            Number of relationships consolidated
        """
        consolidation_count = 0
        node_pairs = {}
        
        # Group relationships by node pairs
        for edge in self.relationship_graph.graph.edges(data=True):
            source, target, attrs = edge
            pair_key = tuple(sorted([source, target]))
            
            if pair_key not in node_pairs:
                node_pairs[pair_key] = []
            node_pairs[pair_key].append((source, target, attrs))
        
        # Consolidate multiple relationships between same nodes
        for pair_key, relationships in node_pairs.items():
            if len(relationships) > 1:
                # Find the strongest relationship
                best_rel = max(relationships, key=lambda x: x[2].get('weight', 0.0))
                
                # Remove all relationships for this pair
                for source, target, _ in relationships:
                    if self.relationship_graph.graph.has_edge(source, target):
                        self.relationship_graph.graph.remove_edge(source, target)
                
                # Add back the consolidated relationship
                source, target, attrs = best_rel
                consolidated_attrs = attrs.copy()
                consolidated_attrs['consolidated'] = True
                consolidated_attrs['original_count'] = len(relationships)
                
                self.relationship_graph.add_relationship(
                    source, target,
                    **consolidated_attrs
                )
                
                consolidation_count += len(relationships) - 1
        
        return consolidation_count
    
    def query(self, query_text: str, n_results: int = 5, include_graph_context: bool = True) -> Dict[str, Any]:
        """
        Query the semantic RAG system.
        
        Args:
            query_text (str): Query text
            n_results (int): Number of results to return
            include_graph_context (bool): Whether to include graph-based context
            
        Returns:
            Dict[str, Any]: Query results with context and answer
        """
        logger.info(f"Processing query: {query_text[:100]}...")
        
        # Generate embedding for query
        query_embedding = self.embedding_generator.get_embedding(query_text, use_local=True)
        
        if query_embedding is None:
            return {
                "success": False,
                "message": "Failed to generate query embedding",
                "query": query_text
            }
        
        # Search vector database
        search_results = self.vector_db.query(
            query_embedding=query_embedding,
            n_results=n_results,
            include_distances=True
        )
        
        # Gather context from search results
        context_documents = []
        for i, doc_id in enumerate(search_results['ids']):
            doc_info = {
                'id': doc_id,
                'content': search_results['documents'][i],
                'metadata': search_results['metadatas'][i],
                'similarity': 1 - search_results['distances'][i] if 'distances' in search_results else 0.0
            }
            context_documents.append(doc_info)
        
        # Add graph-based context if requested
        graph_context = []
        if include_graph_context and search_results['ids']:
            for doc_id in search_results['ids'][:3]:  # Top 3 results
                if doc_id in self.relationship_graph.graph.nodes:
                    # Find related nodes
                    related_nodes = self.relationship_graph.get_neighbors(doc_id)
                    for related_id in related_nodes[:5]:  # Limit to 5 related nodes
                        if related_id in self.processed_documents:
                            related_doc = self.processed_documents[related_id]
                            graph_context.append({
                                'id': related_id,
                                'type': related_doc['type'],
                                'content': related_doc['content'][:300],  # Truncate
                                'relationship': 'related_to',
                                'source': related_doc['source_file']
                            })
        
        # Prepare context for answer generation
        full_context = ""
        for doc in context_documents:
            full_context += f"\n--- Document {doc['id']} (similarity: {doc['similarity']:.3f}) ---\n"
            full_context += doc['content'][:1000]  # Truncate for context
        
        if graph_context:
            full_context += "\n\n--- Related Documents from Graph ---\n"
            for doc in graph_context:
                full_context += f"\n{doc['content']}\n"
        
        # Generate answer using Claude
        answer = self.model_orchestrator.claude_model.answer_question(
            question=query_text,
            context=full_context,
            max_tokens=800
        )
        
        results = {
            "success": True,
            "query": query_text,
            "answer": answer,
            "context_documents": context_documents,
            "graph_context": graph_context,
            "total_context_length": len(full_context),
            "search_results_count": len(search_results['ids'])
        }
        
        logger.info(f"Query processed successfully. Answer length: {len(answer) if answer else 0}")
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            Dict[str, Any]: System status information
        """
        vector_db_stats = self.vector_db.get_collection_stats()
        graph_stats = self.relationship_graph.get_statistics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "processed_documents": len(self.processed_documents),
            "vector_database": vector_db_stats,
            "relationship_graph": graph_stats,
            "components": {
                "file_parser": "ready",
                "embedding_generator": "ready",
                "semantic_matcher": "ready",
                "relationship_graph": "ready",
                "vector_database": "ready",
                "model_orchestrator": "ready"
            }
        }
    
    def export_system_state(self, output_dir: str) -> Dict[str, str]:
        """
        Export the complete system state.
        
        Args:
            output_dir (str): Output directory for exports
            
        Returns:
            Dict[str, str]: Paths to exported files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        export_paths = {}
        
        # Export processed documents
        documents_file = os.path.join(output_dir, "processed_documents.json")
        with open(documents_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_documents, f, indent=2, ensure_ascii=False, default=str)
        export_paths['documents'] = documents_file
        
        # Export relationship graph
        graph_file = os.path.join(output_dir, "relationship_graph.json")
        self.relationship_graph.save_graph(graph_file, format='json')
        export_paths['graph'] = graph_file
        
        # Export vector database
        vectordb_file = os.path.join(output_dir, "vector_database.json")
        self.vector_db.export_collection(vectordb_file)
        export_paths['vectordb'] = vectordb_file
        
        # Export system status
        status_file = os.path.join(output_dir, "system_status.json")
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(self.get_system_status(), f, indent=2, default=str)
        export_paths['status'] = status_file
        
        logger.info(f"System state exported to: {output_dir}")
        return export_paths
    
    def visualize_graph(self, output_file: str = None, interactive: bool = True) -> None:
        """
        Create visualization of the relationship graph.
        
        Args:
            output_file (str, optional): Output file path
            interactive (bool): Whether to create interactive visualization
        """
        self.relationship_graph.visualize_graph(
            output_file=output_file,
            interactive=interactive
        )
