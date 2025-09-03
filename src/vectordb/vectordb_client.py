import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBClient:
    """Client for managing vector database operations using ChromaDB"""
    
    def __init__(self, collection_name: str = "semantic_rag", persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB client.
        
        Args:
            collection_name (str): Name of the collection
            persist_directory (str): Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            logger.info(f"ChromaDB client initialized with persist directory: {persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            # Fallback to in-memory client
            self.client = chromadb.Client()
            logger.warning("Using in-memory ChromaDB client as fallback")
        
        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Semantic RAG collection for document embeddings"}
            )
            logger.info(f"Collection '{collection_name}' ready")
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata to ensure all values are ChromaDB-compatible types.
        
        Args:
            metadata (Dict[str, Any]): Original metadata
            
        Returns:
            Dict[str, Any]: Sanitized metadata with only str, int, float, bool, or None values
        """
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                sanitized[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                sanitized[key] = ", ".join(str(item) for item in value)
            elif isinstance(value, dict):
                # Convert dicts to JSON strings
                sanitized[key] = str(value)
            else:
                # Convert other types to strings
                sanitized[key] = str(value)
        return sanitized
    
    def add_embedding(self, doc_id: str, embedding: List[float], 
                     document: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add an embedding to the collection.
        
        Args:
            doc_id (str): Unique document ID
            embedding (List[float]): Document embedding vector
            document (str): Original document text
            metadata (Dict[str, Any], optional): Additional metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if metadata is None:
                metadata = {}
            
            # Sanitize metadata before adding
            metadata = self._sanitize_metadata(metadata)
            
            # Add timestamp
            metadata['timestamp'] = datetime.now().isoformat()
            metadata['doc_length'] = len(document)
            
            self.collection.add(
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.debug(f"Added embedding for document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding embedding for {doc_id}: {str(e)}")
            return False
    
    def add_batch_embeddings(self, doc_ids: List[str], embeddings: List[List[float]], 
                           documents: List[str], metadatas: List[Dict[str, Any]] = None) -> int:
        """
        Add multiple embeddings to the collection.
        
        Args:
            doc_ids (List[str]): List of document IDs
            embeddings (List[List[float]]): List of embedding vectors
            documents (List[str]): List of document texts
            metadatas (List[Dict[str, Any]], optional): List of metadata dictionaries
            
        Returns:
            int: Number of successfully added embeddings
        """
        if len(doc_ids) != len(embeddings) or len(doc_ids) != len(documents):
            logger.error("Mismatch in input list lengths")
            return 0
        
        if metadatas is None:
            metadatas = [{}] * len(doc_ids)
        
        # Add timestamps and doc lengths
        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            metadata = self._sanitize_metadata(metadata)
            metadata['timestamp'] = datetime.now().isoformat()
            metadata['doc_length'] = len(doc)
            metadatas[i] = metadata
        
        try:
            # Filter out None embeddings
            valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
            
            if not valid_indices:
                logger.warning("No valid embeddings to add")
                return 0
            
            valid_doc_ids = [doc_ids[i] for i in valid_indices]
            valid_embeddings = [embeddings[i] for i in valid_indices]
            valid_documents = [documents[i] for i in valid_indices]
            valid_metadatas = [metadatas[i] for i in valid_indices]
            
            self.collection.add(
                embeddings=valid_embeddings,
                documents=valid_documents,
                metadatas=valid_metadatas,
                ids=valid_doc_ids
            )
            
            logger.info(f"Added {len(valid_indices)} embeddings to collection")
            return len(valid_indices)
            
        except Exception as e:
            logger.error(f"Error adding batch embeddings: {str(e)}")
            return 0
    
    def query(self, query_embedding: List[float], n_results: int = 5, 
             where: Dict[str, Any] = None, include_distances: bool = True) -> Dict[str, Any]:
        """
        Query the collection for similar embeddings.
        
        Args:
            query_embedding (List[float]): Query embedding vector
            n_results (int): Number of results to return
            where (Dict[str, Any], optional): Metadata filter conditions
            include_distances (bool): Whether to include similarity distances
            
        Returns:
            Dict[str, Any]: Query results containing IDs, documents, metadata, and optionally distances
        """
        try:
            include_list = ["documents", "metadatas"]
            if include_distances:
                include_list.append("distances")
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=include_list
            )
            
            # Flatten results (ChromaDB returns lists of lists)
            flattened_results = {
                'ids': results['ids'][0] if results['ids'] else [],
                'documents': results['documents'][0] if results['documents'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else []
            }
            
            if include_distances and 'distances' in results:
                flattened_results['distances'] = results['distances'][0]
            
            logger.debug(f"Query returned {len(flattened_results['ids'])} results")
            return flattened_results
            
        except Exception as e:
            logger.error(f"Error querying collection: {str(e)}")
            return {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}
    
    def query_by_text(self, query_text: str, n_results: int = 5, 
                     where: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Query the collection using text (requires embedding generation).
        
        Args:
            query_text (str): Query text
            n_results (int): Number of results to return
            where (Dict[str, Any], optional): Metadata filter conditions
            
        Returns:
            Dict[str, Any]: Query results
        """
        # Note: This would require an embedding generator
        # For now, we'll use a simple text search as fallback
        try:
            # Use where clause to search in document text
            # This is a simplified text search - ideally use embedding similarity
            results = self.collection.get(
                where_document={"$contains": query_text},
                limit=n_results,
                include=["documents", "metadatas"]
            )
            
            return {
                'ids': results['ids'],
                'documents': results['documents'],
                'metadatas': results['metadatas']
            }
            
        except Exception as e:
            logger.error(f"Error in text query: {str(e)}")
            return {'ids': [], 'documents': [], 'metadatas': []}
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            doc_id (str): Document ID
            
        Returns:
            Optional[Dict[str, Any]]: Document data or None if not found
        """
        try:
            results = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'document': results['documents'][0] if results['documents'] else '',
                    'metadata': results['metadatas'][0] if results['metadatas'] else {},
                    'embedding': results['embeddings'][0] if results['embeddings'] else []
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {str(e)}")
            return None
    
    def update_document(self, doc_id: str, embedding: List[float] = None, 
                       document: str = None, metadata: Dict[str, Any] = None) -> bool:
        """
        Update an existing document.
        
        Args:
            doc_id (str): Document ID
            embedding (List[float], optional): New embedding
            document (str, optional): New document text
            metadata (Dict[str, Any], optional): New metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            update_data = {}
            
            if embedding is not None:
                update_data['embeddings'] = [embedding]
            if document is not None:
                update_data['documents'] = [document]
            if metadata is not None:
                metadata['timestamp'] = datetime.now().isoformat()
                update_data['metadatas'] = [metadata]
            
            if update_data:
                update_data['ids'] = [doc_id]
                self.collection.update(**update_data)
                logger.debug(f"Updated document: {doc_id}")
                return True
            else:
                logger.warning(f"No data to update for document: {doc_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {str(e)}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the collection.
        
        Args:
            doc_id (str): Document ID to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.debug(f"Deleted document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False
    
    def delete_collection(self) -> bool:
        """
        Delete the entire collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dict[str, Any]: Collection statistics
        """
        try:
            # Get all documents to count them
            results = self.collection.get(include=[])
            count = len(results['ids'])
            
            # Get sample metadata to analyze structure
            if count > 0:
                sample_results = self.collection.get(
                    limit=min(100, count),
                    include=["metadatas"]
                )
                
                # Analyze metadata keys
                metadata_keys = set()
                for metadata in sample_results['metadatas']:
                    if metadata:
                        metadata_keys.update(metadata.keys())
                
                stats = {
                    'total_documents': count,
                    'collection_name': self.collection_name,
                    'metadata_keys': list(metadata_keys),
                    'sample_size': len(sample_results['metadatas'])
                }
            else:
                stats = {
                    'total_documents': 0,
                    'collection_name': self.collection_name,
                    'metadata_keys': [],
                    'sample_size': 0
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                'total_documents': 0,
                'collection_name': self.collection_name,
                'metadata_keys': [],
                'error': str(e)
            }
    
    def find_similar_documents(self, doc_id: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Find documents similar to a given document.
        
        Args:
            doc_id (str): Reference document ID
            n_results (int): Number of similar documents to return
            
        Returns:
            Dict[str, Any]: Similar documents
        """
        try:
            # Get the reference document's embedding
            doc_data = self.get_document(doc_id)
            if not doc_data or not doc_data['embedding']:
                logger.error(f"Document {doc_id} not found or has no embedding")
                return {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}
            
            # Query for similar documents (excluding the original)
            results = self.query(
                query_embedding=doc_data['embedding'],
                n_results=n_results + 1,  # +1 to account for excluding original
                include_distances=True
            )
            
            # Filter out the original document
            filtered_results = {
                'ids': [],
                'documents': [],
                'metadatas': [],
                'distances': []
            }
            
            for i, result_id in enumerate(results['ids']):
                if result_id != doc_id:
                    filtered_results['ids'].append(result_id)
                    filtered_results['documents'].append(results['documents'][i])
                    filtered_results['metadatas'].append(results['metadatas'][i])
                    if 'distances' in results:
                        filtered_results['distances'].append(results['distances'][i])
            
            # Limit to requested number of results
            for key in filtered_results:
                filtered_results[key] = filtered_results[key][:n_results]
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error finding similar documents for {doc_id}: {str(e)}")
            return {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}
    
    def export_collection(self, output_file: str) -> bool:
        """
        Export the entire collection to a JSON file.
        
        Args:
            output_file (str): Output file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get all documents
            results = self.collection.get(
                include=["documents", "metadatas", "embeddings"]
            )
            
            export_data = {
                'collection_name': self.collection_name,
                'export_timestamp': datetime.now().isoformat(),
                'total_documents': len(results['ids']),
                'documents': []
            }
            
            for i, doc_id in enumerate(results['ids']):
                doc_data = {
                    'id': doc_id,
                    'document': results['documents'][i] if i < len(results['documents']) else '',
                    'metadata': results['metadatas'][i] if i < len(results['metadatas']) else {},
                    'embedding': results['embeddings'][i] if i < len(results['embeddings']) else []
                }
                export_data['documents'].append(doc_data)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Collection exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting collection: {str(e)}")
            return False
    
    def import_collection(self, input_file: str, overwrite: bool = False) -> bool:
        """
        Import documents from a JSON file.
        
        Args:
            input_file (str): Input file path
            overwrite (bool): Whether to overwrite existing documents
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            if overwrite:
                # Clear existing collection
                self.delete_collection()
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "Semantic RAG collection for document embeddings"}
                )
            
            documents = import_data.get('documents', [])
            
            # Batch import
            batch_size = 100
            imported_count = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                doc_ids = [doc['id'] for doc in batch]
                embeddings = [doc.get('embedding', []) for doc in batch]
                texts = [doc.get('document', '') for doc in batch]
                metadatas = [doc.get('metadata', {}) for doc in batch]
                
                # Filter out documents without embeddings
                valid_batch = [(doc_id, emb, text, meta) for doc_id, emb, text, meta 
                              in zip(doc_ids, embeddings, texts, metadatas) if emb]
                
                if valid_batch:
                    valid_ids, valid_embs, valid_texts, valid_metas = zip(*valid_batch)
                    count = self.add_batch_embeddings(
                        list(valid_ids), list(valid_embs), 
                        list(valid_texts), list(valid_metas)
                    )
                    imported_count += count
            
            logger.info(f"Imported {imported_count} documents from {input_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing collection: {str(e)}")
            return False
