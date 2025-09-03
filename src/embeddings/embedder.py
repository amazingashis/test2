import os
import requests
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import json
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Handles embedding generation using different models"""
    
    def __init__(self, model_type: str = "bge_large_en_v1_5"):
        """
        Initialize embedding generator.
        
        Args:
            model_type (str): Type of embedding model to use
        """
        self.model_type = model_type
        
        # Databricks API configuration
        self.endpoint_url = os.getenv("BGE_API_URL", "https://dbc-3735add4-1cb6.cloud.databricks.com/serving-endpoints/bge_large_en_v1_5/invocations")
        self.api_key = os.getenv("DATABRICKS_TOKEN")
        
        if not self.api_key:
            logger.warning("No Databricks API key provided. Embedding functionality will be limited.")
            
        logger.info("BGE embedding generator initialized (Databricks-only mode)")
    
    def get_embedding(self, text: str, use_local: bool = False) -> Optional[List[float]]:
        """
        Generate embedding for given text using Databricks BGE API.
        
        Args:
            text (str): Text to embed
            use_local (bool): Ignored - only Databricks API is used
            
        Returns:
            Optional[List[float]]: Embedding vector or None if failed
        """
        return self._get_databricks_embedding(text)
    
    def _get_databricks_embedding_silent(self, text: str) -> Optional[List[float]]:
        """
        Get embedding using Databricks BGE API without verbose logging.
        
        Args:
            text (str): Text to embed
            
        Returns:
            Optional[List[float]]: Embedding vector or None if failed
        """
        if not self.api_key:
            return None
            
        try:
            response = requests.post(
                self.endpoint_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": text,
                    "model": "bge-large-en-v1.5"
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if 'data' in result and result['data']:
                embedding = result['data'][0].get('embedding')
                return embedding if embedding else None
            else:
                return None
                
        except Exception:
            return None
    
    def _get_databricks_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding using Databricks BGE API.
        
        Args:
            text (str): Text to embed
            
        Returns:
            Optional[List[float]]: Embedding vector or None if failed
        """
        if not self.api_key:
            logger.error("Databricks API key not available")
            return None
            
        try:
            response = requests.post(
                self.endpoint_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": text,
                    "model": "bge-large-en-v1.5"
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if 'data' in result and result['data']:
                embedding = result['data'][0].get('embedding')
                if embedding:
                    logger.info(f"Generated embedding via Databricks BGE API (dim: {len(embedding)})")
                    return embedding
                else:
                    logger.error("No embedding data in Databricks response")
                    return None
            else:
                logger.error("Invalid response format from Databricks BGE API")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Databricks BGE API request failed: {e}")
            return None
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Error parsing Databricks BGE API response: {e}")
            return None
    
    def get_batch_embeddings(self, texts: List[str], batch_size: int = 32, use_local: bool = False, show_progress: bool = True) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Number of texts to process at once (ignored - processing individually)
            use_local (bool): Ignored - only Databricks API is used
            show_progress (bool): Whether to show progress percentage
            
        Returns:
            List[Optional[List[float]]]: List of embedding vectors
        """
        embeddings = []
        total_texts = len(texts)
        
        # Process individually for API calls
        for i, text in enumerate(texts):
            emb = self._get_databricks_embedding_silent(text) if show_progress else self.get_embedding(text, use_local=False)
            embeddings.append(emb)
            
            # Show progress every 10% or for small datasets every few items
            if show_progress and total_texts > 0:
                progress = ((i + 1) / total_texts) * 100
                if progress % 10 < (100 / total_texts) or i == total_texts - 1:
                    print(f"\r🔄 Generating embeddings: {progress:.1f}% ({i + 1}/{total_texts})", end="", flush=True)
        
        if show_progress:
            print()  # New line after progress
        
        successful_embeddings = len([e for e in embeddings if e is not None])
        logger.info(f"Generated {successful_embeddings} embeddings out of {total_texts} texts")
        return embeddings
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1 (List[float]): First embedding
            embedding2 (List[float]): Second embedding
            
        Returns:
            float: Cosine similarity score
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def find_similar_embeddings(self, query_embedding: List[float], embeddings: List[List[float]], threshold: float = 0.7) -> List[int]:
        """
        Find embeddings similar to query embedding.
        
        Args:
            query_embedding (List[float]): Query embedding
            embeddings (List[List[float]]): List of embeddings to compare
            threshold (float): Similarity threshold
            
        Returns:
            List[int]: Indices of similar embeddings
        """
        similar_indices = []
        
        for i, embedding in enumerate(embeddings):
            if embedding is not None:
                similarity = self.calculate_similarity(query_embedding, embedding)
                if similarity >= threshold:
                    similar_indices.append(i)
        
        return similar_indices


class SemanticMatcher:
    """Handles semantic matching and relationship detection"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        """
        Initialize semantic matcher.
        
        Args:
            embedding_generator (EmbeddingGenerator): Embedding generator instance
        """
        self.embedding_generator = embedding_generator
    
    def find_field_similarities(self, csv_data: List[Dict[str, Any]], similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Find similar fields across CSV data.
        
        Args:
            csv_data (List[Dict[str, Any]]): CSV data rows
            similarity_threshold (float): Threshold for similarity matching
            
        Returns:
            List[Dict[str, Any]]: List of field similarities
        """
        if not csv_data:
            return []
        
        # Get all unique field names
        all_fields = set()
        for row in csv_data:
            all_fields.update(row.keys())
        
        field_list = list(all_fields)
        
        # Generate embeddings for field names
        field_embeddings = self.embedding_generator.get_batch_embeddings(field_list, use_local=True)
        
        similarities = []
        
        for i, field1 in enumerate(field_list):
            if field_embeddings[i] is None:
                continue
                
            for j, field2 in enumerate(field_list[i+1:], i+1):
                if field_embeddings[j] is None:
                    continue
                
                similarity = self.embedding_generator.calculate_similarity(
                    field_embeddings[i], field_embeddings[j]
                )
                
                if similarity >= similarity_threshold:
                    similarities.append({
                        'field1': field1,
                        'field2': field2,
                        'similarity': similarity,
                        'type': 'field_similarity'
                    })
        
        return similarities
    
    def find_content_relationships(self, documents: List[Dict[str, Any]], similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find content relationships between documents.
        
        Args:
            documents (List[Dict[str, Any]]): List of documents with text content
            similarity_threshold (float): Threshold for similarity matching
            
        Returns:
            List[Dict[str, Any]]: List of content relationships
        """
        relationships = []
        
        # Extract text content from documents
        texts = []
        doc_ids = []
        
        for i, doc in enumerate(documents):
            if 'text' in doc and doc['text']:
                texts.append(doc['text'])
                doc_ids.append(i)
            elif 'content' in doc and doc['content']:
                texts.append(doc['content'])
                doc_ids.append(i)
        
        if len(texts) < 2:
            return relationships
        
        # Generate embeddings
        embeddings = self.embedding_generator.get_batch_embeddings(texts, use_local=True)
        
        # Find relationships
        for i, emb1 in enumerate(embeddings):
            if emb1 is None:
                continue
                
            for j, emb2 in enumerate(embeddings[i+1:], i+1):
                if emb2 is None:
                    continue
                
                similarity = self.embedding_generator.calculate_similarity(emb1, emb2)
                
                if similarity >= similarity_threshold:
                    relationships.append({
                        'doc1_id': doc_ids[i],
                        'doc2_id': doc_ids[j],
                        'similarity': similarity,
                        'type': 'content_similarity',
                        'doc1_info': documents[doc_ids[i]].get('metadata', {}),
                        'doc2_info': documents[doc_ids[j]].get('metadata', {})
                    })
        
        return relationships
