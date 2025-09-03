import os
import requests
import json
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabricksModelClient:
    """Base client for Databricks hosted models"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize Databricks model client.
        
        Args:
            api_key (str, optional): Databricks API key
            base_url (str, optional): Base URL for Databricks API
        """
        self.api_key = api_key or os.getenv("DATABRICKS_TOKEN")
        self.base_url = base_url or os.getenv("DATABRICKS_BASE_URL", "https://dbc-3735add4-1cb6.cloud.databricks.com")
        
        if not self.api_key:
            logger.warning("No Databricks API key provided. Some functionality may be limited.")
    
    def _make_request(self, endpoint: str, payload: Dict[str, Any], timeout: int = 30) -> Optional[Dict[str, Any]]:
        """
        Make API request to Databricks.
        
        Args:
            endpoint (str): API endpoint
            payload (Dict[str, Any]): Request payload
            timeout (int): Request timeout in seconds
            
        Returns:
            Optional[Dict[str, Any]]: Response data or None if failed
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            url = f"{self.base_url}/{endpoint}"
            
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error making API request: {str(e)}")
            return None


class BGELargeEnV15(DatabricksModelClient):
    """BGE Large EN v1.5 embedding model client"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize BGE model client.
        
        Args:
            api_key (str, optional): Databricks API key
        """
        super().__init__(api_key)
        self.model_endpoint = "serving-endpoints/bge_large_en_v1_5/invocations"
        self.max_input_length = 512
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for a single text.
        
        Args:
            text (str): Input text
            
        Returns:
            Optional[List[float]]: Embedding vector or None if failed
        """
        # Truncate text to max length
        truncated_text = text[:self.max_input_length]
        
        payload = {
            "inputs": [truncated_text]
        }
        
        response = self._make_request(self.model_endpoint, payload)
        
        if response and "embeddings" in response and len(response["embeddings"]) > 0:
            return response["embeddings"][0]
        else:
            return None
    
    def get_batch_embeddings(self, texts: List[str], batch_size: int = 32) -> List[Optional[List[float]]]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts (List[str]): List of input texts
            batch_size (int): Number of texts to process in each batch
            
        Returns:
            List[Optional[List[float]]]: List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Truncate texts in batch
            truncated_batch = [text[:self.max_input_length] for text in batch]
            
            payload = {
                "inputs": truncated_batch
            }
            
            response = self._make_request(self.model_endpoint, payload)
            
            if response and "embeddings" in response:
                all_embeddings.extend(response["embeddings"])
            else:
                # Add None for failed embeddings
                all_embeddings.extend([None] * len(batch))
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        return all_embeddings
    
    def calculate_similarity(self, text1: str, text2: str) -> Optional[float]:
        """
        Calculate similarity between two texts.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            Optional[float]: Similarity score or None if failed
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        if emb1 and emb2:
            import numpy as np
            vec1 = np.array(emb1)
            vec2 = np.array(emb2)
            
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        
        return None


class MetaLlama370BInstruct(DatabricksModelClient):
    """Meta Llama 3 70B Instruct model client"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Llama model client.
        
        Args:
            api_key (str, optional): Databricks API key
        """
        super().__init__(api_key)
        self.model_endpoint = "serving-endpoints/databricks-meta-llama-3-3-70b-instruct/invocations"
        self.max_tokens = 4096
    
    def generate_response(self, prompt: str, max_tokens: int = 1000, 
                         temperature: float = 0.7, system_prompt: str = None) -> Optional[str]:
        """
        Generate response from the model.
        
        Args:
            prompt (str): Input prompt
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            system_prompt (str, optional): System prompt
            
        Returns:
            Optional[str]: Generated response or None if failed
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = self._make_request(self.model_endpoint, payload)
        
        if response and "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        else:
            return None
    
    def analyze_relationships(self, text1: str, text2: str) -> Optional[Dict[str, Any]]:
        """
        Analyze relationships between two texts.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            Optional[Dict[str, Any]]: Relationship analysis or None if failed
        """
        system_prompt = """You are an expert at analyzing relationships between texts. 
        Analyze the relationship between the two provided texts and return a JSON response with:
        - relationship_type: type of relationship (e.g., 'similarity', 'mapping', 'dependency', 'hierarchy')
        - strength: relationship strength from 0.0 to 1.0
        - description: brief description of the relationship
        - key_connections: list of specific connections found"""
        
        prompt = f"""Analyze the relationship between these two texts:

TEXT 1:
{text1[:1000]}

TEXT 2:
{text2[:1000]}

Provide your analysis in JSON format."""
        
        response = self.generate_response(prompt, max_tokens=500, system_prompt=system_prompt)
        
        if response:
            try:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    # Fallback: parse response as text
                    return {
                        "relationship_type": "unknown",
                        "strength": 0.5,
                        "description": response[:200],
                        "key_connections": []
                    }
            except Exception as e:
                logger.error(f"Error parsing relationship analysis: {str(e)}")
                return None
        
        return None

    def analyze_relationships(self, analysis_prompt: str, max_tokens: int = 2000) -> Optional[str]:
        """
        Analyze relationships between multiple documents using AI.
        
        Args:
            analysis_prompt (str): Prompt containing documents and analysis instructions
            max_tokens (int): Maximum tokens for response
            
        Returns:
            Optional[str]: AI analysis response or None if failed
        """
        system_prompt = """You are an expert data analyst specializing in healthcare claims data and semantic relationships. 
        You excel at identifying meaningful connections, data flows, and conceptual relationships between documents.
        Always return valid JSON responses in the exact format requested."""
        
        try:
            response = self.generate_response(
                analysis_prompt,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=0.3  # Lower temperature for more consistent JSON output
            )
            return response
        except Exception as e:
            logger.error(f"Error in AI relationship analysis: {str(e)}")
            return None
    
    def generate_mapping(self, source_fields: List[str], target_schema: str) -> Optional[Dict[str, str]]:
        """
        Generate field mapping between source and target schemas.
        
        Args:
            source_fields (List[str]): List of source field names
            target_schema (str): Description of target schema
            
        Returns:
            Optional[Dict[str, str]]: Field mapping or None if failed
        """
        system_prompt = """You are an expert data mapper. Generate field mappings between source fields and target schema.
        Return a JSON object where keys are source fields and values are target field names or null if no mapping exists."""
        
        prompt = f"""Generate mapping for these source fields to the target schema:

SOURCE FIELDS:
{', '.join(source_fields)}

TARGET SCHEMA:
{target_schema}

Provide mapping in JSON format."""
        
        response = self.generate_response(prompt, max_tokens=1000, system_prompt=system_prompt)
        
        if response:
            try:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except Exception as e:
                logger.error(f"Error parsing mapping: {str(e)}")
        
        return None


class ClaudeSonet4(DatabricksModelClient):
    """Claude Sonet 4 model client via Databricks"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Claude client via Databricks.
        
        Args:
            api_key (str, optional): Databricks API key
        """
        super().__init__(api_key)
        self.model_endpoint = "serving-endpoints/databricks-claude-sonnet-4/invocations"
        
        if not self.api_key:
            logger.warning("No Databricks API key provided. Claude functionality will be limited.")
    
    def answer_question(self, question: str, context: str, max_tokens: int = 1000) -> Optional[str]:
        """
        Answer a question given context using Claude via Databricks.
        
        Args:
            question (str): Question to answer
            context (str): Context information
            max_tokens (int): Maximum tokens in response
            
        Returns:
            Optional[str]: Answer or None if failed
        """
        try:
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Based on the following context, please answer the question:

CONTEXT:
{context[:3000]}

QUESTION:
{question}

Please provide a clear, accurate answer based on the context provided."""
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = self._make_request(self.model_endpoint, payload)
            
            if response and "choices" in response and len(response["choices"]) > 0:
                return response["choices"][0]["message"]["content"]
            elif response and "content" in response:
                # Handle different response format
                if isinstance(response["content"], list) and len(response["content"]) > 0:
                    return response["content"][0].get("text", "")
                return str(response["content"])
            else:
                logger.error(f"Unexpected Claude response format: {response}")
                return None
            
        except Exception as e:
            logger.error(f"Error calling Claude via Databricks: {str(e)}")
            return None
    
    def summarize_document(self, document: str, max_tokens: int = 500) -> Optional[str]:
        """
        Summarize a document using Claude via Databricks.
        
        Args:
            document (str): Document to summarize
            max_tokens (int): Maximum tokens in summary
            
        Returns:
            Optional[str]: Summary or None if failed
        """
        return self.answer_question(
            "Please provide a comprehensive summary of this document, highlighting key points and main topics.",
            document,
            max_tokens
        )
    
    def extract_entities(self, text: str) -> Optional[List[str]]:
        """
        Extract named entities from text using Claude via Databricks.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Optional[List[str]]: List of entities or None if failed
        """
        question = "Extract and list all named entities (people, organizations, locations, dates, etc.) from this text. Return as a comma-separated list."
        
        response = self.answer_question(question, text, max_tokens=300)
        
        if response:
            # Parse comma-separated entities
            entities = [entity.strip() for entity in response.split(',')]
            return [entity for entity in entities if entity]  # Filter empty strings
        
        return None


class ModelOrchestrator:
    """Orchestrates multiple models for comprehensive analysis"""
    
    def __init__(self, databricks_api_key: str = None):
        """
        Initialize model orchestrator.
        
        Args:
            databricks_api_key (str, optional): Databricks API key
        """
        self.bge_model = BGELargeEnV15(databricks_api_key)
        self.llama_model = MetaLlama370BInstruct(databricks_api_key)
        self.claude_model = ClaudeSonet4(databricks_api_key)
    
    def comprehensive_analysis(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of documents using all models.
        
        Args:
            documents (List[Dict[str, Any]]): List of documents to analyze
            
        Returns:
            Dict[str, Any]: Comprehensive analysis results
        """
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_documents": len(documents),
            "embeddings": {},
            "relationships": [],
            "summaries": {},
            "entities": {},
            "field_mappings": {}
        }
        
        logger.info(f"Starting comprehensive analysis of {len(documents)} documents")
        
        # Generate embeddings
        texts = []
        doc_ids = []
        for i, doc in enumerate(documents):
            text = doc.get('text', doc.get('content', ''))
            if text:
                texts.append(text)
                doc_ids.append(str(i))
        
        if texts:
            embeddings = self.bge_model.get_batch_embeddings(texts)
            for doc_id, embedding in zip(doc_ids, embeddings):
                if embedding:
                    analysis["embeddings"][doc_id] = embedding
        
        # Analyze relationships between documents
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                text1 = documents[i].get('text', documents[i].get('content', ''))
                text2 = documents[j].get('text', documents[j].get('content', ''))
                
                if text1 and text2:
                    relationship = self.llama_model.analyze_relationships(text1, text2)
                    if relationship:
                        relationship['doc1_id'] = str(i)
                        relationship['doc2_id'] = str(j)
                        analysis["relationships"].append(relationship)
        
        # Generate summaries and extract entities
        for i, doc in enumerate(documents):
            text = doc.get('text', doc.get('content', ''))
            if text and len(text) > 100:  # Only process substantial texts
                doc_id = str(i)
                
                # Generate summary
                summary = self.claude_model.summarize_document(text)
                if summary:
                    analysis["summaries"][doc_id] = summary
                
                # Extract entities
                entities = self.claude_model.extract_entities(text)
                if entities:
                    analysis["entities"][doc_id] = entities
        
        logger.info("Comprehensive analysis completed")
        return analysis
