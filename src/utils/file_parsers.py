import os
import csv
import ast
from PyPDF2 import PdfReader
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataMappingParser:
    """Enhanced parser for data mapping CSV files"""
    
    @staticmethod
    def parse_mapping_csv(file_path: str) -> List[Dict[str, Any]]:
        """
        Parse data mapping CSV with enhanced relationship detection.
        
        Args:
            file_path (str): Path to mapping CSV file
            
        Returns:
            List[Dict[str, Any]]: List of mapping rows with enhanced metadata
        """
        rows = []
        try:
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                # Detect delimiter
                sample = csvfile.read(1024)
                csvfile.seek(0)
                delimiter = ',' if ',' in sample else ';' if ';' in sample else '\t'
                
                reader = csv.DictReader(csvfile, delimiter=delimiter)
                
                # Normalize column names for common mapping patterns
                normalized_columns = {}
                for field in reader.fieldnames or []:
                    field_clean = field.strip()
                    field_lower = field_clean.lower()
                    
                    # Map various column name patterns to standard names
                    if any(term in field_lower for term in ['source table', 'source_table', 'src_table']):
                        normalized_columns[field_clean] = 'source_table'
                    elif any(term in field_lower for term in ['source column', 'source_column', 'source field', 'source_field', 'src_column']):
                        normalized_columns[field_clean] = 'source_column'
                    elif any(term in field_lower for term in ['target table', 'target_table', 'tgt_table', 'destination table']):
                        normalized_columns[field_clean] = 'target_table'
                    elif any(term in field_lower for term in ['target column', 'target_column', 'target field', 'target_field', 'field', 'tgt_column']):
                        normalized_columns[field_clean] = 'target_field'
                    elif any(term in field_lower for term in ['transformation', 'mapping rule', 'mapping_rule', 'transform', 'rule', 'mappping']):
                        normalized_columns[field_clean] = 'transformation_rule'
                    elif any(term in field_lower for term in ['type', 'data type', 'data_type', 'field_type']):
                        normalized_columns[field_clean] = 'data_type'
                    elif any(term in field_lower for term in ['description', 'desc', 'comment']):
                        normalized_columns[field_clean] = 'description'
                    else:
                        normalized_columns[field_clean] = field_clean.lower().replace(' ', '_')
                
                for i, row in enumerate(reader):
                    # Skip empty rows
                    if not any(value.strip() for value in row.values() if value):
                        continue
                        
                    # Clean and normalize row data
                    cleaned_row = {}
                    normalized_row = {}
                    
                    for key, value in row.items():
                        if key:
                            cleaned_key = key.strip()
                            cleaned_value = value.strip() if value else ''
                            cleaned_row[cleaned_key] = cleaned_value
                            
                            # Add normalized mapping
                            normalized_key = normalized_columns.get(cleaned_key, cleaned_key.lower().replace(' ', '_'))
                            normalized_row[normalized_key] = cleaned_value
                    
                    # Add metadata
                    mapping_metadata = {
                        '_row_number': i + 1,
                        '_source_file': os.path.basename(file_path),
                        '_mapping_type': 'data_mapping',
                        '_has_source_table': bool(normalized_row.get('source_table')),
                        '_has_target_field': bool(normalized_row.get('target_field')),
                        '_has_transformation': bool(normalized_row.get('transformation_rule'))
                    }
                    
                    # Create mapping relationship metadata
                    if normalized_row.get('source_table') and normalized_row.get('target_field'):
                        mapping_metadata['_mapping_relationship'] = {
                            'source': f"{normalized_row.get('source_table', '')}.{normalized_row.get('source_column', '')}",
                            'target': normalized_row.get('target_field', ''),
                            'transformation': normalized_row.get('transformation_rule', ''),
                            'data_type': normalized_row.get('data_type', ''),
                            'description': normalized_row.get('description', '')
                        }
                    
                    # Combine original and normalized data
                    final_row = {**cleaned_row, **normalized_row, **mapping_metadata}
                    rows.append(final_row)
                    
            logger.info(f"Successfully parsed {len(rows)} mapping rows from {file_path}")
            return rows
            
        except Exception as e:
            logger.error(f"Error parsing mapping CSV file {file_path}: {str(e)}")
            return []

class FileParser:
    """Base class for file parsers"""
    
    @staticmethod
    def parse_csv(file_path: str) -> List[Dict[str, Any]]:
        """
        Parse CSV file with automatic detection of mapping structure.
        
        Args:
            file_path (str): Path to CSV file
            
        Returns:
            List[Dict[str, Any]]: List of rows as dictionaries
        """
        # First, check if this is a data mapping CSV
        try:
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                # Read first few lines to check for mapping indicators
                lines = []
                for _ in range(5):  # Check first 5 lines
                    try:
                        line = csvfile.readline()
                        if line:
                            lines.append(line.lower())
                        else:
                            break
                    except:
                        break
                
                # Check for mapping-related column names in any of the first lines
                all_content = ' '.join(lines)
                mapping_indicators = [
                    'source table', 'target table', 'source column', 'target field',
                    'mapping', 'transformation', 'source_table', 'target_field',
                    'field', 'mappping rules'  # Handle typo in your data
                ]
                
                if any(indicator in all_content for indicator in mapping_indicators):
                    logger.info(f"Detected data mapping CSV: {file_path}")
                    return DataMappingParser.parse_mapping_csv(file_path)
        except Exception:
            pass  # Fall back to regular CSV parsing
        
        # Regular CSV parsing
        rows = []
        try:
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                # Try to detect delimiter
                sample = csvfile.read(1024)
                csvfile.seek(0)
                delimiter = ',' if ',' in sample else ';' if ';' in sample else '\t'
                
                reader = csv.DictReader(csvfile, delimiter=delimiter)
                for i, row in enumerate(reader):
                    # Clean up row data
                    cleaned_row = {}
                    for key, value in row.items():
                        if key:  # Skip empty keys
                            cleaned_row[key.strip()] = value.strip() if value else ''
                    
                    # Add metadata
                    cleaned_row['_row_number'] = i + 1
                    cleaned_row['_source_file'] = os.path.basename(file_path)
                    rows.append(cleaned_row)
                    
            logger.info(f"Successfully parsed {len(rows)} rows from {file_path}")
            return rows
            
        except Exception as e:
            logger.error(f"Error parsing CSV file {file_path}: {str(e)}")
            return []

    @staticmethod
    def parse_pdf(file_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF using PyPDF2.
        
        Args:
            file_path (str): Path to PDF file
            
        Returns:
            Dict[str, Any]: Extracted text and metadata
        """
        try:
            text = ""
            metadata = {}
            
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                
                # Extract metadata
                if reader.metadata:
                    metadata = {
                        'title': reader.metadata.get('/Title', ''),
                        'author': reader.metadata.get('/Author', ''),
                        'subject': reader.metadata.get('/Subject', ''),
                        'creator': reader.metadata.get('/Creator', ''),
                        'producer': reader.metadata.get('/Producer', ''),
                        'creation_date': reader.metadata.get('/CreationDate', ''),
                        'modification_date': reader.metadata.get('/ModDate', '')
                    }
                
                metadata['num_pages'] = len(reader.pages)
                metadata['source_file'] = os.path.basename(file_path)
                
                # Extract text from all pages
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
            logger.info(f"Successfully parsed PDF {file_path}: {len(text)} characters, {metadata['num_pages']} pages")
            
            return {
                'text': text,
                'metadata': metadata,
                'chunks': FileParser._chunk_text(text, chunk_size=2000, overlap=400)
            }
            
        except Exception as e:
            logger.error(f"Error parsing PDF file {file_path}: {str(e)}")
            return {'text': '', 'metadata': {}, 'chunks': []}

    @staticmethod
    def parse_python_file(file_path: str) -> Dict[str, Any]:
        """
        Parse Python file and extract functions, classes, and imports.
        
        Args:
            file_path (str): Path to Python file
            
        Returns:
            Dict[str, Any]: Parsed Python code information
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            functions = []
            classes = []
            imports = []
            variables = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'line_number': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node) or ''
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        'name': node.name,
                        'line_number': node.lineno,
                        'docstring': ast.get_docstring(node) or ''
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            'module': alias.name,
                            'alias': alias.asname,
                            'type': 'import'
                        })
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imports.append({
                            'module': node.module,
                            'name': alias.name,
                            'alias': alias.asname,
                            'type': 'from_import'
                        })
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            variables.append({
                                'name': target.id,
                                'line_number': node.lineno
                            })
            
            logger.info(f"Successfully parsed Python file {file_path}: {len(functions)} functions, {len(classes)} classes")
            
            return {
                'content': content,
                'functions': functions,
                'classes': classes,
                'imports': imports,
                'variables': variables,
                'metadata': {
                    'source_file': os.path.basename(file_path),
                    'file_path': file_path
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing Python file {file_path}: {str(e)}")
            return {
                'content': '',
                'functions': [],
                'classes': [],
                'imports': [],
                'variables': [],
                'metadata': {}
            }

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 2000, overlap: int = 400) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Size of each chunk (default: 2000 characters)
            overlap (int): Overlap between chunks (default: 400 characters)
            
        Returns:
            List[str]: List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at word boundary
            if end < len(text):
                last_space = chunk.rfind(' ')
                if last_space > chunk_size * 0.8:  # Only if we don't lose too much
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks

    @staticmethod
    def find_files(root_dir: str) -> Dict[str, List[str]]:
        """
        Recursively find all relevant files in the directory structure.
        
        Args:
            root_dir (str): Root directory to search
            
        Returns:
            Dict[str, List[str]]: Dictionary categorizing found files by type
        """
        file_paths = {
            'csv': [],
            'pdf': [],
            'python': [],
            'other': []
        }
        
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                
                if filename.endswith('.csv'):
                    file_paths['csv'].append(file_path)
                elif filename.endswith('.pdf'):
                    file_paths['pdf'].append(file_path)
                elif filename.endswith('.py'):
                    file_paths['python'].append(file_path)
                else:
                    file_paths['other'].append(file_path)
        
        logger.info(f"Found files: {sum(len(files) for files in file_paths.values())} total")
        for file_type, files in file_paths.items():
            logger.info(f"  {file_type}: {len(files)} files")
            
        return file_paths
