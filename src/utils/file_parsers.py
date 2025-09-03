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

class FileParser:
    """Base class for file parsers"""
    
    @staticmethod
    def parse_csv(file_path: str) -> List[Dict[str, Any]]:
        """
        Parse CSV file row by row and return list of dictionaries.
        
        Args:
            file_path (str): Path to CSV file
            
        Returns:
            List[Dict[str, Any]]: List of rows as dictionaries
        """
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
                'chunks': FileParser._chunk_text(text, chunk_size=1000, overlap=200)
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
    def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Size of each chunk
            overlap (int): Overlap between chunks
            
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
