#document_processor.py
import os
from typing import List, Optional
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from pathlib import Path

from .config import config
from .logging_utils import logger
from .file_utils import read_file_with_encoding

def read_file_content(file_path: str) -> Optional[str]:
    """Read file content with proper encoding"""
    content, encoding = read_file_with_encoding(file_path)
    if content is None:
        logger.error(f"Failed to read file {file_path} with any encoding")
    return content

def process_single_file(file_path: str, text_splitter) -> List[Document]:
    """
    Process a single file and return document chunks with enhanced metadata

    Args:
        file_path (str): Path to the text file
        text_splitter: Text splitting utility

    Returns:
        List[Document]: List of document chunks with enhanced metadata
    """
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Convert file_path to Path object for easier metadata extraction
        file_path_obj = Path(file_path)

        # Split the text into chunks
        text_chunks = text_splitter.split_text(text)

        # Create documents with enhanced metadata
        documents = [
            Document(
                page_content=chunk, 
                metadata={
                    'source': str(file_path),  # Full file path
                    'filename': file_path_obj.name,  # Filename
                    'directory': str(file_path_obj.parent),  # Directory path
                    'file_extension': file_path_obj.suffix  # File extension
                }
            ) for chunk in text_chunks
        ]

        return documents

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return []

def process_documents(documents: List[Document], embeddings) -> FAISS:
    """Create a FAISS vector store from documents"""
    if not documents:
        raise ValueError("No documents to process")

    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store