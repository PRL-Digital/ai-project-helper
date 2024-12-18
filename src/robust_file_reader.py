#/robust_file_reader.py
import os
from pathlib import Path
from typing import Optional, List, Tuple
import chardet
from logging import Logger

class RobustFileReader:
    """A utility class for robustly reading text files with various encodings."""
    
    FALLBACK_ENCODINGS = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """
        Detect the encoding of a file using chardet.
        Returns the detected encoding or 'utf-8' as fallback.
        """
        try:
            # Read a sample of the file (first 10KB) to detect encoding
            with open(file_path, 'rb') as file:
                raw_data = file.read(10000)
            result = chardet.detect(raw_data)
            return result['encoding'] if result['encoding'] else 'utf-8'
        except Exception:
            return 'utf-8'

    @staticmethod
    def read_file(file_path: str, logger: Optional[Logger] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Attempt to read a file using multiple encodings.
        Returns a tuple of (content, encoding_used) if successful, (None, None) if failed.
        """
        try:
            # First try with detected encoding
            detected_encoding = RobustFileReader.detect_encoding(file_path)
            encodings_to_try = [detected_encoding] + [
                enc for enc in RobustFileReader.FALLBACK_ENCODINGS 
                if enc != detected_encoding
            ]
            
            file_content = None
            encoding_used = None
            
            for encoding in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        file_content = file.read()
                        encoding_used = encoding
                        break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    if logger:
                        logger.error(f"Error reading file with {encoding} encoding: {str(e)}")
                    continue
            
            if file_content is None:
                if logger:
                    logger.error(f"Failed to read file {file_path} with any encoding")
                return None, None
                
            return file_content, encoding_used
            
        except Exception as e:
            if logger:
                logger.error(f"Unexpected error reading file {file_path}: {str(e)}")
            return None, None
    
    @staticmethod
    def get_text_files(folder_path: str, extensions: List[str], logger: Optional[Logger] = None) -> List[str]:
        """
        Recursively find all text-based files in a folder that can be successfully read.
        Returns a list of valid file paths.
        """
        valid_files = []
        try:
            for root, _, files in os.walk(folder_path):
                if logger:
                    logger.info(f"Scanning directory: {root}")
                    
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        full_path = os.path.abspath(os.path.join(root, file))
                        
                        # Test if we can actually read the file
                        content, encoding = RobustFileReader.read_file(full_path, logger)
                        if content is not None:
                            valid_files.append(full_path)
                            if logger:
                                logger.info(f"Found readable file: {file} (encoding: {encoding})")
                        else:
                            if logger:
                                logger.warning(f"Skipping unreadable file: {file}")
                                
        except Exception as e:
            if logger:
                logger.error(f"Error walking directory {folder_path}: {str(e)}")
                logger.exception(e)
                
        return valid_files