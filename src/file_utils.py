# file_utils.py
from typing import List, Optional, Tuple
import chardet
from .robust_file_reader import RobustFileReader
from .config import config
from .logging_utils import logger

def get_text_files(folder_path: str) -> List[str]:
    """Recursively find all text-based files"""
    return RobustFileReader.get_text_files(
        folder_path, 
        config.TEXT_EXTENSIONS,
        logger
    )

def read_file_with_encoding(file_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Read file content with robust encoding detection and fallbacks"""
    return RobustFileReader.read_file(file_path, logger)