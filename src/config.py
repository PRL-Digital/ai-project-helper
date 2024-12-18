from dataclasses import dataclass, field
from typing import Set

@dataclass
class Config:
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 0
    SEARCH_K: int = 5
    FETCH_K: int = 20
    MODEL_NAME: str = 'claude-3-5-haiku-latest'
    MAX_TOKENS: int = 4000
    TEXT_EXTENSIONS: Set[str] = field(default_factory=lambda: {
        '.txt', '.md', '.py', '.js', '.ts', '.html', '.css', 
        '.json', '.yaml', '.yml', '.xml', '.csv', '.log',
        '.sh', '.bash', '.sql', '.ini', '.conf'
    })

config = Config()