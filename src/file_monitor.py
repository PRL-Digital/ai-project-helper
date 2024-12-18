import os
import hashlib
from typing import List, Dict, Set, Tuple

class FileMonitor:
    def __init__(self, base_folder: str = None):
        self.file_hashes: Dict[str, str] = {}
        self.base_folder = base_folder

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def initialize_hashes(self, files: List[str], base_folder: str = None) -> None:
        """Initialize hash tracking for files"""
        self.base_folder = base_folder or self.base_folder
        if not self.base_folder:
            raise ValueError("base_folder must be provided either during initialization or when calling initialize_hashes")
            
        for file in files:
            relative_path = os.path.relpath(file, self.base_folder)
            self.file_hashes[relative_path] = self._calculate_file_hash(file)

    def check_file_changes(self, files: List[str], base_folder: str = None) -> Tuple[Set[str], Set[str]]:
        """Check for added, updated, and removed files"""
        base_folder = base_folder or self.base_folder
        if not base_folder:
            raise ValueError("base_folder must be provided either during initialization or when checking files")
            
        current_files = {os.path.relpath(f, base_folder) for f in files}
        tracked_files = set(self.file_hashes.keys())

        added_files = set()
        updated_files = set()
        removed_files = tracked_files - current_files

        for file in current_files:
            full_path = os.path.join(base_folder, file)
            current_hash = self._calculate_file_hash(full_path)

            if file not in self.file_hashes:
                added_files.add(file)
                self.file_hashes[file] = current_hash
            elif self.file_hashes[file] != current_hash:
                updated_files.add(file)
                self.file_hashes[file] = current_hash

        # Remove tracking for deleted files
        for file in removed_files:
            del self.file_hashes[file]

        return added_files.union(updated_files), removed_files