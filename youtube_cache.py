import json
import os
import hashlib
from typing import Optional, Dict, Any

class YouTubeCache:
    def __init__(self, cache_dir: str = 'cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_storage = {}
    
    def _hash_key(self, key: str) -> str:
        """Create a hash of the key to use as filename"""
        return hashlib.md5(key.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, key: str, cache_type: str) -> str:
        hashed_key = self._hash_key(key)
        return os.path.join(self.cache_dir, f"{hashed_key}_{cache_type}.json")
    
    def get_cached_data(self, key: str, cache_type: str) -> Optional[Dict[str, Any]]:
        cache_path = self._get_cache_path(key, cache_type)
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def save_to_cache(self, key: str, cache_type: str, data: Dict[str, Any]) -> None:
        cache_path = self._get_cache_path(key, cache_type)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_all_cached_data(self, cache_type: str) -> Dict[str, Any]:
        return {key: data for (key, type_), data in self.cache_storage.items() if type_ == cache_type}