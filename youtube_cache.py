import json
import os
from typing import Optional, Dict, Any

class YouTubeCache:
    def __init__(self, cache_dir: str = 'cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, video_id: str, cache_type: str) -> str:
        return os.path.join(self.cache_dir, f"{video_id}_{cache_type}.json")
    
    def get_cached_data(self, video_id: str, cache_type: str) -> Optional[Dict[str, Any]]:
        cache_path = self._get_cache_path(video_id, cache_type)
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def save_to_cache(self, video_id: str, cache_type: str, data: Dict[str, Any]) -> None:
        cache_path = self._get_cache_path(video_id, cache_type)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2) 