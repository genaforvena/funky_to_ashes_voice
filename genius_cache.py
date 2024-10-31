import json
import os
import time
from typing import Optional, Dict, Any

class GeniusCache:
    def __init__(self, cache_dir: str = 'cache/genius', expiration_days: int = 30):
        self.cache_dir = cache_dir
        self.expiration_seconds = expiration_days * 24 * 60 * 60
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, phrase: str, operation: str) -> str:
        """Generate a cache file path for a given phrase and operation"""
        # Create a safe filename from the phrase
        safe_phrase = "".join(x for x in phrase if x.isalnum() or x in [' ', '-', '_'])
        safe_phrase = safe_phrase.replace(' ', '_')[:100]  # Limit length
        return os.path.join(self.cache_dir, f"{safe_phrase}_{operation}.json")
    
    def get_cached_data(self, phrase: str, operation: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached data if it exists and hasn't expired"""
        cache_path = self._get_cache_path(phrase, operation)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if cache has expired
                if time.time() - data['timestamp'] < self.expiration_seconds:
                    return data['result']
                    
            except Exception as e:
                print(f"Error reading cache: {e}")
        
        return None
    
    def save_to_cache(self, phrase: str, operation: str, result: Dict[str, Any]) -> None:
        """Save data to cache with current timestamp"""
        cache_path = self._get_cache_path(phrase, operation)
        
        try:
            cache_data = {
                'timestamp': time.time(),
                'result': result
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error saving to cache: {e}") 