import requests
from functools import lru_cache
from typing import Tuple, List, Optional
import os

class LyricsSplitter:
    def __init__(self, genius_token: str):
        self.genius_token = genius_token
        self.headers = {'Authorization': f'Bearer {genius_token}'}
        self.base_url = "https://api.genius.com"
        self.cache = {}
        
    @lru_cache(maxsize=1000)
    def check_phrase_exists(self, phrase: str) -> bool:
        """
        Check if a phrase exists in Genius database
        Uses LRU cache to avoid repeated API calls
        """
        if not phrase.strip():
            return False
            
        search_url = f"{self.base_url}/search"
        params = {'q': phrase}
        
        try:
            response = requests.get(
                search_url,
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            
            hits = response.json()['response']['hits']
            # Check if the exact phrase appears in any of the results
            for hit in hits:
                if phrase.lower() in hit['result']['full_title'].lower():
                    return True
            return False
            
        except Exception as e:
            print(f"API Error: {e}")
            return False

    def get_title_and_artist(self, phrase: str) -> Optional[Tuple[str, str]]:
        """
        Get the title and artist name for a given phrase from Genius
        
        Args:
            phrase: The phrase to search for
        
        Returns:
            A tuple of (title, artist) if found, otherwise None
        """
        search_url = f"{self.base_url}/search"
        params = {'q': phrase}
        
        try:
            response = requests.get(
                search_url,
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            
            hits = response.json()['response']['hits']
            for hit in hits:
                if phrase.lower() in hit['result']['full_title'].lower():
                    title = hit['result']['title']
                    artist = hit['result']['primary_artist']['name']
                    return title, artist
            return None
            
        except Exception as e:
            print(f"API Error: {e}")
            return None

    def score_split(self, phrases: List[str]) -> float:
        """
        Score a potential split based on:
        - Length of phrases (longer is better)
        - Number of splits (fewer is better)
        - Whether phrases exist in database
        """
        if not all(self.check_phrase_exists(phrase) for phrase in phrases):
            return 0
            
        # Base score from lengths
        length_score = sum(len(phrase) ** 2 for phrase in phrases)
        
        # Penalize splits
        split_penalty = (len(phrases) - 1) * 5
        
        return length_score - split_penalty

    def find_best_split(self, text: str, depth: int = 0, max_depth: int = 3) -> Tuple[float, List[str]]:
        """
        Recursively find the best way to split the text
        Returns (score, list of phrases)
        """
        if depth >= max_depth:
            return 0, [text]
            
        # Try the whole phrase first
        if self.check_phrase_exists(text):
            whole_score = self.score_split([text])
            if whole_score > 0:
                return whole_score, [text]
        
        # Cache key for this text
        cache_key = text
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        best_score = 0
        best_split = [text]
        
        # Try all possible split points
        words = text.split()
        for i in range(1, len(words)):
            left = " ".join(words[:i])
            right = " ".join(words[i:])
            
            # Recursively find best splits for left and right parts
            left_score, left_splits = self.find_best_split(left, depth + 1, max_depth)
            right_score, right_splits = self.find_best_split(right, depth + 1, max_depth)
            
            # Combine scores and splits
            combined_score = left_score + right_score
            if combined_score > best_score:
                best_score = combined_score
                best_split = left_splits + right_splits
        
        # Cache the result
        self.cache[cache_key] = (best_score, best_split)
        return best_score, best_split

    def split_lyrics(self, text: str) -> Tuple[float, List[str]]:
        """
        Main entry point - splits lyrics into optimal phrases
        Returns (score, list of phrases)
        """
        self.cache = {}  # Reset cache for new text
        return self.find_best_split(text)

def example_usage():
    # Initialize with your Genius API token
    genius_token = os.getenv('GENIUS_TOKEN')
    splitter = LyricsSplitter(genius_token)
    
    # Example texts
    examples = [
        "the future is now",
        "the future is now today",
        "I can see the future is now"
    ]
    
    # Process each example
    for text in examples:
        print(f"\nSplitting: {text}")
        score, phrases = splitter.split_lyrics(text)
        print(f"Best split: Score {score}: {' | '.join(phrases)}")
        
        # Get title and artist for each phrase
        for phrase in phrases:
            title_artist = splitter.get_title_and_artist(phrase)
            if title_artist:
                title, artist = title_artist
                print(f"Phrase '{phrase}' found in '{title}' by {artist}")
            else:
                print(f"Phrase '{phrase}' not found in Genius database")

if __name__ == "__main__":
    example_usage()