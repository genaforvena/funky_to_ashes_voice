import requests
from functools import lru_cache
from typing import Tuple, List, Optional
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LyricsSplitter:
    def __init__(self, token: str):
        self.genius_api = GeniusAPI(token)
        logging.info("Initialized LyricsSplitter with provided Genius token")
    
    def split_lyrics(self, text: str) -> Tuple[int, List[str]]:
        logging.info(f"Splitting lyrics for text: '{text}'")
        
        # Check if the entire text is a valid phrase
        if self.genius_api.check_phrase_exists(text):
            logging.info(f"Found match for entire phrase: {text}")
            return len(text), [text]
        
        # If not, attempt to split into smaller phrases
        best_split = self.find_best_split(text)
        score = len(best_split) if best_split else 0
        return score, [best_split] if best_split else []
    
    def find_best_split(self, text: str) -> str:
        logging.info(f"Finding best split for text: '{text}'")
        words = text.split()
        longest_match = ""
        
        for start in range(len(words)):
            current_phrase = ""
            for end in range(start, len(words)):
                current_phrase = " ".join(words[start:end + 1])
                if not self.genius_api.check_phrase_exists(current_phrase):
                    break  # Stop if the current phrase is not found
                longest_match = current_phrase  # Update longest match if found
        
        logging.info(f"Longest uninterrupted match found: '{longest_match}'")
        return longest_match

    def get_title_and_artist(self, phrase: str) -> Tuple[str, str]:
        """
        Retrieve the title and artist for a given phrase from Genius
        """
        logging.info(f"Retrieving title and artist for phrase: {phrase}")
        # Example implementation
        search_url = f"{self.genius_api.base_url}/search"
        params = {'q': phrase}
        
        try:
            response = requests.get(search_url, headers=self.genius_api.headers, params=params)
            response.raise_for_status()
            hits = response.json().get('response', {}).get('hits', [])
            
            for hit in hits:
                if hit['result'].get('lyrics_state') == 'complete':
                    title = hit['result']['title']
                    artist = hit['result']['primary_artist']['name']
                    logging.info(f"Found title: {title}, artist: {artist} for phrase: {phrase}")
                    return title, artist
            
            logging.info(f"No title and artist found for phrase: {phrase}")
            return None, None
        
        except Exception as e:
            logging.error(f"API Error: {e}")
            return None, None

class GeniusAPI:
    def __init__(self, token: str):
        self.base_url = "https://api.genius.com"
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def check_phrase_exists(self, phrase: str) -> bool:
        """Check if a phrase exists in Genius"""
        logging.info(f"Checking if phrase exists: {phrase}")
        if not phrase.strip():
            logging.warning("Empty phrase provided")
            return False
        
        search_url = f"{self.base_url}/search"
        params = {'q': phrase}
        
        try:
            logging.debug(f"Request URL: {search_url}")
            logging.debug(f"Request Headers: {self.headers}")
            logging.debug(f"Request Params: {params}")
            
            response = requests.get(search_url, headers=self.headers, params=params)
            response.raise_for_status()
            
            logging.debug(f"Response Status Code: {response.status_code}")
            logging.debug(f"Response JSON: {response.json()}")
            
            hits = response.json().get('response', {}).get('hits', [])
            
            # Check for exact match in search results
            for hit in hits:
                if hit['result'].get('lyrics_state') == 'complete':
                    logging.info(f"Found exact match in Genius")
                    return True
            
            logging.info(f"No exact match found in Genius")
            return False
        
        except Exception as e:
            logging.error(f"API Error: {e}")
            return False

# Export the class
__all__ = ['GeniusAPI']

def example_usage():
    genius_token = os.getenv('GENIUS_TOKEN')
    splitter = LyricsSplitter(genius_token)
    
    examples = [
        "the future is now",
        "the future is now today",
        "I can see the future is now"
    ]
    
    for text in examples:
        logging.info(f"Processing example text: '{text}'")
        score, phrases = splitter.split_lyrics(text)
        logging.info(f"Best split: Score {score}: {' | '.join(phrases)}")
        
        for phrase in phrases:
            title_artist = splitter.get_title_and_artist(phrase)
            if title_artist:
                title, artist = title_artist
                logging.info(f"Phrase '{phrase}' found in '{title}' by {artist}")
            else:
                logging.warning(f"Phrase '{phrase}' not found in Genius database")

if __name__ == "__main__":
    example_usage()