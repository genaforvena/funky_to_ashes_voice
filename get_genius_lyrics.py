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
        
        words = text.split()
        phrases = []
        start = 0
        
        while start < len(words):
            current_phrase = ""
            last_successful_end = start
            
            for end in range(start, len(words)):
                current_phrase = " ".join(words[start:end + 1])
                logging.info(f"Checking if phrase exists: {current_phrase}")
                
                if self.genius_api.check_phrase_exists(current_phrase):
                    last_successful_end = end + 1
                else:
                    break
            
            if last_successful_end > start:
                successful_phrase = " ".join(words[start:last_successful_end])
                logging.info(f"Found match for phrase: {successful_phrase}")
                phrases.append(successful_phrase)
            
            start = last_successful_end if last_successful_end > start else start + 1
        
        score = sum(len(phrase) for phrase in phrases)
        return score, phrases
    
    def retry_search(self, phrase: str, max_retries: int = 5) -> Optional[str]:
        """
        Retry searching for a phrase in Genius with a limit on retries.
        """
        logging.info(f"Retrying search for phrase: '{phrase}'")
        for attempt in range(max_retries):
            logging.info(f"Attempt {attempt + 1} for phrase: '{phrase}'")
            if self.genius_api.check_phrase_exists(phrase):
                logging.info(f"Found match for phrase on retry: {phrase}")
                return phrase
            # Optionally, modify the phrase slightly for the next attempt
            # For example, remove punctuation or try synonyms
            # phrase = modify_phrase(phrase)
        
        logging.info(f"No match found for phrase after {max_retries} retries: {phrase}")
        return None
    
    def get_title_and_artist(self, phrase: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Retrieve the title and artist for a given phrase from Genius
        """
        logging.info(f"Retrieving title and artist for phrase: {phrase}")
        search_url = f"{self.genius_api.base_url}/search"
        params = {'q': phrase}
        
        try:
            response = requests.get(search_url, headers=self.genius_api.headers, params=params)
            response.raise_for_status()
            hits = response.json().get('response', {}).get('hits', [])
            
            for hit in hits:
                # Check if the hit is a complete song
                if hit['result'].get('lyrics_state') == 'complete':
                    title = hit['result']['title']
                    artist = hit['result']['primary_artist']['name']
                    logging.info(f"Found title: {title}, artist: {artist} for phrase: {phrase}")
                    return title, artist
            
            logging.info(f"No title and artist found for phrase: {phrase}")
            return None, None
        
        except requests.exceptions.RequestException as e:
            logging.error(f"API Request Error: {e}")
            return None, None
        except Exception as e:
            logging.error(f"Unexpected Error: {e}")
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