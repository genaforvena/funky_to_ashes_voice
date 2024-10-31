import requests
from functools import lru_cache
from typing import Tuple, List, Optional
import os
import logging
from genius_cache import GeniusCache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LyricsSplitter:
    def __init__(self, token: str):
        self.genius_api = GeniusAPI(token)
        self.genius_cache = GeniusCache()
        self.max_words = 6  # Initial maximum words per phrase
        logging.info("Initialized LyricsSplitter with provided Genius token")
    
    def split_lyrics(self, text: str) -> Tuple[int, List[str]]:
        """Split lyrics into phrases, using current max_words setting"""
        logging.info(f"Splitting lyrics for text: '{text}' with max_words={self.max_words}")
        
        words = text.split()
        phrases = []
        start = 0
        
        while start < len(words):
            current_phrase = ""
            last_successful_end = start
            
            for end in range(start, len(words)):
                current_phrase = " ".join(words[start:end + 1])
                current_words = current_phrase.split()
                
                # Start new phrase if current one exceeds max_words
                if len(current_words) > self.max_words:
                    if last_successful_end > start:
                        successful_phrase = " ".join(words[start:last_successful_end])
                        logging.info(f"Phrase exceeded {self.max_words} words. Adding: {successful_phrase}")
                        phrases.append(successful_phrase)
                    break
                
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
    
    def reduce_chunk_size(self):
        """Reduce the maximum number of words per phrase"""
        self.max_words = max(1, self.max_words - 1)
        logging.info(f"Reduced max words per phrase to {self.max_words}")
    
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
    
    def get_title_and_artist(self, phrase: str) -> Optional[Tuple[str, str]]:
        """Get the song title and artist for a given phrase using Genius API with caching"""
        # Check cache first
        cached_result = self.genius_cache.get_cached_data(phrase, 'title_artist')
        if cached_result:
            return cached_result.get('title'), cached_result.get('artist')
        
        try:
            # Search Genius for the phrase
            result = self.genius_api.get_title_and_artist(phrase)
            
            if result and result[0]:
                title = result[0]
                artist = result[1]
                
                # Save to cache
                self.genius_cache.save_to_cache(
                    phrase, 
                    'title_artist',
                    {'title': title, 'artist': artist}
                )
                
                logging.info(f"Found title: {title}, artist: {artist} for phrase: {phrase}")
                return title, artist
                
        except Exception as e:
            logging.error(f"Error searching Genius for phrase '{phrase}': {str(e)}")
        
        return None

class GeniusAPI:
    def __init__(self, token: str):
        self.base_url = "https://api.genius.com"
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def check_phrase_exists(self, phrase: str) -> bool:
        """Check if a phrase exists in Genius hip-hop tracks"""
        logging.info(f"Checking if phrase exists: {phrase}")
        if not phrase.strip():
            logging.warning("Empty phrase provided")
            return False
        
        search_url = f"{self.base_url}/search"
        params = {
            'q': phrase,
            'per_page': 20  # Increase results to have better chance of finding hip-hop tracks
        }
        
        try:
            response = requests.get(search_url, headers=self.headers, params=params)
            response.raise_for_status()
            
            hits = response.json().get('response', {}).get('hits', [])
            
            # Check for hip-hop tracks in search results
            for hit in hits:
                result = hit['result']
                if result.get('lyrics_state') == 'complete':
                    # Check if primary genre is hip-hop
                    return True
            
            logging.info(f"No hip-hop match found in Genius")
            return False
        
        except Exception as e:
            logging.error(f"API Error: {e}")
            return False
    
    def get_title_and_artist(self, phrase: str) -> Tuple[Optional[str], Optional[str]]:
        """Retrieve the title and artist for a given phrase from hip-hop tracks in Genius"""
        logging.info(f"Retrieving title and artist for phrase: {phrase}")
        search_url = f"{self.base_url}/search"
        params = {
            'q': phrase,
            'per_page': 20
        }
        
        try:
            response = requests.get(search_url, headers=self.headers, params=params)
            response.raise_for_status()
            hits = response.json().get('response', {}).get('hits', [])
            
            for hit in hits:
                result = hit['result']
                # Check if the hit is a complete song and is hip-hop
                if result.get('lyrics_state') == 'complete':
                    title = result['title']
                    artist = result['primary_artist']['name']
                    logging.info(f"Found hip-hop track: {title} by {artist}")
                    return title, artist
            
            return None, None
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API Request Error: {e}")
            return None, None
        except Exception as e:
            logging.error(f"Unexpected Error: {e}")
            return None, None

# Export the class
__all__ = ['GeniusAPI']

def example_usage():
    genius_token = os.getenv('GENIUS_TOKEN')
    splitter = LyricsSplitter(genius_token)
    
    examples = [
        "the future is now",
        "the future is now today",
        "Cause I'm going for the steel For half, half of his niggas'll take him out the picture Exercise index, won't need BowFlex Or cultivated a better class of friends Cell block and locked, I never clock it, y'all"
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