import requests
from functools import lru_cache
from typing import Tuple, List, Optional
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LyricsSplitter:
    def __init__(self, genius_token: str):
        self.genius_token = genius_token
        self.headers = {'Authorization': f'Bearer {genius_token}'}
        self.base_url = "https://api.genius.com"
        self.cache = {}
        logging.info("Initialized LyricsSplitter with provided Genius token")

    @lru_cache(maxsize=1000)
    def check_phrase_exists(self, phrase: str) -> bool:
        """Check if exact phrase exists in Genius"""
        logging.info(f"Checking if phrase exists: {phrase}")
        if not phrase.strip():
            logging.warning("Empty phrase provided")
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
            
            # Look for exact match in search results
            for hit in hits:
                if hit['result'].get('lyrics_state') == 'complete':
                    # Note: We're not including the actual lyrics comparison here
                    # Just checking if the phrase exists in Genius database
                    logging.info(f"Found exact match in Genius")
                    return True
                    
            logging.info(f"No exact match found in Genius")
            return False
            
        except Exception as e:
            logging.error(f"API Error: {e}")
            return False

    def retrieve_title_and_artist(self, phrase: str) -> Tuple[str, str]:
        logging.info(f"Retrieving title and artist for phrase: {phrase}")
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
            if hits:
                first_hit = hits[0]['result']
                title = first_hit.get('title', 'Unknown Title')
                artist = first_hit.get('primary_artist', {}).get('name', 'Unknown Artist')
                logging.info(f"Found title: {title}, artist: {artist} for phrase: {phrase}")
                return title, artist
            else:
                logging.info(f"No title and artist found for phrase: {phrase}")
                return "Unknown Title", "Unknown Artist"
            
        except Exception as e:
            logging.error(f"API Error: {e}")
            return "Unknown Title", "Unknown Artist"

    def score_split(self, phrases: List[str]) -> float:
        logging.info(f"Scoring split for phrases: {phrases}")
        if not all(self.check_phrase_exists(phrase) for phrase in phrases):
            logging.warning("One or more phrases do not exist in the database")
            return 0
            
        length_score = sum(len(phrase) ** 2 for phrase in phrases)
        split_penalty = (len(phrases) - 1) * 5
        score = length_score - split_penalty
        logging.info(f"Calculated score: {score}")
        return score

    def find_best_split(self, text: str) -> Tuple[float, List[str]]:
        logging.info(f"Finding best split for text: '{text}'")
        words = text.split()
        best_score = 0
        best_split = [text]
        
        current_pos = 0
        phrases = []
        
        while current_pos < len(words):
            # Start with minimum 3 words
            current_phrase = " ".join(words[current_pos:current_pos + 3])
            
            if not self.check_phrase_exists(current_phrase):
                # If we can't find even 3 words, move forward by 1
                current_pos += 1
                continue
                
            # Try to extend the phrase
            next_pos = current_pos + 3
            while next_pos < len(words):
                extended_phrase = " ".join(words[current_pos:next_pos + 1])
                if self.check_phrase_exists(extended_phrase):
                    current_phrase = extended_phrase
                    next_pos += 1
                else:
                    break
                    
            phrases.append(current_phrase)
            current_pos = next_pos
            
        if phrases:
            score = self.score_split(phrases)
            if score > best_score:
                best_score = score
                best_split = phrases

        logging.info(f"Best split for text '{text}': Score {best_score}, Splits: {best_split}")
        return best_score, best_split

    def split_lyrics(self, text: str) -> Tuple[float, List[str]]:
        logging.info(f"Splitting lyrics for text: '{text}'")
        self.cache = {}
        return self.find_best_split(text)

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
            title_artist = splitter.retrieve_title_and_artist(phrase)
            if title_artist:
                title, artist = title_artist
                logging.info(f"Phrase '{phrase}' found in '{title}' by {artist}")
            else:
                logging.warning(f"Phrase '{phrase}' not found in Genius database")

if __name__ == "__main__":
    example_usage()