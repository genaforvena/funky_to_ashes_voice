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
            for hit in hits:
                if phrase.lower() in hit['result']['full_title'].lower():
                    logging.info(f"Phrase found in Genius database: {phrase}")
                    return True
            logging.info(f"Phrase not found in Genius database: {phrase}")
            return False
            
        except Exception as e:
            logging.error(f"API Error: {e}")
            return False

    def get_title_and_artist(self, phrase: str) -> Optional[Tuple[str, str]]:
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
            for hit in hits:
                if phrase.lower() in hit['result']['full_title'].lower():
                    title = hit['result']['title']
                    artist = hit['result']['primary_artist']['name']
                    logging.info(f"Found title '{title}' and artist '{artist}' for phrase: {phrase}")
                    return title, artist
            logging.info(f"No title and artist found for phrase: {phrase}")
            return None
            
        except Exception as e:
            logging.error(f"API Error: {e}")
            return None

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

    def find_best_split(self, text: str, depth: int = 0, max_depth: int = 3) -> Tuple[float, List[str]]:
        logging.info(f"Finding best split for text: '{text}' at depth {depth}")
        if depth >= max_depth:
            logging.info("Reached maximum recursion depth")
            return 0, [text]
            
        if self.check_phrase_exists(text):
            whole_score = self.score_split([text])
            if whole_score > 0:
                logging.info(f"Whole text is a valid phrase with score: {whole_score}")
                return whole_score, [text]
        
        cache_key = text
        if cache_key in self.cache:
            logging.info(f"Cache hit for text: '{text}'")
            return self.cache[cache_key]
            
        best_score = 0
        best_split = [text]
        
        words = text.split()
        for i in range(1, len(words)):
            left = " ".join(words[:i])
            right = " ".join(words[i:])
            
            left_score, left_splits = self.find_best_split(left, depth + 1, max_depth)
            right_score, right_splits = self.find_best_split(right, depth + 1, max_depth)
            
            combined_score = left_score + right_score
            if combined_score > best_score:
                best_score = combined_score
                best_split = left_splits + right_splits
        
        self.cache[cache_key] = (best_score, best_split)
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
            title_artist = splitter.get_title_and_artist(phrase)
            if title_artist:
                title, artist = title_artist
                logging.info(f"Phrase '{phrase}' found in '{title}' by {artist}")
            else:
                logging.warning(f"Phrase '{phrase}' not found in Genius database")

if __name__ == "__main__":
    example_usage()