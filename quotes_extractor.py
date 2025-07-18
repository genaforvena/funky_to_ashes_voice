import lyricsgenius
import time
import os
import datetime

GENIUS_API_TOKEN = os.getenv("GENIUS_TOKEN")

def log_message(message):
    print(f"[{datetime.datetime.now()}] {message}")

def find_longest_phrase_matches(input_text):
    """
    Finds the longest phrases in the input text that match song lyrics from Genius.
    """
    words = input_text.lower().split()
    phrases = [" ".join(words[i:j+1]) for i in range(len(words)) for j in range(i, len(words))]
    phrases.sort(key=len, reverse=True) # Prioritize longer phrases

    matches = []
    found_phrases = set()

    # Limit the number of phrases to search for
    phrases = [p for p in phrases if len(p.split()) >= 3 and len(p.split()) <= 5]
    phrases = phrases[:20]

    genius = lyricsgenius.Genius(GENIUS_API_TOKEN, timeout=15, retries=3, sleep_time=0.5)
    genius.remove_section_headers = True
    genius.skip_non_songs = True

    for phrase in phrases:
        if any(p.startswith(phrase) for p in found_phrases):
            continue # Skip if a longer version of this phrase has been found

        try:
            log_message(f"Searching for phrase: '{phrase}'")
            results = genius.search_songs(phrase, per_page=5)
            log_message(f"Finished searching for phrase: '{phrase}'")
            for hit in results['hits']:
                song = hit['result']
                log_message(f"Fetching lyrics for '{song['title']}' by {song['primary_artist']['name']}")
                lyrics = genius.lyrics(song_url=song['url'])
                log_message(f"Finished fetching lyrics for '{song['title']}' by {song['primary_artist']['name']}")
                if lyrics and phrase in lyrics.lower():
                    log_message(f"Found match for '{phrase}' in '{song['title']}' by {song['primary_artist']['name']}")
                    matches.append({
                        'phrase': phrase,
                        'title': song['title'],
                        'artist': song['primary_artist']['name'],
                        'url': song['url']
                    })
                    found_phrases.add(phrase)
                    break # Move to the next phrase after finding one match
            time.sleep(0.5) # Rate limiting
        except Exception as e:
            log_message(f"Error searching for phrase '{phrase}': {e}")

    return matches

if __name__ == "__main__":
    input_text = "We have black holes laughing after every word of the black hole you are in is saying."
    matches = find_longest_phrase_matches(input_text)
    for match in matches:
        print(f"Phrase: '{match['phrase']}' found in '{match['title']}' by {match['artist']}")
