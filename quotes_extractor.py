import lyricsgenius
import time
import os

GENIUS_API_TOKEN = os.getenv("GENIUS_TOKEN")

def find_longest_phrase_matches(input_text):
    words = input_text.split()
    matches = []
    cache = {}

    # Initialize Genius client with your API token
    genius = lyricsgenius.Genius(GENIUS_API_TOKEN)
    genius.timeout = 15  # Set timeout in seconds
    genius.retries = 3   # Number of retries in case of timeout
    genius.sleep_time = 1  # Time to wait between requests to prevent rate limiting
    genius.remove_section_headers = True  # Remove section headers from lyrics
    genius.skip_non_songs = True  # Skip tracks that aren't songs
    genius.excluded_terms = ["(Remix)", "(Live)"]  # Exclude songs with these terms

    # Generate all possible phrases, starting with the longest
    for length in range(len(words), 0, -1):
        for start in range(len(words) - length + 1):
            end = start + length
            phrase = " ".join(words[start:end])

            if phrase in cache:
                continue  # Skip if already searched
            cache[phrase] = True

            try:
                # Use search_songs to get up to 10 matches
                result = genius.search_songs(phrase, per_page=10)
                song_info_list = []
                if result and 'hits' in result:
                    for hit in result['hits']:
                        song_data = hit['result']
                        song_title = song_data['title']
                        song_artist = song_data['primary_artist']['name']
                        song_url = song_data['url']

                        # Fetch the lyrics using the song's URL
                        song_lyrics = genius.lyrics(song_url=song_url)
                        if song_lyrics:
                            song_lyrics_lower = song_lyrics.lower()
                            # Check if the phrase is in the song's lyrics
                            if phrase.lower() in song_lyrics_lower:
                                song_info = {
                                    'title': song_title,
                                    'artist': song_artist,
                                    'url': song_url
                                }
                                song_info_list.append(song_info)
                if song_info_list:
                    # Append the exact matched phrase and its song info
                    matches.append((phrase, song_info_list))
            except Exception as e:
                print(f"An error occurred while searching for '{phrase}': {e}")
            time.sleep(1)  # Delay to prevent rate limiting

    # Sort matches by the length of the phrase, from longest to shortest
    matches.sort(key=lambda x: -len(x[0]))
    return matches


if __name__ == "__main__":
    input_text = "We have black wholes laughing after every word of the black hole you are in is saying."
    matches = find_longest_phrase_matches(input_text)
    for phrase, songs in matches:
        print(f"Phrase: '{phrase}'")
        for song in songs:
            print(f"  - {song['title']} by {song['artist']} ({song['url']})")
