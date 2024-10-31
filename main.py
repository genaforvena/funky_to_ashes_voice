import os
import argparse
from get_genius_lyrics import LyricsSplitter
from get_captions import get_captions
from get_clips import PhraseExtractor, process_videos
from googleapiclient.discovery import build
from typing import List, Tuple
import logging
from youtube_verification import verify_youtube_video

def search_youtube_video_ids(titles_and_artists: List[Tuple[str, str]], api_key: str, max_results: int = 5) -> List[str]:
    """
    Search YouTube for videos matching the given titles and artists and return video IDs.
    
    Args:
        titles_and_artists: List of tuples containing (title, artist)
        api_key: YouTube API key
        max_results: Maximum number of results to return per title-artist pair
    
    Returns:
        List of video IDs
    """
    youtube = build('youtube', 'v3', developerKey=api_key)
    video_ids = []
    
    for title, artist in titles_and_artists:
        search_query = f"{title} {artist}"
        search_response = youtube.search().list(
            q=search_query,
            part='id',
            maxResults=max_results,
            type='video'
        ).execute()
        
        for item in search_response['items']:
            video_id = item['id']['videoId']
            video_ids.append(video_id)
    
    return video_ids

def combine_quotes_to_audio(input_text: str, genius_token: str, youtube_api_key: str, output_dir: str = 'output'):
    # Initialize the LyricsSplitter with the Genius API token
    splitter = LyricsSplitter(genius_token)
    
    # Assuming split_lyrics returns a tuple (score, list_of_phrases)
    score, phrases = splitter.split_lyrics(input_text)
    
    # Convert all elements in phrases to strings before joining
    phrases_str = [str(phrase) for phrase in phrases]
    
    print(f"Best split: Score {score}: {' | '.join(phrases_str)}")
    
    # Get title and artist for each phrase
    titles_and_artists = []
    for phrase in phrases:
        title_artist = splitter.get_title_and_artist(phrase)
        if title_artist:
            titles_and_artists.append(title_artist)
        else:
            print(f"Phrase '{phrase}' not found in Genius database")
    
    # Search for YouTube video IDs based on titles and artists
    video_ids = search_youtube_video_ids(titles_and_artists, youtube_api_key)
    
    # Process each video to find and extract clips
    results = process_videos(video_ids, phrases, output_dir, youtube_api_key)
    
    # Combine the clips into a single audio file with a "chopped and screwed" aesthetic
    combined_audio_path = os.path.join(output_dir, 'combined_audio.mp3')
    combine_audio_clips(results, combined_audio_path)
    
    print(f"Combined audio saved to: {combined_audio_path}")

def combine_audio_clips(results: dict, output_path: str):
    from pydub import AudioSegment
    
    combined_audio = AudioSegment.empty()
    
    for video_id, matches in results.items():
        for match in matches:
            clip_path = match['clip_path']
            clip_audio = AudioSegment.from_mp3(clip_path)
            combined_audio += clip_audio
    
    combined_audio.export(output_path, format='mp3')

def process_videos(video_ids, phrases, output_dir, youtube_api_key):
    extractor = PhraseExtractor(phrases)
    results = {}
    
    for video_id in video_ids:
        expected_title, expected_artist = get_expected_track_info(phrases)
        
        if not verify_youtube_video(video_id, expected_title, expected_artist, youtube_api_key):
            logging.warning(f"Skipping video {video_id} as it does not match the expected track.")
            continue
        
        captions = get_captions(video_id)
        if not captions:
            logging.warning(f"No captions found for video {video_id}")
            continue
        
        matches = extractor.find_matches(captions, phrases)
        if not matches:
            logging.info(f"No matches found for phrases in video {video_id}")
            continue
        
        # Process matches if found
        # Your logic to handle matches
        results[video_id] = matches
    
    return results

def get_expected_track_info(phrases):
    """
    Determine the expected title and artist for each phrase.
    This is a placeholder implementation and should be replaced with your actual logic.
    """
    # Example logic: Assume the first phrase contains the title and artist
    # This should be replaced with your actual logic to determine the title and artist
    if phrases:
        # For demonstration, let's assume the first phrase contains the title and artist
        # In practice, you might have a more complex logic to determine this
        title_artist = phrases[0].split(' - ')
        if len(title_artist) == 2:
            return title_artist[0], title_artist[1]
    
    # Default return if no valid title and artist found
    return "Unknown Title", "Unknown Artist"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine quotes to audio.')
    parser.add_argument('input_text', type=str, help='The input text containing song quotes.')
    args = parser.parse_args()

    genius_token = os.getenv('GENIUS_TOKEN')
    youtube_api_key = os.getenv('YOUTUBE_API_KEY')
    
    combine_quotes_to_audio(args.input_text, genius_token, youtube_api_key)