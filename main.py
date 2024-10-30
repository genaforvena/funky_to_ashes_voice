import os
import argparse
from get_genius_lyrics import LyricsSplitter
from get_captions import get_captions
from get_clips import PhraseExtractor, process_videos
from googleapiclient.discovery import build
from typing import List, Tuple

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
    
    # Split the input text into song quotes
    score, phrases = splitter.split_lyrics(input_text)
    print(f"Best split: Score {score}: {' | '.join(phrases)}")
    
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
    results = process_videos(video_ids, phrases, output_dir, youtube_api_key=youtube_api_key)
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine quotes to audio.')
    parser.add_argument('input_text', type=str, help='The input text containing song quotes.')
    args = parser.parse_args()

    genius_token = os.getenv('GENIUS_TOKEN')
    youtube_api_key = os.getenv('YOUTUBE_API_KEY')
    
    combine_quotes_to_audio(args.input_text, genius_token, youtube_api_key)