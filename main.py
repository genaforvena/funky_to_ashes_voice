import os
import argparse
import logging
from get_genius_lyrics import LyricsSplitter
from get_captions import get_captions
from get_clips import PhraseExtractor, process_videos
from googleapiclient.discovery import build
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def search_youtube_video_ids(titles_and_artists: List[Tuple[str, str]], api_key: str, max_results: int = 1) -> List[str]:
    logging.info("Starting YouTube video search")
    youtube = build('youtube', 'v3', developerKey=api_key)
    video_ids = []
    
    for title, artist in titles_and_artists:
        search_query = f"{title} {artist}"
        logging.info(f"Searching for: {search_query}")
        search_response = youtube.search().list(
            q=search_query,
            part='id',
            maxResults=max_results,
            type='video'
        ).execute()
        
        if search_response['items']:
            video_id = search_response['items'][0]['id']['videoId']
            logging.info(f"Found video ID: {video_id} for {search_query}")
            video_ids.append(video_id)
        else:
            logging.warning(f"No video found for {search_query}")
    
    return video_ids

def combine_quotes_to_audio(input_text: str, genius_token: str, youtube_api_key: str, output_dir: str = 'output'):
    logging.info("Initializing LyricsSplitter")
    splitter = LyricsSplitter(genius_token)
    
    logging.info("Splitting input text into song quotes")
    score, phrases = splitter.split_lyrics(input_text)
    logging.info(f"Best split: Score {score}: {' | '.join(phrases)}")
    
    titles_and_artists = []
    for phrase in phrases:
        title_artist = splitter.retrieve_title_and_artist(phrase)
        if title_artist:
            logging.info(f"Found title and artist for phrase: {phrase}")
            titles_and_artists.append(title_artist)
        else:
            logging.warning(f"Phrase '{phrase}' not found in Genius database")
    
    logging.info("Searching for YouTube video IDs")
    video_ids = search_youtube_video_ids(titles_and_artists, youtube_api_key)
    
    logging.info("Processing videos to find and extract clips")
    results = process_videos(video_ids, phrases, output_dir)
    
    logging.info("Combining clips into a single audio file")
    combined_audio_path = os.path.join(output_dir, 'combined_audio.mp3')
    combine_audio_clips(results, combined_audio_path)
    
    logging.info(f"Combined audio saved to: {combined_audio_path}")

def combine_audio_clips(results: dict, output_path: str):
    from pydub import AudioSegment
    
    logging.info("Starting audio combination process")
    combined_audio = AudioSegment.empty()
    
    for video_id, matches in results.items():
        for match in matches:
            clip_path = match['clip_path']
            logging.info(f"Adding clip from {clip_path}")
            clip_audio = AudioSegment.from_mp3(clip_path)
            combined_audio += clip_audio
    
    combined_audio.export(output_path, format='mp3')
    logging.info(f"Exported combined audio to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine quotes to audio.')
    parser.add_argument('input_text', type=str, help='The input text containing song quotes.')
    args = parser.parse_args()

    genius_token = os.getenv('GENIUS_TOKEN')
    youtube_api_key = os.getenv('YOUTUBE_API_KEY')
    
    logging.info("Starting the combine quotes to audio process")
    combine_quotes_to_audio(args.input_text, genius_token, youtube_api_key)