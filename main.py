import os
from get_genius_lyrics import LyricsSplitter
from get_captions import get_captions
from get_clips import PhraseExtractor, process_videos
from googleapiclient.discovery import build

def search_youtube_videos(song_title: str, artist_name: str, api_key: str, max_results: int = 5):
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    # Search query combining song title and artist name
    search_query = f"{song_title} {artist_name}"
    
    # Perform the search
    search_response = youtube.search().list(
        q=search_query,
        part='id,snippet',
        maxResults=max_results,
        type='video'
    ).execute()
    
    # Extract video IDs from the search results
    video_ids = [item['id']['videoId'] for item in search_response['items']]
    
    return video_ids

def combine_quotes_to_audio(input_text: str, genius_token: str, youtube_api_key: str, output_dir: str = 'output'):
    # Initialize the LyricsSplitter with the Genius API token
    splitter = LyricsSplitter(genius_token)
    
    # Split the input text into song quotes
    score, phrases = splitter.split_lyrics(input_text)
    print(f"Best split: Score {score}: {' | '.join(phrases)}")
    
    # Initialize the PhraseExtractor
    extractor = PhraseExtractor(phrases)
    
    # Search for YouTube video IDs based on song title and artist name
    video_ids = []
    for phrase in phrases:
        # Assuming each phrase is in the format "Song Title - Artist Name"
        song_title, artist_name = phrase.split(' - ')
        video_ids.extend(search_youtube_videos(song_title, artist_name, youtube_api_key))
    
    # Process each video to find and extract clips
    results = process_videos(video_ids, phrases, output_dir)
    
    # Combine the clips into a single audio file
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
    # Example usage
    input_text = "the future is now today"
    genius_token = os.getenv('GENIUS_TOKEN')
    youtube_api_key = os.getenv('YOUTUBE_API_KEY')
    
    combine_quotes_to_audio(input_text, genius_token, youtube_api_key)