import os
import argparse
from get_genius_lyrics import LyricsSplitter
from get_captions import get_captions
from get_clips import PhraseExtractor, process_videos
from googleapiclient.discovery import build
from typing import List, Tuple, Dict
import logging
from youtube_verification import verify_youtube_video
from pydub import AudioSegment
from get_clips import download_audio
from youtube_cache import YouTubeCache

youtube_cache = YouTubeCache()

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
    # Check cache first
    cache_key = '_'.join([f"{title}_{artist}" for title, artist in titles_and_artists])
    cached_results = youtube_cache.get_cached_data(cache_key, 'search')
    if cached_results:
        return cached_results.get('video_ids', [])
    
    # If not in cache, perform search and save
    youtube = build('youtube', 'v3', developerKey=api_key)
    video_ids = []
    
    for title, artist in titles_and_artists:
        search_query = f"{title} {artist}"
        logging.info(f"Searching YouTube for: {search_query}")
        
        search_response = youtube.search().list(
            q=search_query,
            part='id',
            maxResults=max_results,
            type='video'
        ).execute()
        
        for item in search_response.get('items', []):
            video_id = item['id']['videoId']
            logging.info(f"Found video ID: {video_id} for query: {search_query}")
            video_ids.append(video_id)
    
    if not video_ids:
        logging.warning("No video IDs found for the given titles and artists.")
    
    youtube_cache.save_to_cache(cache_key, 'search', {'video_ids': video_ids})
    return video_ids

def combine_quotes_to_audio(input_text: str, genius_token: str, youtube_api_key: str, output_dir: str = 'output'):
    # Initialize the LyricsSplitter with the Genius API token
    splitter = LyricsSplitter(genius_token)
    remaining_text = input_text
    all_results = []
    
    # Iteratively process the text with smaller and smaller chunks
    while remaining_text:
        # Split the remaining text into phrases
        score, phrases = splitter.split_lyrics(remaining_text)
        phrases_str = [str(phrase) for phrase in phrases]
        
        print(f"Attempting split: Score {score}: {' | '.join(phrases_str)}")
        
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
        
        # Process videos and get matches
        results = process_videos(video_ids, phrases, output_dir, youtube_api_key)
        
        # Update remaining text by removing matched phrases
        matched_phrases = {result['phrase'] for result in results}
        remaining_phrases = []
        
        for phrase in phrases:
            if str(phrase) not in matched_phrases:
                remaining_phrases.append(str(phrase))
        
        # If we found matches, remove them from remaining_text and continue
        if matched_phrases:
            all_results.extend(results)
            remaining_text = ' '.join(remaining_phrases)
            if remaining_text:
                print(f"Remaining text to process: {remaining_text}")
        else:
            # If no matches found, try with smaller chunks
            if len(phrases) > 1:
                # Reduce chunk size for next iteration
                splitter.reduce_chunk_size()
                print("No matches found, reducing chunk size and retrying...")
            else:
                # If we're already at single words and still no match, skip this word
                print(f"Warning: Could not find match for: {remaining_text}")
                break
    
    # Combine all matched clips into final audio
    combined_audio_path = os.path.join(output_dir, 'combined_audio.mp3')
    combine_audio_clips(all_results, combined_audio_path)
    
    print(f"Combined audio saved to: {combined_audio_path}")

def combine_audio_clips(results, output_path):
    combined_audio = AudioSegment.empty()
    
    for result in results:
        clip_path = result.get('clip_path')
        if clip_path and os.path.exists(clip_path):
            audio_clip = AudioSegment.from_file(clip_path)
            combined_audio += audio_clip
    
    if len(combined_audio) == 0:
        logging.error("No audio clips to combine.")
        return
    
    combined_audio.export(output_path, format='mp3')
    logging.info(f"Combined audio exported to {output_path}")

def process_videos(video_ids: List[str], phrases: List[str], output_dir: str, youtube_api_key: str) -> List[Dict]:
    # Create both temp and output directories
    os.makedirs('temp', exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the extractor with phrases
    extractor = PhraseExtractor(phrases)
    
    # Initialize tracking variables
    processed_phrases = set()
    best_matches = {}
    
    results = []
    for video_id in video_ids:
        try:
            logging.info(f"Processing video {video_id}")
            
            # Get captions and check if they exist
            captions = get_captions(video_id)
            if captions is None:
                logging.warning(f"No captions found for video {video_id}. Skipping...")
                continue
                
            # Find matches using the initialized extractor
            unmatched_phrases = [p for p in phrases if str(p) not in processed_phrases]
            matches = extractor.find_matches(captions, unmatched_phrases)
            
            if not matches:
                logging.info(f"No matches found for phrases in video {video_id}")
                continue
            
            # Download audio only if we found matches
            audio_path = download_audio(video_id)
            if not audio_path:
                logging.warning(f"Failed to download audio for video {video_id}")
                continue
            
            # Extract clips for each match
            clip_paths = extractor.extract_clips(audio_path, matches, output_dir)
            
            # Update best matches and processed phrases
            for match, clip_path in zip(matches, clip_paths):
                phrase = match['phrase']
                similarity = match['similarity']
                
                if (phrase not in best_matches or 
                    similarity > best_matches[phrase]['similarity']):
                    match['clip_path'] = clip_path
                    best_matches[phrase] = match
                    processed_phrases.add(str(phrase))
                    logging.info(f"Found match for '{phrase}' with similarity {similarity:.2f}")
                else:
                    # Clean up unused clip
                    if os.path.exists(clip_path):
                        os.remove(clip_path)
                
        except Exception as e:
            logging.error(f"Error processing video {video_id}: {str(e)}")
            continue
        
        finally:
            # Clean up downloaded audio file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logging.error(f"Error removing audio file {audio_path}: {str(e)}")
    
    # Convert dictionary to list of best matches
    results = list(best_matches.values())
    logging.info(f"Found best matches for {len(results)} phrases")
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
    print(genius_token)
    print(youtube_api_key)
    
    combine_quotes_to_audio(args.input_text, genius_token, youtube_api_key)