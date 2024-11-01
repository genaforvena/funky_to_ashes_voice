import logging
from typing import List, Dict, Optional
import os
from phrase_extractor import PhraseExtractor
from audio_clipper import process_matches
from get_captions import get_captions
import yt_dlp
from pydub import AudioSegment

def download_audio(video_id: str, output_dir: str = 'temp') -> Optional[str]:
    """
    Download audio from YouTube video
    
    Args:
        video_id: YouTube video ID
        output_dir: Directory to save temporary files
        
    Returns:
        Path to downloaded audio file or None if failed
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_id}.mp3")
        
        # Return cached file if it exists
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logging.info(f"Using cached audio for video {video_id}")
            return output_path
        
        logging.info(f"Downloading audio for video {video_id}")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(output_dir, f"{video_id}.%(ext)s"),
            'quiet': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logging.info(f"Successfully downloaded audio for video {video_id}")
            return output_path
            
        logging.error(f"Failed to download audio for video {video_id}")
        return None
        
    except Exception as e:
        logging.error(f"Error downloading audio: {str(e)}")
        return None

def process_video(video_id: str, search_phrases: List[str], captions: List[Dict], output_dir: str) -> List[Dict]:
    """
    Process a single video to extract and save clips
    
    Args:
        video_id: YouTube video ID
        search_phrases: List of phrases to search for
        captions: List of caption dictionaries
        output_dir: Directory to save output clips
        
    Returns:
        List of processed matches with clip paths
    """
    try:
        # Create phrase extractor
        extractor = PhraseExtractor(
            phrases=search_phrases,
            lead_seconds=0.5,
            trail_seconds=1.0,
            min_match_ratio=0.8
        )
        
        # Find matches in captions
        matches = extractor.process_captions(captions)
        if not matches:
            logging.info(f"No matches found in video {video_id}")
            return []
            
        # Download audio
        audio_path = download_audio(video_id)
        if not audio_path:
            logging.error(f"Could not download audio for video {video_id}")
            return []
            
        try:
            # Process matches and create clips
            processed_matches = process_matches(matches, audio_path, output_dir)
            logging.info(f"Processed {len(processed_matches)} clips from video {video_id}")
            
            # Add video ID to matches
            for match in processed_matches:
                match['video_id'] = video_id
                
            return processed_matches
            
        finally:
            # Clean up downloaded audio file
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    logging.debug(f"Cleaned up temporary audio file: {audio_path}")
                except Exception as e:
                    logging.error(f"Error removing audio file: {str(e)}")
                    
    except Exception as e:
        logging.error(f"Error processing video {video_id}: {str(e)}")
        return []

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Example configuration
    video_id = "SsKT0s5J8ko"
    search_phrases = [
        "I could fly home",
        "I remember it all",
    ]
    output_dir = "output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get captions (you'll need to implement or import get_captions function)
    captions = get_captions(video_id)  # Import this from your existing code
    
    if not captions:
        logging.error(f"No captions found for video {video_id}")
        return
    
    # Process video and get results
    results = process_video(video_id, search_phrases, captions, output_dir)
    
    # Print results
    print("\nProcessed clips:")
    print("-" * 50)
    for result in results:
        print(f"\nPhrase: '{result['phrase']}'")
        print(f"Video ID: {result['video_id']}")
        print(f"Similarity: {result['similarity']:.2f}")
        print(f"Duration: {result['end_time'] - result['start_time']:.2f}s")
        print(f"Clip path: {result['clip_path']}")
        print("-" * 30)

if __name__ == "__main__":
    main()