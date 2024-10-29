from typing import List, Dict, Tuple
import yt_dlp
from pydub import AudioSegment
import os
from datetime import timedelta

class PhraseExtractor:
    def __init__(self, phrases: List[str], context_seconds: int = 5):
        """
        Initialize the phrase extractor
        
        Args:
            phrases: List of phrases to search for
            context_seconds: Number of seconds before and after the match to include in clips
        """
        self.phrases = [phrase.lower() for phrase in phrases]
        self.context_seconds = context_seconds
        
    def find_matches(self, captions: List[Dict]) -> List[Tuple[str, float, str]]:
        """
        Find timestamps where phrases appear in captions
        
        Args:
            captions: List of caption dictionaries with 'text' and 'start' keys
            
        Returns:
            List of tuples containing (matched_phrase, timestamp, context)
        """
        matches = []
        
        # Convert captions to lowercase for case-insensitive matching
        caption_texts = [cap['text'].lower() for cap in captions]
        
        for phrase in self.phrases:
            for i, text in enumerate(caption_texts):
                if phrase in text:
                    timestamp = float(captions[i]['start'])
                    context = captions[i]['text']  # Original case preserved for context
                    matches.append((phrase, timestamp, context))
        
        return matches

    def download_audio(self, video_id: str, output_dir: str) -> str:
        """
        Download audio from YouTube video
        
        Args:
            video_id: YouTube video ID
            output_dir: Directory to save the audio file
            
        Returns:
            Path to downloaded audio file
        """
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': f'{output_dir}/{video_id}.%(ext)s'
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
        
        return f'{output_dir}/{video_id}.mp3'

    def extract_clips(self, audio_path: str, matches: List[Tuple[str, float, str]], 
                     output_dir: str) -> List[str]:
        """
        Extract audio clips around each match
        
        Args:
            audio_path: Path to the full audio file
            matches: List of (phrase, timestamp, context) tuples
            output_dir: Directory to save clips
            
        Returns:
            List of paths to extracted clips
        """
        audio = AudioSegment.from_mp3(audio_path)
        clip_paths = []
        
        for i, (phrase, timestamp, _) in enumerate(matches):
            start_ms = max(0, int((timestamp - self.context_seconds) * 1000))
            end_ms = int((timestamp + self.context_seconds) * 1000)
            
            clip = audio[start_ms:end_ms]
            clip_path = f'{output_dir}/clip_{i}_{phrase.replace(" ", "_")}.mp3'
            clip.export(clip_path, format='mp3')
            clip_paths.append(clip_path)
            
        return clip_paths

def process_videos(video_ids: List[str], search_phrases: List[str], 
                  output_dir: str = 'output') -> Dict[str, List[Dict]]:
    """
    Main function to process multiple videos and find phrases
    
    Args:
        video_ids: List of YouTube video IDs
        search_phrases: List of phrases to find
        output_dir: Directory to save output files
        
    Returns:
        Dictionary mapping video IDs to lists of matches with clip information
    """
    os.makedirs(output_dir, exist_ok=True)
    
    extractor = PhraseExtractor(search_phrases)
    results = {}
    
    for video_id in video_ids:
        try:
            # Get captions using the provided function
            captions = get_captions(video_id)
            if not captions:
                print(f"No captions found for video {video_id}")
                continue
                
            # Find matches in captions
            matches = extractor.find_matches(captions)
            if not matches:
                print(f"No matches found in video {video_id}")
                continue
                
            # Download audio and extract clips
            audio_path = extractor.download_audio(video_id, output_dir)
            clip_paths = extractor.extract_clips(audio_path, matches, output_dir)
            
            # Store results
            results[video_id] = [{
                'phrase': phrase,
                'timestamp': str(timedelta(seconds=int(timestamp))),
                'context': context,
                'clip_path': clip_path
            } for (phrase, timestamp, context), clip_path in zip(matches, clip_paths)]
            
            # Clean up full audio file
            os.remove(audio_path)
            
        except Exception as e:
            print(f"Error processing video {video_id}: {str(e)}")
            continue
            
    return results

# Example usage:
if __name__ == "__main__":
    video_ids = ["VIDEO_ID1", "VIDEO_ID2"]
    search_phrases = ["example phrase", "another phrase"]
    
    results = process_videos(video_ids, search_phrases)
    
    # Print results
    for video_id, matches in results.items():
        print(f"\nResults for video {video_id}:")
        for match in matches:
            print(f"Found '{match['phrase']}' at {match['timestamp']}")
            print(f"Clip saved to: {match['clip_path']}")