from typing import List, Dict, Tuple
import yt_dlp
from pydub import AudioSegment
import os
from datetime import timedelta
from get_captions import get_captions

class PhraseExtractor:
    def __init__(self, phrases: List[str], context_seconds: int = 5, max_phrase_gap: float = 2.0):
        """
        Initialize the phrase extractor
        
        Args:
            phrases: List of phrases to search for
            context_seconds: Number of seconds before and after the match to include in clips
            max_phrase_gap: Maximum time gap (in seconds) between caption segments to consider them connected
        """
        self.phrases = [phrase.lower() for phrase in phrases]
        self.context_seconds = context_seconds
        self.max_phrase_gap = max_phrase_gap
        
    def create_caption_windows(self, captions: List[Dict]) -> List[Dict]:
        """
        Create sliding windows of concatenated captions to detect split phrases
        
        Args:
            captions: List of caption dictionaries with 'text' and 'start' keys
            
        Returns:
            List of dictionaries containing merged caption windows
        """
        windows = []
        
        for i in range(len(captions)):
            current_window = {
                'text': captions[i]['text'],
                'start': captions[i]['start'],
                'end': captions[i].get('end', captions[i]['start'] + 5),  # Default 5 sec if no end time
                'segments': [i]
            }
            
            # Look ahead and merge with subsequent captions if they're close enough
            j = i + 1
            while j < len(captions):
                time_gap = float(captions[j]['start']) - float(current_window['end'])
                
                if time_gap > self.max_phrase_gap:
                    break
                    
                current_window['text'] += ' ' + captions[j]['text']
                current_window['end'] = captions[j].get('end', float(captions[j]['start']) + 5)
                current_window['segments'].append(j)
                j += 1
                
            windows.append(current_window)
            
        return windows

    def find_matches(self, captions: List[Dict]) -> List[Tuple[str, float, str, float]]:
        """
        Find timestamps where phrases appear in captions, including split phrases
        
        Args:
            captions: List of caption dictionaries with 'text' and 'start' keys
            
        Returns:
            List of tuples containing (matched_phrase, start_time, context, end_time)
        """
        matches = []
        caption_windows = self.create_caption_windows(captions)
        
        for window in caption_windows:
            window_text = window['text'].lower()
            
            for phrase in self.phrases:
                if phrase in window_text:
                    # Calculate the relative position of the phrase in the window text
                    # to get a more accurate timestamp
                    phrase_pos = window_text.find(phrase)
                    chars_before = len(window_text[:phrase_pos])
                    total_chars = len(window_text)
                    
                    # Estimate the timestamp based on position in the text
                    time_span = float(window['end']) - float(window['start'])
                    relative_time = time_span * (chars_before / total_chars)
                    match_timestamp = float(window['start']) + relative_time
                    
                    matches.append((
                        phrase,
                        match_timestamp,
                        window['text'],  # Original case preserved for context
                        float(window['end'])
                    ))
        
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

    def extract_clips(self, audio_path: str, matches: List[Tuple[str, float, str, float]], 
                     output_dir: str) -> List[str]:
        """
        Extract audio clips around each match
        
        Args:
            audio_path: Path to the full audio file
            matches: List of (phrase, start_timestamp, context, end_timestamp) tuples
            output_dir: Directory to save clips
            
        Returns:
            List of paths to extracted clips
        """
        audio = AudioSegment.from_mp3(audio_path)
        clip_paths = []
        
        for i, (phrase, start_time, _, end_time) in enumerate(matches):
            # Calculate clip boundaries
            clip_start = max(0, int((start_time - self.context_seconds) * 1000))
            clip_end = int((end_time + self.context_seconds) * 1000)
            
            clip = audio[clip_start:clip_end]
            clip_path = f'{output_dir}/clip_{i}_{phrase.replace(" ", "_")}.mp3'
            clip.export(clip_path, format='mp3')
            clip_paths.append(clip_path)
            
        return clip_paths

def process_videos(video_ids: List[str], search_phrases: List[str], 
                  output_dir: str = 'output', max_phrase_gap: float = 2.0) -> Dict[str, List[Dict]]:
    """
    Main function to process multiple videos and find phrases
    
    Args:
        video_ids: List of YouTube video IDs
        search_phrases: List of phrases to find
        output_dir: Directory to save output files
        max_phrase_gap: Maximum time gap between captions to consider them connected
        
    Returns:
        Dictionary mapping video IDs to lists of matches with clip information
    """
    os.makedirs(output_dir, exist_ok=True)
    
    extractor = PhraseExtractor(search_phrases, max_phrase_gap=max_phrase_gap)
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
                'timestamp': str(timedelta(seconds=int(start_time))),
                'end_time': str(timedelta(seconds=int(end_time))),
                'context': context,
                'clip_path': clip_path
            } for (phrase, start_time, context, end_time), clip_path in zip(matches, clip_paths)]
            
            # Clean up full audio file
            os.remove(audio_path)
            
        except Exception as e:
            print(f"Error processing video {video_id}: {str(e)}")
            continue
            
    return results

# Example usage:
if __name__ == "__main__":
    video_ids = ["ZM5_6js19eM", "SsKT0s5J8ko"]
    search_phrases = ["switched the time zone", "could find me"]
    
    results = process_videos(video_ids, search_phrases, max_phrase_gap=2.0)
    
    # Print results
    for video_id, matches in results.items():
        print(f"\nResults for video {video_id}:")
        for match in matches:
            print(f"Found '{match['phrase']}' at {match['timestamp']} - {match['end_time']}")
            print(f"Context: {match['context']}")
            print(f"Clip saved to: {match['clip_path']}")