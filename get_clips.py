from typing import List, Dict, Tuple
import yt_dlp
from pydub import AudioSegment
import os
from datetime import timedelta
from get_captions import get_captions
from difflib import SequenceMatcher

class PhraseExtractor:
    def __init__(self, phrases: List[str], lead_seconds: int = 0, trail_seconds: int = 2):
        """
        Initialize the phrase extractor
        
        Args:
            phrases: List of phrases to search for
            lead_seconds: Number of seconds before the match to include in clips
            trail_seconds: Number of seconds after the match to include in clips
        """
        self.phrases = [phrase.lower() for phrase in phrases]
        self.lead_seconds = lead_seconds
        self.trail_seconds = trail_seconds

    def create_merged_text_and_mappings(self, captions: List[Dict]) -> Tuple[str, List[Tuple[int, float]]]:
        """
        Create a single string from all captions while maintaining character to timestamp mappings
        
        Args:
            captions: List of caption dictionaries with 'text' and 'start' keys
            
        Returns:
            Tuple of (merged_text, list of (character_index, timestamp) mappings)
        """
        merged_text = ""
        char_mappings = []  # List of (char_index, timestamp) tuples
        
        for caption in captions:
            # Store mapping for the start of this caption text
            char_mappings.append((len(merged_text), float(caption['start'])))
            
            # Add the caption text with a space to separate captions
            merged_text += caption['text'] + " "
        
        # Add a final mapping for the end of the text
        if captions:
            last_caption = captions[-1]
            end_time = last_caption.get('end', float(last_caption['start']) + 5)  # Default 5 sec if no end time
            char_mappings.append((len(merged_text), end_time))
            
        return merged_text.lower(), char_mappings

    def find_timestamp_for_position(self, position: int, char_mappings: List[Tuple[int, float]]) -> float:
        """
        Find the timestamp for a given character position using the mappings
        
        Args:
            position: Character position in the merged text
            char_mappings: List of (char_index, timestamp) tuples
            
        Returns:
            Estimated timestamp for the character position
        """
        # Find the two mappings that bracket this position
        for i in range(len(char_mappings) - 1):
            start_idx, start_time = char_mappings[i]
            end_idx, end_time = char_mappings[i + 1]
            
            if start_idx <= position < end_idx:
                # Linear interpolation between the two timestamps
                char_progress = (position - start_idx) / (end_idx - start_idx)
                return start_time + (end_time - start_time) * char_progress
                
        return char_mappings[-1][1]  # Return last timestamp if position is beyond end

    def find_matches(self, captions: List[Dict]) -> List[Tuple[str, float, str, float]]:
        """
        Find timestamps where phrases appear in captions, allowing for partial matches
        
        Args:
            captions: List of caption dictionaries with 'text' and 'start' keys
            
        Returns:
            List of tuples containing (matched_phrase, start_time, context, end_time)
        """
        merged_text, char_mappings = self.create_merged_text_and_mappings(captions)
        matches = []
        
        # Split merged text into words for more efficient searching
        words = merged_text.split()
        SIMILARITY_THRESHOLD = 0.7  # 70% similarity required
        
        for phrase in self.phrases:
            phrase_words = phrase.split()
            phrase_len = len(phrase_words)
            
            # Slide through the text word by word
            for i in range(len(words) - phrase_len + 1):
                # Get the candidate text of the same length as our phrase
                candidate = ' '.join(words[i:i + phrase_len])
                
                # Calculate similarity ratio
                similarity = SequenceMatcher(None, phrase, candidate).ratio()
                
                if similarity >= SIMILARITY_THRESHOLD:
                    # Find position in original merged text
                    pos = len(' '.join(words[:i]))
                    if i > 0:
                        pos += 1  # Account for space
                        
                    # Get timestamps for the start and end of the match
                    start_time = self.find_timestamp_for_position(pos, char_mappings)
                    end_time = self.find_timestamp_for_position(pos + len(candidate), char_mappings)
                    
                    # Extract context
                    context_start = max(0, pos - 50)
                    context_end = min(len(merged_text), pos + len(candidate) + 50)
                    context = merged_text[context_start:context_end]
                    
                    matches.append((phrase, start_time, context, end_time))
        
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
        Extract audio clips around each match with adjusted timing window
        
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
            # Start 4 seconds before the phrase starts
            clip_start = max(0, int((start_time - self.lead_seconds) * 1000))
            # End 2 seconds after the phrase ends
            clip_end = int((end_time + self.trail_seconds) * 1000)
            
            clip = audio[clip_start:clip_end]
            
            # Generate a filename that includes timing information
            timestamp = str(timedelta(seconds=int(start_time))).replace(':', '-')
            clip_path = f'{output_dir}/clip_{timestamp}_{phrase.replace(" ", "_")}.mp3'
            clip.export(clip_path, format='mp3')
            clip_paths.append(clip_path)
            
        return clip_paths

def process_videos(video_ids: List[str], search_phrases: List[str], 
                  output_dir: str = 'output',
                  lead_seconds: int = 1,
                  trail_seconds: int = 2) -> Dict[str, List[Dict]]:
    """
    Main function to process multiple videos and find phrases
    
    Args:
        video_ids: List of YouTube video IDs
        search_phrases: List of phrases to find
        output_dir: Directory to save output files
        lead_seconds: Number of seconds to include before each match
        trail_seconds: Number of seconds to include after each match
        
    Returns:
        Dictionary mapping video IDs to lists of matches with clip information
    """
    os.makedirs(output_dir, exist_ok=True)
    
    extractor = PhraseExtractor(search_phrases, lead_seconds, trail_seconds)
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
    search_phrases = ["I could fly home", "I remember it all", "letter from the government", "I dwell in my cell"]
    
    results = process_videos(video_ids, search_phrases)
    
    # Print results
    for video_id, matches in results.items():
        print(f"\nResults for video {video_id}:")
        for match in matches:
            print(f"Found '{match['phrase']}' at {match['timestamp']} - {match['end_time']}")
            print(f"Context: {match['context']}")
            print(f"Clip saved to: {match['clip_path']}")