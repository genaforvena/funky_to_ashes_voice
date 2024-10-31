from typing import List, Dict, Tuple
import yt_dlp
from pydub import AudioSegment
import os
from datetime import timedelta
from get_captions import get_captions
from difflib import SequenceMatcher
import re
import logging

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
        Find sequential matches for parts of the input phrase, starting with the longest possible matches
        """
        logging.info("Starting find_matches")
        merged_text, char_mappings = self.create_merged_text_and_mappings(captions)
        matches = []
        
        # Split transcript into words
        transcript_words = merged_text.lower().split()
        logging.info(f"Transcript length: {len(transcript_words)} words")
        SIMILARITY_THRESHOLD = 0.7
        
        for search_phrase in self.phrases:
            remaining_words = search_phrase.lower().split()
            current_pos = 0
            phrase_matches = []
            
            logging.info(f"\nSearching for phrase: '{search_phrase}' ({len(remaining_words)} words)")
            
            while remaining_words and current_pos < len(transcript_words):
                # Try to match the longest possible portion of remaining words
                found_match = False
                for length in range(len(remaining_words), 0, -1):
                    current_phrase = " ".join(remaining_words[:length])
                    logging.info(f"Trying to match: '{current_phrase}' ({length} words)")
                    
                    # Search through transcript from current position
                    search_pos = current_pos
                    while search_pos <= len(transcript_words) - length:
                        window = " ".join(transcript_words[search_pos:search_pos + length])
                        similarity = SequenceMatcher(None, current_phrase.lower(), window.lower()).ratio()
                        
                        if similarity >= SIMILARITY_THRESHOLD:
                            logging.info(f"Found match with similarity {similarity:.2f}: '{window}'")
                            
                            # Calculate position and timestamps
                            start_pos = len(' '.join(transcript_words[:search_pos]))
                            if search_pos > 0:
                                start_pos += 1
                            end_pos = len(' '.join(transcript_words[:search_pos + length]))
                            
                            start_time = self.find_timestamp_for_position(start_pos, char_mappings)
                            end_time = self.find_timestamp_for_position(end_pos, char_mappings)
                            
                            # Get context
                            context_start = max(0, start_pos - 50)
                            context_end = min(len(merged_text), end_pos + 50)
                            context = merged_text[context_start:context_end]
                            
                            logging.info(f"Match found at time {start_time:.2f}s to {end_time:.2f}s")
                            logging.info(f"Context: ...{context}...")
                            
                            phrase_matches.append((current_phrase, start_time, context, end_time))
                            remaining_words = remaining_words[length:]  # Remove matched words
                            current_pos = search_pos + length  # Continue search from after this match
                            found_match = True
                            break
                        
                        search_pos += 1
                    
                    if found_match:
                        break
                
                if not found_match:
                    logging.info(f"Could not find match for remaining words: '{' '.join(remaining_words)}'")
                    break
            
            # If we found any matches for this phrase
            if phrase_matches:
                # Combine the sequential matches into one
                combined_phrase = " / ".join(match[0] for match in phrase_matches)
                start_time = phrase_matches[0][1]
                end_time = phrase_matches[-1][3]
                context = phrase_matches[-1][2]  # Use the context from the last match
                matches.append((search_phrase, start_time, context, end_time))
                logging.info(f"Combined matches into: '{combined_phrase}'")
        
        logging.info(f"\nFound {len(matches)} total matches")
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

    def save_clip(self, start_time: float, end_time: float, phrase: str) -> str:
        """
        Save audio clip with a sanitized, shortened filename
        """
        # Create a safe, shortened filename
        safe_phrase = phrase.replace('/', '_AND_')  # Replace phrase separator
        # Take first 50 chars of phrase and remove/replace invalid characters
        safe_phrase = re.sub(r'[^\w\s-]', '', safe_phrase[:50])
        safe_phrase = re.sub(r'[-\s]+', '_', safe_phrase).strip('-_')
        
        # Format timestamp
        timestamp = f"{int(start_time//60):02d}-{int(start_time%60):02d}"
        
        # Create final filename
        filename = f"output/clip_{timestamp}_{safe_phrase}.mp3"
        
        # Extract the clip
        clip = self.audio[start_time * 1000:end_time * 1000]
        clip.export(filename, format="mp3")
        
        return filename

def setup_logging():
    """Configure logging to avoid duplicates"""
    logger = logging.getLogger()
    # Clear any existing handlers
    logger.handlers = []
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

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
    setup_logging()
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