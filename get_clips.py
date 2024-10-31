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
        Find matches for phrases from Genius in the captions
        """
        logging.info("Starting find_matches")
        merged_text, char_mappings = self.create_merged_text_and_mappings(captions)
        matches = []
        
        # Split transcript into words
        transcript_words = merged_text.lower().split()
        logging.info(f"Transcript length: {len(transcript_words)} words")
        
        for search_phrase in self.phrases:
            # Split the search phrase into smaller chunks
            phrase_parts = search_phrase.lower().split(',')  # Split on commas
            for part in phrase_parts:
                # Further split into smaller meaningful chunks
                sub_parts = part.strip().split(' ')
                
                # Look for chunks of 3-4 words that might match
                for i in range(len(sub_parts)):
                    for chunk_size in [4, 3, 2]:
                        if i + chunk_size <= len(sub_parts):
                            chunk = ' '.join(sub_parts[i:i+chunk_size])
                            logging.info(f"Searching for chunk: '{chunk}'")
                            
                            # Look for this chunk in transcript
                            current_pos = 0
                            while current_pos <= len(transcript_words) - chunk_size:
                                window = ' '.join(transcript_words[current_pos:current_pos + chunk_size])
                                
                                if window.lower() == chunk.lower():
                                    logging.info(f"Found matching chunk: '{window}'")
                                    
                                    # Calculate exact positions
                                    full_text_before = ' '.join(transcript_words[:current_pos])
                                    start_pos = len(full_text_before)
                                    if current_pos > 0:
                                        start_pos += 1
                                    end_pos = start_pos + len(window)
                                    
                                    # Get timestamps
                                    start_time = self.find_timestamp_for_position(start_pos, char_mappings)
                                    end_time = self.find_timestamp_for_position(end_pos, char_mappings)
                                    
                                    logging.info(f"Match found at time {start_time:.2f}s to {end_time:.2f}s")
                                    logging.info(f"Original text: '{window}'")
                                    
                                    matches.append((chunk, start_time, window, end_time))
                                    break  # Found a match for this chunk
                                
                                current_pos += 1
                            
                            if matches and matches[-1][0] == chunk:
                                break  # Found a match, move to next part
        
        logging.info(f"\nFound {len(matches)} matches")
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
        Save audio clip with precise cut points
        """
        try:
            # Convert seconds to milliseconds for pydub
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            # Extract exactly the phrase duration
            clip = self.audio[start_ms:end_ms]
            
            # Create filename from phrase
            safe_phrase = "".join(x for x in phrase if x.isalnum() or x.isspace())
            filename = f"clip_{safe_phrase[:30]}_{start_time:.2f}_{end_time:.2f}.mp3"
            output_path = os.path.join(self.output_dir, filename)
            
            # Export with exact boundaries
            clip.export(output_path, format="mp3")
            logging.info(f"Saved clip: {filename} ({(end_time - start_time):.2f}s)")
            
            return output_path
            
        except Exception as e:
            logging.error(f"Error saving clip: {e}")
            return None

    def process_videos(self, video_ids: List[str], phrases: List[str], output_dir: str) -> Dict:
        """Process all downloaded videos and find matches in each"""
        self.phrases = phrases
        results = {}
        
        for video_id in video_ids:
            logging.info(f"\nProcessing video {video_id}")
            try:
                # Get captions for this video
                captions = get_captions(video_id)
                if not captions:
                    logging.warning(f"No captions found for video {video_id}")
                    continue
                
                # Download audio for this video
                audio_path = self.download_audio(video_id, output_dir)
                if not audio_path:
                    logging.warning(f"Could not download audio for video {video_id}")
                    continue
                
                self.audio = AudioSegment.from_file(audio_path)
                
                # Find matches in this video
                matches = self.find_matches(captions)
                if matches:
                    # Save clips for each match
                    video_results = []
                    for phrase, start_time, context, end_time in matches:
                        clip_path = self.save_clip(start_time, end_time, phrase)
                        video_results.append({
                            'phrase': phrase,
                            'start_time': start_time,
                            'end_time': end_time,
                            'context': context,
                            'clip_path': clip_path
                        })
                    results[video_id] = video_results
                    logging.info(f"Found and saved {len(video_results)} clips from video {video_id}")
                else:
                    logging.info(f"No matches found in video {video_id}")
                
            except Exception as e:
                logging.error(f"Error processing video {video_id}: {str(e)}")
                continue
                
            finally:
                # Clean up downloaded audio file
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
        
        return results

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