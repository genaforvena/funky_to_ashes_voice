from typing import List, Dict, Tuple
import yt_dlp
from pydub import AudioSegment
import os
from datetime import timedelta
from get_captions import get_captions
from difflib import SequenceMatcher
import re
import logging

class PhraseFinder:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.audio = None
        os.makedirs(output_dir, exist_ok=True)
    
    def create_merged_text_and_mappings(self, captions: List[Dict]) -> Tuple[str, List[Tuple[int, float]]]:
        """Create merged text and character to timestamp mappings from captions"""
        merged_text = ""
        char_mappings = []  # List of (char_position, timestamp)
        
        for caption in captions:
            start_pos = len(merged_text)
            if start_pos > 0 and not merged_text.endswith(' '):
                merged_text += ' '
                start_pos += 1
            
            text = caption.get('text', '')
            start_time = float(caption.get('start', 0))
            
            merged_text += text
            char_mappings.append((start_pos, start_time))
        
        return merged_text, char_mappings
    
    def find_timestamp_for_position(self, pos: int, char_mappings: List[Tuple[int, float]]) -> float:
        """Find the timestamp for a given character position"""
        for i in range(len(char_mappings) - 1):
            if char_mappings[i][0] <= pos < char_mappings[i + 1][0]:
                return char_mappings[i][1]
        return char_mappings[-1][1] if char_mappings else 0
    
    def find_matches(self, captions: List[Dict], genius_phrases: List[str]) -> List[Tuple[str, float, str, float]]:
        """Find longest possible matches for each phrase found in Genius"""
        # [Previous find_matches implementation]
        
    def save_clip(self, start_time: float, end_time: float, phrase: str) -> str:
        """Save audio clip with precise cut points"""
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

    def find_matches(self, captions: List[Dict], phrases: List[str]) -> List[Dict]:
        """Find matches for phrases in captions using incremental matching"""
        best_matches = {}  # Dictionary to store best match for each phrase
        merged_text, char_mappings = self.create_merged_text_and_mappings(captions)
        merged_text = merged_text.lower()
        
        logging.info(f"Starting find_matches")
        
        for phrase in phrases:
            words = phrase.lower().split()
            start = 0
            best_similarity = 0
            best_match = None
            
            while start < len(words):
                current_phrase = ""
                last_successful_end = start
                
                # Try increasingly longer segments
                for end in range(start, len(words)):
                    current_phrase = " ".join(words[start:end + 1])
                    logging.info(f"Checking segment: {current_phrase}")
                    
                    # Look for this segment in captions
                    transcript_words = merged_text.split()
                    for i in range(len(transcript_words) - (end - start + 1) + 1):
                        window = " ".join(transcript_words[i:i + (end - start + 1)])
                        similarity = SequenceMatcher(None, window, current_phrase).ratio()
                        
                        if similarity >= 0.8 and similarity > best_similarity:  # Keep track of best match
                            logging.info(f"Found better match with similarity {similarity:.2f}")
                            
                            # Calculate positions and timestamps
                            start_pos = merged_text.find(window)
                            end_pos = start_pos + len(window)
                            start_time = self.find_timestamp_for_position(start_pos, char_mappings)
                            end_time = self.find_timestamp_for_position(end_pos, char_mappings)
                            
                            best_similarity = similarity
                            best_match = {
                                'phrase': current_phrase,
                                'start_time': start_time,
                                'context': window,
                                'end_time': end_time,
                                'similarity': similarity
                            }
                            last_successful_end = end + 1
                
                start = last_successful_end if last_successful_end > start else start + 1
            
            if best_match:
                best_matches[phrase] = best_match
        
        logging.info(f"Found {len(best_matches)} best matches")
        return list(best_matches.values())

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

    def extract_clips(self, audio_path: str, matches: List[Dict], output_dir: str) -> List[str]:
        """
        Extract audio clips around each match with adjusted timing window
        
        Args:
            audio_path: Path to the full audio file
            matches: List of match dictionaries with start_time, end_time, and phrase
            output_dir: Directory to save clips
            
        Returns:
            List of paths to extracted clips
        """
        audio = AudioSegment.from_mp3(audio_path)
        clip_paths = []
        
        for match in matches:
            # Calculate clip boundaries with lead/trail time
            clip_start = max(0, int((match['start_time'] - self.lead_seconds) * 1000))
            clip_end = int((match['end_time'] + self.trail_seconds) * 1000)
            
            clip = audio[clip_start:clip_end]
            
            # Generate filename
            timestamp = str(timedelta(seconds=int(match['start_time']))).replace(':', '-')
            safe_phrase = "".join(x for x in match['phrase'] if x.isalnum() or x.isspace())
            clip_path = f'{output_dir}/clip_{timestamp}_{safe_phrase[:30]}.mp3'
            
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
                matches = self.find_matches(captions, self.phrases)
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
    """
    os.makedirs(output_dir, exist_ok=True)
    
    extractor = PhraseExtractor(search_phrases, lead_seconds, trail_seconds)
    results = {}
    
    for video_id in video_ids:
        try:
            logging.info(f"Processing video {video_id}")
            
            # Get captions using the provided function
            captions = get_captions(video_id)
            if captions is None:
                logging.warning(f"No captions found for video {video_id}. Skipping...")
                continue
                
            # Now we can safely get the length
            logging.info(f"Captions retrieved for video {video_id}: {len(captions)} words")
            
            # Find matches in captions
            matches = extractor.find_matches(captions, extractor.phrases)
            if not matches:
                logging.info(f"No matches found in video {video_id}")
                continue
                
            # Download audio and extract clips
            audio_path = extractor.download_audio(video_id, output_dir)
            if not audio_path:
                logging.error(f"Failed to download audio for video {video_id}")
                continue
                
            clip_paths = extractor.extract_clips(audio_path, matches, output_dir)
            
            # Store results
            results[video_id] = [{
                'phrase': phrase,
                'timestamp': str(timedelta(seconds=int(start_time))),
                'end_time': str(timedelta(seconds=int(end_time))),
                'context': context,
                'clip_path': clip_path
            } for (phrase, start_time, context, end_time), clip_path in zip(matches, clip_paths)]
            
            logging.info(f"Successfully processed video {video_id} with {len(matches)} matches")
            
            # Clean up full audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logging.debug(f"Cleaned up audio file for video {video_id}")
            
        except Exception as e:
            logging.error(f"Error processing video {video_id}: {str(e)}")
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