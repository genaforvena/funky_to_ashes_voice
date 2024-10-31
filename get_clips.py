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
            
            # Create filename using start_time instead of undefined timestamp
            safe_phrase = "".join(x for x in phrase if x.isalnum() or x.isspace())
            filename = f"clip_{start_time:.2f}_{safe_phrase[:30]}.mp3"
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
        """Find matches for phrases in captions with looser matching criteria"""
        matches = []
        logging.info(f"Starting find_matches with {len(phrases)} phrases")
        
        for phrase in phrases:
            phrase_lower = phrase.lower().strip()
            best_similarity = 0
            best_match = None
            
            # Look for matches in each caption and combinations of captions
            for i in range(len(captions)):
                caption = captions[i]
                caption_text = caption['text'].lower().strip()
                start_time = float(caption['start'])
                
                # Try exact substring match first
                if phrase_lower in caption_text:
                    similarity = 1.0
                else:
                    # Use fuzzy matching with lower threshold
                    similarity = SequenceMatcher(None, caption_text, phrase_lower).ratio()
                
                if similarity >= 0.6 and similarity > best_similarity:
                    logging.info(f"Found match for '{phrase}' with similarity {similarity:.2f}")
                    
                    # Calculate end time
                    if i + 1 < len(captions):
                        end_time = float(captions[i + 1]['start'])
                    else:
                        end_time = float(caption.get('end', start_time + 5))
                    
                    best_similarity = similarity
                    best_match = {
                        'phrase': phrase,
                        'start_time': start_time,
                        'end_time': end_time,
                        'context': caption['text'],
                        'similarity': similarity
                    }
                
                # Try combining with next captions (up to 3)
                combined_text = caption_text
                combined_start = start_time
                
                for j in range(i + 1, min(i + 3, len(captions))):
                    combined_text += " " + captions[j]['text'].lower().strip()
                    
                    if phrase_lower in combined_text:
                        similarity = 1.0
                    else:
                        similarity = SequenceMatcher(None, combined_text, phrase_lower).ratio()
                    
                    if similarity >= 0.6 and similarity > best_similarity:
                        logging.info(f"Found multi-caption match for '{phrase}' with similarity {similarity:.2f}")
                        
                        if j + 1 < len(captions):
                            end_time = float(captions[j + 1]['start'])
                        else:
                            end_time = float(captions[j].get('end', float(captions[j]['start']) + 5))
                        
                        best_similarity = similarity
                        best_match = {
                            'phrase': phrase,
                            'start_time': combined_start,
                            'end_time': end_time,
                            'context': combined_text,
                            'similarity': similarity
                        }
            
            if best_match:
                matches.append(best_match)
                logging.info(f"Added match for phrase: '{phrase}' ({best_match['similarity']:.2f})")
            else:
                logging.info(f"No match found for phrase: '{phrase}'")
        
        logging.info(f"Found total {len(matches)} matches")
        return matches

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

def download_audio(video_id: str, output_dir: str = 'temp') -> str:
    """Download audio from a YouTube video using yt-dlp"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_template,
            'quiet': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        
        output_path = os.path.join(output_dir, f"{video_id}.mp3")
        if os.path.exists(output_path):
            logging.info(f"Successfully downloaded audio for video {video_id}")
            return output_path
        else:
            logging.error(f"Output file not found at {output_path}")
            return None
            
    except Exception as e:
        logging.error(f"Error downloading audio for video {video_id}: {str(e)}")
        return None

def process_videos(video_ids: List[str], phrases: List[str], output_dir: str, youtube_api_key: str) -> List[Dict]:
    """Process videos to find and extract matching phrases"""
    # Create directories
    os.makedirs('temp', exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    extractor = PhraseExtractor(phrases)
    best_matches = {}
    processed_phrases = set()
    
    for video_id in video_ids:
        try:
            if len(processed_phrases) == len(phrases):
                break
                
            logging.info(f"Processing video {video_id}")
            audio_path = None
            
            captions = get_captions(video_id)
            if captions is None:
                logging.warning(f"No captions found for video {video_id}. Skipping...")
                continue
            
            unmatched_phrases = [p for p in phrases if str(p) not in processed_phrases]
            matches = extractor.find_matches(captions, unmatched_phrases)
            
            if not matches:
                logging.info(f"No matches found for phrases in video {video_id}")
                continue
            
            audio_path = download_audio(video_id)
            if not audio_path:
                logging.warning(f"Failed to download audio for video {video_id}")
                continue
            
            # Use the extract_clips method from PhraseExtractor
            clip_paths = extractor.extract_clips(audio_path, matches, output_dir)
            
            # Update matches with clip paths
            for match, clip_path in zip(matches, clip_paths):
                phrase = match['phrase']
                similarity = match['similarity']
                
                if phrase not in best_matches or similarity > best_matches[phrase]['similarity']:
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
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logging.error(f"Error removing audio file {audio_path}: {str(e)}")
    
    results = list(best_matches.values())
    logging.info(f"Found best matches for {len(results)} phrases")
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