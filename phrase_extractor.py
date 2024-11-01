import re
from typing import List, Dict, Optional, Set
from difflib import SequenceMatcher
import logging
import os
import yt_dlp
from pydub import AudioSegment


class PhraseExtractor:
    def __init__(self, phrases: List[str], lead_seconds: float = 0.5, trail_seconds: float = 1.0, min_match_ratio: float = 0.8):
        self.phrases = [self.normalize_phrase(phrase) for phrase in phrases]
        self.lead_seconds = lead_seconds
        self.trail_seconds = trail_seconds
        self.min_match_ratio = min_match_ratio
        self.found_matches: Set[str] = set()  # Track unique matches

    def normalize_phrase(self, text: str) -> str:
        """Normalize text for consistent matching"""
        # Remove music notation, punctuation, and non-word characters
        cleaned = re.sub(r'♪|[^\w\s]', '', text)
        # Convert to lowercase and normalize whitespace
        cleaned = ' '.join(cleaned.lower().split())
        return cleaned

    def get_window_text(self, captions: List[Dict], start_idx: int, window_size: int) -> Optional[Dict]:
        """Get text window across multiple captions"""
        if start_idx >= len(captions):
            return None
            
        window_text = []
        start_time = float(captions[start_idx]['start'])
        end_idx = start_idx
        
        for i in range(start_idx, min(len(captions), start_idx + window_size)):
            window_text.append(self.normalize_phrase(captions[i]['text']))
            end_idx = i
            
        # Get end time from next caption or estimate
        if end_idx + 1 < len(captions):
            end_time = float(captions[end_idx + 1]['start'])
        else:
            last_caption = captions[end_idx]
            end_time = float(last_caption['start']) + len(last_caption['text']) / 10
            
        return {
            'text': ' '.join(window_text),
            'start_time': start_time,
            'end_time': end_time,
            'start_idx': start_idx,
            'end_idx': end_idx
        }

    def find_phrase_in_window(self, phrase: str, window_info: Dict) -> Optional[Dict]:
        """Find phrase match within a text window"""
        window_text = window_info['text']
        
        # Try exact match first
        if phrase in window_text:
            return {
                'phrase': phrase,
                'text': phrase,
                'similarity': 1.0,
                'start_time': window_info['start_time'],
                'end_time': window_info['end_time']
            }
            
        # Try fuzzy matching
        words = window_text.split()
        phrase_words = phrase.split()
        
        for i in range(len(words) - len(phrase_words) + 1):
            candidate = ' '.join(words[i:i + len(phrase_words)])
            similarity = SequenceMatcher(None, phrase, candidate).ratio()
            
            if similarity >= self.min_match_ratio:
                return {
                    'phrase': phrase,
                    'text': candidate,
                    'similarity': similarity,
                    'start_time': window_info['start_time'],
                    'end_time': window_info['end_time']
                }
                
        return None

    def find_matches(self, captions: List[Dict]) -> List[Dict]:
        """Find matches including across caption boundaries"""
        matches = []
        self.found_matches.clear()  # Reset found matches
        
        # Try different window sizes for cross-caption matching
        max_window_size = 3  # Maximum captions to combine
        
        for start_idx in range(len(captions)):
            for window_size in range(1, max_window_size + 1):
                window_info = self.get_window_text(captions, start_idx, window_size)
                if not window_info:
                    continue
                    
                for phrase in self.phrases:
                    # Skip if we already found this phrase
                    if phrase in self.found_matches:
                        continue
                        
                    match = self.find_phrase_in_window(phrase, window_info)
                    if match:
                        # Add padding to timestamps
                        match['start_time'] = max(0, match['start_time'] - self.lead_seconds)
                        match['end_time'] = match['end_time'] + self.trail_seconds
                        
                        matches.append(match)
                        self.found_matches.add(phrase)  # Track that we found this phrase
                        logging.debug(f"Found match for '{phrase}' with similarity {match['similarity']:.2f}")
                        break  # Stop looking for this phrase
        
        # Sort matches by timestamp
        matches.sort(key=lambda x: x['start_time'])
        
        return matches

    def process_captions(self, captions: List[Dict]) -> List[Dict]:
        """Main processing method to find matches in captions"""
        # Log normalized transcript for debugging
        transcript = ' '.join(caption['text'] for caption in captions)
        logging.info("Normalized transcript:")
        logging.info(self.normalize_phrase(transcript))
        
        # Find matches
        return self.find_matches(captions)

def download_audio(video_id: str, output_dir: str = 'temp') -> str:
    """Download audio from a YouTube video using yt-dlp"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_id}.mp3")
        
        # Check if file already exists AND is valid
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logging.info(f"Using cached audio for video {video_id}")
            return output_path
            
        logging.info(f"Downloading audio for video {video_id}")
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
            # Download and verify file exists
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logging.info(f"Successfully downloaded audio for video {video_id}")
                return output_path
            else:
                logging.error(f"Failed to download audio for video {video_id}")
                return None
                
    except Exception as e:
        logging.error(f"Error downloading audio for video {video_id}: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)  # Clean up partial download
        return None

def process_videos(video_ids: List[str], phrases: List[str], output_dir: str, youtube_api_key: str) -> List[Dict]:
    """Process videos to find and extract matching phrases"""
    os.makedirs('temp', exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    extractor = PhraseExtractor(phrases)
    best_matches = {}
    processed_phrases = set()
    
    for video_id in video_ids:
        audio_path = None
        try:
            if len(processed_phrases) == len(phrases):
                break
                
            logging.info(f"Processing video {video_id}")
            
            # Get captions for this video
            captions = get_captions(video_id)
            if not captions:
                logging.warning(f"No captions found for video {video_id}. Skipping...")
                continue
            
            # Only search for phrases we haven't found yet
            unmatched_phrases = [p for p in phrases if str(p) not in processed_phrases]
            matches, unmatched_phrases = extractor.find_matches(captions, unmatched_phrases)
            
            if not matches:
                logging.info(f"No matches found for phrases in video {video_id}")
                continue
            
            # Check video duration using last caption timestamp
            if captions:
                last_caption = captions[-1]
                video_duration = float(last_caption.get('start', 0))
                if video_duration > 600:  # 600 seconds = 10 minutes
                    logging.warning(f"Video {video_id} is longer than 10 minutes. Skipping...")
                    continue
            
            # Download audio only if we found matches
            audio_path = download_audio(video_id, 'temp')
            if not audio_path:
                logging.warning(f"Failed to download audio for video {video_id}")
                continue
                
            # Process each match
            for match in matches:
                phrase = match['phrase']
                similarity = match['similarity']
                
                # Extract the clip
                try:
                    audio = AudioSegment.from_file(audio_path)
                    start_ms = int(match['start_time'] * 1000)
                    end_ms = int(match['end_time'] * 1000)
                    clip = audio[start_ms:end_ms]
                    
                    # Generate clip filename
                    safe_phrase = "".join(x for x in phrase if x.isalnum() or x.isspace())
                    clip_path = os.path.join(output_dir, f"clip_{video_id}_{safe_phrase[:30]}.mp3")
                    
                    # Export clip
                    clip.export(clip_path, format="mp3")
                    match['clip_path'] = clip_path
                    
                    if (phrase not in best_matches or 
                        similarity > best_matches[phrase]['similarity']):
                        
                        if phrase in best_matches and 'clip_path' in best_matches[phrase]:
                            old_clip = best_matches[phrase]['clip_path']
                            if old_clip and os.path.exists(old_clip):
                                os.remove(old_clip)
                        
                        match['video_id'] = video_id
                        best_matches[phrase] = match
                        processed_phrases.add(str(phrase))
                        logging.info(f"Saved clip for '{phrase}' with similarity {similarity:.2f} from video {video_id}")
                    
                except Exception as e:
                    logging.error(f"Error saving clip for phrase '{phrase}' from video {video_id}: {str(e)}")
                    continue
                    
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
    
    results = list(best_matches.values())
    logging.info(f"Found best matches for {len(results)} phrases")
    return results

def get_sample_captions() -> List[Dict]:
    """Generate sample captions that match the actual structure"""
    return [
        {
            "text": "♪ Music playing ♪",
            "start": "0.0"
        },
        {
            "text": "Sometimes I wish I could fly home",
            "start": "3.5"
        },
        {
            "text": "Through the clouds and the rain",
            "start": "6.2"
        },
        {
            "text": "When I look back now I remember it all",
            "start": "10.5"
        },
        {
            "text": "Every moment, every fall",
            "start": "13.8"
        },
        {
            "text": "Then I got a letter from the government",
            "start": "18.2"
        },
        {
            "text": "The other day, I opened and read it",
            "start": "21.5"
        },
        {
            "text": "Now I dwell in my cell",
            "start": "25.8"
        },
        {
            "text": "Thinking about all that has happened",
            "start": "28.4"
        }
    ]

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize test phrases
    search_phrases = [
        "I could fly home",
        "I remember it all",
        "letter from the government",
        "I dwell in my cell",
        "And the rain when I look back now"
    ]

    # Create PhraseExtractor instance
    extractor = PhraseExtractor(
        phrases=search_phrases,
        lead_seconds=0.5,
        trail_seconds=1.0,
        min_match_ratio=0.8  # Can be adjusted if needed
    )

    # Get sample captions and process them
    captions = get_sample_captions()
    matches = extractor.process_captions(captions)

    # Print results
    print("\nMatches found:")
    print("-" * 50)
    for match in matches:
        print(f"\nOriginal phrase: '{match['phrase']}'")
        print(f"Matched text: '{match['text']}'")
        print(f"Similarity: {match['similarity']:.2f}")
        print(f"Start time: {match['start_time']:.2f}s")
        print(f"End time: {match['end_time']:.2f}s")
        print("-" * 30)

    # Print summary
    print(f"\nFound {len(matches)} matches out of {len(search_phrases)} search phrases")
    
    # Print unmatched phrases
    matched_phrases = {match['phrase'] for match in matches}
    unmatched_phrases = set(extractor.phrases) - matched_phrases
    if unmatched_phrases:
        print("\nUnmatched phrases:")
        for phrase in unmatched_phrases:
            print(f"- {phrase}")

if __name__ == "__main__":
    main()