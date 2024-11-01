from pydub import AudioSegment
import os
from typing import List, Dict
import logging
import re

class AudioClipper:
    def __init__(self, output_dir: str = 'output'):
        """
        Initialize the audio clipper
        
        Args:
            output_dir: Directory where clips will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def sanitize_filename(self, phrase: str) -> str:
        """
        Convert phrase to safe filename
        
        Args:
            phrase: Phrase to convert to filename
            
        Returns:
            Safe filename string
        """
        # Remove non-alphanumeric characters and convert spaces to underscores
        safe_name = re.sub(r'[^a-zA-Z0-9\s]', '', phrase)
        safe_name = safe_name.strip().replace(' ', '_').lower()
        return safe_name[:50]  # Limit length to avoid too long filenames

    def create_clip(self, audio_path: str, match: Dict) -> str:
        """
        Create audio clip from match information
        
        Args:
            audio_path: Path to source audio file
            match: Dictionary containing match information
            
        Returns:
            Path to the saved clip file
        """
        try:
            # Load the audio file
            audio = AudioSegment.from_file(audio_path)
            
            # Convert times to milliseconds
            start_ms = int(match['start_time'] * 1000)
            end_ms = int(match['end_time'] * 1000)
            
            # Extract the clip
            clip = audio[start_ms:end_ms]
            
            # Generate output filename
            safe_name = self.sanitize_filename(match['phrase'])
            output_path = os.path.join(self.output_dir, f"{safe_name}.mp3")
            
            # Export clip with good quality settings
            clip.export(
                output_path,
                format="mp3",
                parameters=[
                    "-q:a", "0",  # Use highest quality
                    "-filter:a", "loudnorm"  # Normalize audio levels
                ]
            )
            
            logging.info(f"Created clip: {output_path} ({(end_ms - start_ms)/1000:.2f}s)")
            return output_path
            
        except Exception as e:
            logging.error(f"Error creating clip for '{match['phrase']}': {str(e)}")
            return ""

def process_matches(matches: List[Dict], source_audio: str, output_dir: str = 'output') -> List[Dict]:
    """
    Process all matches and create clips
    
    Args:
        matches: List of match dictionaries
        source_audio: Path to source audio file
        output_dir: Directory to save clips
        
    Returns:
        List of matches with added clip paths
    """
    clipper = AudioClipper(output_dir)
    processed_matches = []
    
    for match in matches:
        try:
            clip_path = clipper.create_clip(source_audio, match)
            if clip_path:
                match['clip_path'] = clip_path
                processed_matches.append(match)
                
        except Exception as e:
            logging.error(f"Error processing match '{match['phrase']}': {str(e)}")
            continue
            
    return processed_matches

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data
    matches = [
        {
            'phrase': 'I could fly home',
            'start_time': 3.0,
            'end_time': 7.2,
            'similarity': 1.0
        },
        {
            'phrase': 'I remember it all',
            'start_time': 10.0,
            'end_time': 14.8,
            'similarity': 1.0
        }
    ]
    
    source_audio = "1.mp3"
    processed = process_matches(matches, source_audio, 'output')
    
    # Print results
    for match in processed:
        print(f"\nProcessed clip for '{match['phrase']}':")
        print(f"Clip path: {match['clip_path']}")
        print(f"Duration: {match['end_time'] - match['start_time']:.2f}s")