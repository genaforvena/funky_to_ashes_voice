import numpy as np
from pydub import AudioSegment
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import wave
import contextlib
from datetime import timedelta
import groq
import base64

@dataclass
class WordSegment:
    word: str
    start_time: float
    end_time: float
    confidence: float
    chunk_index: int  # New field to track which chunk this word came from

@dataclass
class TranscriptionResult:
    word_segments: List[WordSegment] = field(default_factory=list)
    word_occurrences: Dict[str, List[int]] = field(default_factory=lambda: {})

class LLMWordSampler:
    def __init__(self, 
                 audio_path: str, 
                 lyrics_path: str, 
                 output_dir: str,
                 chunk_size: float = 5.0):  # Process 5 seconds at a time
        self.audio_path = audio_path
        self.lyrics_path = lyrics_path
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.groq_client = groq.Groq(api_key=self.groq_api_key)
        
        os.makedirs(output_dir, exist_ok=True)

    def load_lyrics(self) -> List[str]:
        with open(self.lyrics_path, 'r') as f:
            lyrics = f.read().lower()
            words = [word.strip('.,!?-"\'()[]{}') for word in lyrics.split()]
            return [w for w in words if w]

    def get_audio_duration(self) -> float:
        """Get duration of audio file in seconds."""
        with contextlib.closing(wave.open(self.audio_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration

    def transcribe_with_timestamps(self) -> TranscriptionResult:
        print("Transcribing audio with timestamps...")
        
        audio = AudioSegment.from_file(self.audio_path)
        total_duration = len(audio) / 1000  # Duration in seconds
        
        result = TranscriptionResult()
        
        for chunk_index, start_time in enumerate(np.arange(0, total_duration, self.chunk_size)):
            end_time = min(start_time + self.chunk_size, total_duration)
            chunk = audio[start_time*1000:end_time*1000]
            
            # Export chunk to a temporary file
            temp_file = "temp_chunk.wav"
            chunk.export(temp_file, format="wav")
            
            # Read the audio chunk and encode it to base64
            with open(temp_file, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
            
            # Delete the temporary file
            os.remove(temp_file)
            
            # Make the API call to Groq
            response = self.groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_base64,
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )
            
            # Extract word segments from the response
            for segment in response.words:
                word = segment.word.lower().strip()
                word_segment = WordSegment(
                    word=word,
                    start_time=segment.start + start_time,
                    end_time=segment.end + start_time,
                    confidence=getattr(segment, 'confidence', 1.0),  # Default to 1.0 if not provided
                    chunk_index=chunk_index
                )
                result.word_segments.append(word_segment)
                
                # Track word occurrences
                if word not in result.word_occurrences:
                    result.word_occurrences[word] = []
                result.word_occurrences[word].append(chunk_index)
        
        return result

    def match_lyrics_to_segments(self, 
                               lyrics: List[str], 
                               segments: List[WordSegment]) -> Dict[str, List[WordSegment]]:
        """Match lyrics words to transcribed segments using fuzzy matching."""
        from difflib import SequenceMatcher
        
        word_matches = {}
        
        for lyric_word in lyrics:
            word_matches[lyric_word] = []
            
            for segment in segments:
                # Calculate similarity between lyric word and transcribed word
                similarity = SequenceMatcher(None, 
                                          lyric_word.lower(), 
                                          segment.word.lower()).ratio()
                
                if similarity > 0.8:  # Adjust threshold as needed
                    word_matches[lyric_word].append(segment)
        
        return word_matches

    def extract_samples(self, word_matches: Dict[str, List[WordSegment]]):
        print("Extracting word samples...")
        audio = AudioSegment.from_mp3(self.audio_path)
        
        # Parameters for extraction
        pad_start = 50  # ms padding before word
        min_length = 200  # ms minimum sample length
        
        metadata = {}
        
        for word, segments in word_matches.items():
            if not segments:
                print(f"Warning: No matches found for word '{word}'")
                continue
                
            # Use the segment with highest confidence
            best_segment = max(segments, key=lambda x: x.confidence)
            
            # Convert times to milliseconds
            start_ms = max(0, int(best_segment.start_time * 1000) - pad_start)
            end_ms = int(best_segment.end_time * 1000)
            
            # Ensure minimum length
            if end_ms - start_ms < min_length:
                end_ms = start_ms + min_length
            
            # Extract audio segment
            word_audio = audio[start_ms:end_ms]
            
            # Save sample
            filename = f"{word}.mp3"
            filepath = os.path.join(self.output_dir, filename)
            word_audio.export(filepath, format="mp3")
            
            # Store metadata
            metadata[word] = {
                'filename': filename,
                'start_time': best_segment.start_time,
                'end_time': best_segment.end_time,
                'confidence': best_segment.confidence,
                'duration': (end_ms - start_ms) / 1000.0
            }
        
        # Save metadata
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    def process(self):
        print("Loading lyrics...")
        lyrics = self.load_lyrics()
        
        print("Starting transcription process...")
        transcription_result = self.transcribe_with_timestamps()
        
        print(f"Found {len(transcription_result.word_segments)} word segments")
        print("Matching lyrics to transcribed segments...")
        word_matches = self.match_lyrics_to_segments(lyrics, transcription_result.word_segments)
        
        print("Extracting matched samples...")
        self.extract_samples(word_matches)
        
        print(f"Done! Word samples saved to {self.output_dir}")
        
        # Print summary
        total_matches = sum(len(matches) for matches in word_matches.values())
        print(f"\nSummary:")
        print(f"Total lyrics words: {len(lyrics)}")
        print(f"Total matched segments: {total_matches}")
        print(f"Average matches per word: {total_matches/len(lyrics):.1f}")
        
        # Print word occurrence information
        print("\nWord occurrences by chunk:")
        for word, chunks in transcription_result.word_occurrences.items():
            print(f"'{word}': recognized in chunks {chunks}")

# Example usage
if __name__ == "__main__":
    sampler = LLMWordSampler(
        audio_path="song.mp3",
        lyrics_path="lyrics.txt",
        output_dir="word_samples"
    )
    sampler.process()
