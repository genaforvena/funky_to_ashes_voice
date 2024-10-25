import numpy as np
from pydub import AudioSegment
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional
import wave
import contextlib
from transformers import pipeline
import torch
from datetime import timedelta

@dataclass
class WordSegment:
    word: str
    start_time: float
    end_time: float
    confidence: float

class LLMWordSampler:
    def __init__(self, 
                 audio_path: str, 
                 lyrics_path: str, 
                 output_dir: str,
                 chunk_size: float = 30.0):  # Process 30 seconds at a time
        self.audio_path = audio_path
        self.lyrics_path = lyrics_path
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize the ASR pipeline
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=self.device,
            chunk_length_s=self.chunk_size,
            return_timestamps=True  # Important: get word-level timestamps
        )
        
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

    def transcribe_with_timestamps(self) -> List[WordSegment]:
        print("Transcribing audio with timestamps...")
        
        # Run ASR with word timestamps
        result = self.transcriber(self.audio_path)
        
        # Extract word segments from the chunks
        word_segments = []
        
        # Whisper returns chunks with word-level timestamps
        for chunk in result["chunks"]:
            for word_data in chunk["words"]:
                word_segments.append(WordSegment(
                    word=word_data["text"].lower().strip(),
                    start_time=word_data["start"],
                    end_time=word_data["end"],
                    confidence=word_data["confidence"]
                ))
        
        return word_segments

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
        segments = self.transcribe_with_timestamps()
        
        print(f"Found {len(segments)} word segments")
        print("Matching lyrics to transcribed segments...")
        word_matches = self.match_lyrics_to_segments(lyrics, segments)
        
        print("Extracting matched samples...")
        self.extract_samples(word_matches)
        
        print(f"Done! Word samples saved to {self.output_dir}")
        
        # Print summary
        total_matches = sum(len(matches) for matches in word_matches.values())
        print(f"\nSummary:")
        print(f"Total lyrics words: {len(lyrics)}")
        print(f"Total matched segments: {total_matches}")
        print(f"Average matches per word: {total_matches/len(lyrics):.1f}")

# Example usage
if __name__ == "__main__":
    sampler = LLMWordSampler(
        audio_path="song.mp3",
        lyrics_path="lyrics.txt",
        output_dir="word_samples"
    )
    sampler.process()