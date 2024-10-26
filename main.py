import numpy as np
from pydub import AudioSegment
import json
import os
from typing import List, Dict, Tuple
import librosa
from scipy.spatial.distance import cosine

class LLMWordSampler:
    def __init__(self, audio_path: str, lyrics_path: str, output_dir: str):
        self.audio_path = audio_path
        self.lyrics_path = lyrics_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_lyrics(self) -> List[str]:
        with open(self.lyrics_path, 'r') as f:
            lyrics = f.read().lower()
            words = [word.strip('.,!?-"\'()[]{}') for word in lyrics.split()]
            return [w for w in words if w]

    def detect_voice_segments(self, audio_array: np.ndarray, sample_rate: int) -> List[Tuple[int, int]]:
        frame_length = int(sample_rate * 0.025)  # 25ms frames
        hop_length = frame_length // 2  # 50% overlap
        energy = librosa.feature.rms(y=audio_array, frame_length=frame_length, hop_length=hop_length)[0]
        threshold = np.mean(energy) * 1.5
        voice_activity = energy > threshold

        segments = []
        start = None
        for i, active in enumerate(voice_activity):
            if active and start is None:
                start = i * hop_length
            elif not active and start is not None:
                end = i * hop_length
                segments.append((start, end))
                start = None

        if start is not None:
            segments.append((start, len(audio_array)))

        return segments

    def extract_samples(self, words: List[str], segments: List[Tuple[int, int]], audio: AudioSegment):
        metadata = {}
        for i, (start, end) in enumerate(segments):
            if i >= len(words):
                break
            word = words[i]
            
            # Convert sample indices to milliseconds
            start_ms = start * 1000 // audio.frame_rate
            end_ms = end * 1000 // audio.frame_rate

            # Extract audio segment
            word_audio = audio[start_ms:end_ms]

            # Save sample
            if word not in metadata:
                metadata[word] = []
            
            filename = f"{word}_{len(metadata[word])}.mp3"
            filepath = os.path.join(self.output_dir, filename)
            word_audio.export(filepath, format="mp3")

            # Store metadata
            metadata[word].append({
                'filename': filename,
                'start_time': start_ms / 1000,
                'end_time': end_ms / 1000,
                'duration': (end_ms - start_ms) / 1000
            })

        # Save metadata
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def compare_spectrograms(self, samples: List[Dict]) -> str:
        spectrograms = []
        for sample in samples:
            audio, sr = librosa.load(os.path.join(self.output_dir, sample['filename']))
            spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
            spectrograms.append(spectrogram.flatten())

        best_sample = samples[0]['filename']
        if len(spectrograms) > 1:
            similarities = []
            for i, spec1 in enumerate(spectrograms):
                sim = sum(1 - cosine(spec1, spec2) for j, spec2 in enumerate(spectrograms) if i != j)
                similarities.append(sim)
            best_index = np.argmax(similarities)
            best_sample = samples[best_index]['filename']

        return best_sample

    def select_best_samples(self, metadata: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        best_samples = {}
        for word, samples in metadata.items():
            if len(samples) > 1:
                best_filename = self.compare_spectrograms(samples)
                best_sample = next(sample for sample in samples if sample['filename'] == best_filename)
            else:
                best_sample = samples[0]
            best_samples[word] = best_sample
        return best_samples

    def assemble_audio(self, lyrics: List[str], metadata: Dict[str, Dict]):
        print("Assembling audio from samples...")
        assembled_audio = AudioSegment.silent(duration=0)
        
        for word in lyrics:
            if word in metadata:
                word_audio = AudioSegment.from_mp3(os.path.join(self.output_dir, metadata[word]['filename']))
                assembled_audio += word_audio
            else:
                # If word not found, add a short silence
                assembled_audio += AudioSegment.silent(duration=200)  # 200ms silence
        
        # Export the assembled audio
        output_path = os.path.join(self.output_dir, "assembled_audio.mp3")
        assembled_audio.export(output_path, format="mp3")
        print(f"Assembled audio saved to: {output_path}")

    def process(self):
        print("Loading lyrics...")
        words = self.load_lyrics()

        print("Loading audio and detecting voice segments...")
        audio_array, sample_rate = librosa.load(self.audio_path)
        segments = self.detect_voice_segments(audio_array, sample_rate)

        print(f"Found {len(segments)} voice segments")
        if len(segments) == 0:
            print("No voice segments found. Exiting process.")
            return

        print("Extracting word samples...")
        audio = AudioSegment.from_file(self.audio_path)
        metadata = self.extract_samples(words, segments, audio)

        print("Selecting best samples...")
        best_samples = self.select_best_samples(metadata)

        print(f"Done! Word samples saved to {self.output_dir}")

        # Print summary
        print(f"\nSummary:")
        print(f"Total lyrics words: {len(words)}")
        print(f"Total voice segments: {len(segments)}")
        print(f"Unique words with samples: {len(best_samples)}")

        # Assemble audio from best samples
        self.assemble_audio(words, best_samples)

# Example usage
if __name__ == "__main__":
    sampler = LLMWordSampler(
        audio_path="1.mp3",
        lyrics_path="1.txt",
        output_dir="word_samples"
    )
    sampler.process()
