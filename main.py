import numpy as np
from pydub import AudioSegment
import json
import os
from typing import List, Dict, Tuple
import librosa
from scipy.spatial.distance import cosine
from scipy.interpolate import interp1d

class LLMWordSampler:
    def __init__(self, audio_path: str, lyrics_path: str, output_dir: str, anchor_info: Dict[str, Tuple[float, float]]):
        self.audio_path = audio_path
        self.lyrics_path = lyrics_path
        self.output_dir = output_dir
        self.anchor_info = anchor_info
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
        audio_duration = len(audio) / 1000.0  # Duration in seconds

        for anchor_phrase, (start_time, end_time) in self.anchor_info.items():
            anchor_words = anchor_phrase.lower().split()
            anchor_index = words.index(anchor_words[0])
            
            # Calculate time scaling factor
            time_range = end_time - start_time
            word_count = len(anchor_words)
            time_per_word = time_range / word_count

            for i, word in enumerate(words[anchor_index:]):
                word_start = start_time + i * time_per_word
                word_end = word_start + time_per_word

                if word_end > audio_duration:
                    break

                start_ms = int(word_start * 1000)
                end_ms = int(word_end * 1000)

                # Extract audio segment
                word_audio = audio[start_ms:end_ms]

                # Save sample
                if word not in metadata:
                    metadata[word] = []
                
                filename = f"{word}_{len(metadata[word])}.wav"
                filepath = os.path.join(self.output_dir, filename)
                word_audio.export(filepath, format="wav")

                # Store metadata
                metadata[word].append({
                    'filename': filename,
                    'start_time': word_start,
                    'end_time': word_end,
                    'duration': word_end - word_start
                })

        # Save metadata
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def compare_spectrograms(self, samples: List[Dict]) -> str:
        spectrograms = []
        max_length = 0
        for sample in samples:
            audio, sr = librosa.load(os.path.join(self.output_dir, sample['filename']))
            n_fft = min(2048, len(audio))
            hop_length = n_fft // 4
            spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=n_fft, hop_length=hop_length)
            spectrograms.append(spectrogram)
            max_length = max(max_length, spectrogram.shape[1])

        # Pad or truncate spectrograms to the same length
        target_length = 100  # You can adjust this value
        padded_spectrograms = []
        for spectrogram in spectrograms:
            if spectrogram.shape[1] < target_length:
                padded = np.pad(spectrogram, ((0, 0), (0, target_length - spectrogram.shape[1])), mode='constant')
            else:
                padded = spectrogram[:, :target_length]
            padded_spectrograms.append(padded.flatten())

        best_sample = samples[0]['filename']
        if len(padded_spectrograms) > 1:
            similarities = []
            for i, spec1 in enumerate(padded_spectrograms):
                sim = sum(1 - cosine(spec1, spec2) for j, spec2 in enumerate(padded_spectrograms) if i != j)
                similarities.append(sim)
            best_index = np.argmax(similarities)
            best_sample = samples[best_index]['filename']

        return best_sample

    def select_best_samples(self, metadata: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        best_samples = {}
        for word, samples in metadata.items():
            if len(samples) > 1:
                # Calculate the average amplitude for each sample
                amplitudes = []
                for sample in samples:
                    audio = AudioSegment.from_wav(os.path.join(self.output_dir, sample['filename']))
                    amplitudes.append(audio.rms)
                
                # Select the sample with the median amplitude
                median_index = len(amplitudes) // 2
                sorted_indices = sorted(range(len(amplitudes)), key=lambda k: amplitudes[k])
                best_sample = samples[sorted_indices[median_index]]
            else:
                best_sample = samples[0]
            
            best_samples[word] = best_sample
        return best_samples

    def assemble_audio(self, lyrics: List[str], best_samples: Dict[str, Dict]):
        print("Assembling audio from best samples...")
        assembled_audio = AudioSegment.silent(duration=0)
        
        for word in lyrics:
            if word in best_samples:
                word_audio = AudioSegment.from_wav(os.path.join(self.output_dir, best_samples[word]['filename']))
                assembled_audio += word_audio
            else:
                # If word not found, add a short silence
                assembled_audio += AudioSegment.silent(duration=100)  # 100ms silence for missing words
        
        # Export the assembled audio
        output_path = os.path.join(self.output_dir, "assembled_audio.wav")
        assembled_audio.export(output_path, format="wav")
        print(f"Assembled audio saved to: {output_path}")

    def process(self):
        print("Loading lyrics...")
        words = self.load_lyrics()

        print("Loading audio...")
        audio = AudioSegment.from_file(self.audio_path)

        print("Extracting word samples based on anchor information...")
        metadata = self.extract_samples(words, [], audio)

        print("Selecting best samples...")
        best_samples = self.select_best_samples(metadata)

        print(f"Done! Word samples saved to {self.output_dir}")

        # Print summary
        print(f"\nSummary:")
        print(f"Total lyrics words: {len(words)}")
        print(f"Unique words with samples: {len(best_samples)}")

        # Assemble audio from best samples
        self.assemble_audio(words, best_samples)

# Example usage
if __name__ == "__main__":
    anchor_info = {
        "I switched the time zone": (25.0, 30.0)
    }
    sampler = LLMWordSampler(
        audio_path="1.mp3",
        lyrics_path="1.txt",
        output_dir="word_samples",
        anchor_info=anchor_info
    )
    sampler.process()
