import json
import hashlib
import os
import sys
import datetime
from groq import Groq
from pydub import AudioSegment
from quotes_extractor import find_longest_phrase_matches
from audio_downloader import search_youtube_video, download_audio, sanitize_filename
from fuzzywuzzy import fuzz

CACHE_DIR = 'cache'

def log_message(message, file=None):
    if file:
        print(f"[{datetime.datetime.now()}] {message}", file=file)
    else:
        print(f"[{datetime.datetime.now()}] {message}")

def get_file_checksum(file_path):
    """Computes the MD5 checksum of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()

def split_audio(audio_path, chunk_length_ms=30000):
    """
    Splits an audio file into chunks of a specified length.
    """
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunks.append(audio[i:i+chunk_length_ms])
    return chunks

def transcribe_audio_with_word_timestamps(audio_path):
    """
    Transcribes an audio file using Groq API and returns word-level timestamps.
    Caches the transcription to avoid re-processing.
    """
    client = Groq()

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    checksum = get_file_checksum(audio_path)
    cache_filename = os.path.join(CACHE_DIR, f"{checksum}_transcription.json")

    if os.path.exists(cache_filename):
        log_message(f"Loading cached transcription for '{audio_path}'")
        try:
            with open(cache_filename, 'r', encoding='utf-8') as cache_file:
                return json.load(cache_file)
        except json.JSONDecodeError:
            log_message(f"Corrupted cache file found for '{audio_path}'. Deleting and re-transcribing.")
            os.remove(cache_filename)

    log_message(f"Splitting audio file: {audio_path}")
    chunks = split_audio(audio_path)
    log_message(f"Split audio file into {len(chunks)} chunks.")

    transcription_words = []
    for i, chunk in enumerate(chunks):
        chunk_path = f"chunk_{i}.mp3"
        chunk.export(chunk_path, format="mp3")
        log_message(f"Transcribing chunk {i+1}/{len(chunks)}")
        with open(chunk_path, "rb") as file:
            try:
                transcription = client.audio.transcriptions.create(
                    file=(chunk_path, file.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                )
            except Exception as e:
                log_message(f"An error occurred during transcription of chunk {i+1}: {e}", file=sys.stderr)
                os.remove(chunk_path)
                continue
        log_message(f"Finished transcribing chunk {i+1}/{len(chunks)}")
        os.remove(chunk_path)

        if hasattr(transcription, 'words') and transcription.words:
            for word in transcription.words:
                transcription_words.append({
                    'word': word.word.strip().lower(),
                    'start': word.start + (i * 30),
                    'end': word.end + (i * 30)
                })
        else:
            log_message("Word-level timestamps not available for this chunk, falling back to segment-level.")
            for segment in transcription.segments:
                transcription_words.append({
                    'word': segment['text'].strip().lower(),
                    'start': segment['start'] + (i * 30),
                    'end': segment['end'] + (i * 30)
                })

    with open(cache_filename, 'w', encoding='utf-8') as cache_file:
        json.dump(transcription_words, cache_file, ensure_ascii=False, indent=2)

    return transcription_words

def find_matching_word_segments(input_words, transcription_words):
    """
    Finds segments in the transcription that match the input words.
    """
    log_message(f"Searching for: '{' '.join(input_words)}'")
    log_message(f"In transcription: {[word['word'] for word in transcription_words]}")

    matched_segments = []
    for i in range(len(transcription_words) - len(input_words) + 1):
        # Check for a sequential match of whole words
        if all(transcription_words[i + j]['word'] == input_words[j] for j in range(len(input_words))):
            start_time = transcription_words[i]['start'] * 1000
            end_time = transcription_words[i + len(input_words) - 1]['end'] * 1000
            matched_segments.append({'start': start_time, 'end': end_time})
            # Move index past the matched phrase
            i += len(input_words) -1

    if matched_segments:
        log_message(f"Found {len(matched_segments)} matches.")
    else:
        log_message("No matches found.")

    return matched_segments

def extract_audio_segments(audio_path, segments):
    """
    Extracts audio segments from a file based on start and end times.
    """
    audio = AudioSegment.from_file(audio_path)
    output_audio = AudioSegment.silent(duration=0)
    for segment in segments:
        start_ms = segment['start']
        end_ms = segment['end']
        output_audio += audio[start_ms:end_ms]
    return output_audio

def generate_audio_from_input(input_text):
    """
    Main function to generate audio from an input text string.
    """
    # 1. Find song matches for the input text
    log_message("Finding song matches...")
    matches = find_longest_phrase_matches(input_text)
    if not matches:
        log_message("No song matches found.")
        return
    log_message(f"Found {len(matches)} potential song matches.")

    # 2. Process each matched song
    final_audio = AudioSegment.silent(duration=0)
    processed_songs = set()

    for match in matches:
        title = match['title']
        artist = match['artist']
        phrase = match['phrase']

        song_key = (title, artist)
        if song_key in processed_songs:
            continue
        processed_songs.add(song_key)

        log_message(f"Processing: {title} by {artist} for phrase: '{phrase}'")

        # Process only the first match
        if len(final_audio) > 0:
            break

        # 3. Download audio from YouTube
        log_message(f"Searching for YouTube video for '{title}' by '{artist}'")
        youtube_url = search_youtube_video(title, artist)
        if not youtube_url:
            log_message(f"No YouTube video found for {title} by {artist}.")
            continue
        log_message(f"Found YouTube video: {youtube_url}")

        safe_filename = sanitize_filename(f"{title} - {artist}")
        log_message(f"Downloading audio to '{safe_filename}.mp3'")
        audio_file = download_audio(youtube_url, f"{safe_filename}.mp3")
        if not audio_file or not os.path.exists(audio_file):
            log_message(f"Failed to download audio for {title} by {artist}.")
            continue
        log_message(f"Finished downloading audio to '{audio_file}'")

        # 4. Transcribe audio to get word timestamps
        log_message(f"Transcribing audio file: {audio_file}")
        transcription_words = transcribe_audio_with_word_timestamps(audio_file)
        if not transcription_words:
            log_message(f"Transcription failed for {audio_file}.")
            continue
        log_message(f"Finished transcribing audio file: {audio_file}")

        # 5. Find matching segments in the transcription
        input_words = phrase.lower().split()
        log_message(f"Finding matching segments for '{phrase}'")
        matching_segments = find_matching_word_segments(input_words, transcription_words)
        if not matching_segments:
            log_message(f"Could not find the exact phrase '{phrase}' in the transcription.")
            continue
        log_message(f"Found {len(matching_segments)} matching segments.")

        # 6. Extract and append audio segments
        log_message("Extracting and appending audio segments...")
        song_audio = extract_audio_segments(audio_file, matching_segments)
        final_audio += song_audio
        log_message("Finished extracting and appending audio segments.")

    # 7. Export the final combined audio
    if len(final_audio) > 0:
        output_filename = "final_output.mp3"
        log_message(f"Exporting final audio to '{output_filename}'")
        final_audio.export(output_filename, format="mp3")
        log_message(f"Successfully generated audio: {output_filename}")
    else:
        log_message("Could not generate any audio from the input text.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        generate_audio_from_input(user_input)
    else:
        print("Usage: python main.py <text_to_synthesize>")
