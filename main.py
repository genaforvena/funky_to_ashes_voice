import json
import hashlib
import os
import sys
import subprocess
import argparse
from groq import Groq
from pydub import AudioSegment
from quotes_extractor import find_longest_phrase_matches
from audio_downloader import search_youtube_video, download_audio, sanitize_filename

# Check for required environment variables
GENIUS_TOKEN = os.getenv("GENIUS_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def check_environment():
    """
    Check if all required environment variables are set and FFmpeg is installed.
    Returns True if all checks pass, False otherwise.
    """
    all_checks_passed = True
    
    # Check for GENIUS_TOKEN
    if not GENIUS_TOKEN:
        print("Error: GENIUS_TOKEN environment variable is not set.")
        print("Please set it using:")
        print("  export GENIUS_TOKEN='your_genius_api_key'  # On Linux/macOS")
        print("  set GENIUS_TOKEN=your_genius_api_key  # On Windows")
        all_checks_passed = False
    
    # Check for GROQ_API_KEY
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY environment variable is not set.")
        print("Please set it using:")
        print("  export GROQ_API_KEY='your_groq_api_key'  # On Linux/macOS")
        print("  set GROQ_API_KEY=your_groq_api_key  # On Windows")
        all_checks_passed = False
    
    # Check for FFmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True)
        if result.returncode != 0:
            raise Exception("FFmpeg command returned non-zero exit code")
    except Exception as e:
        print("Error: FFmpeg is not installed or not in PATH.")
        print("Please install FFmpeg:")
        print("  - Windows: Download from https://ffmpeg.org/download.html and add to PATH")
        print("  - macOS: brew install ffmpeg")
        print("  - Linux: sudo apt-get install ffmpeg")
        all_checks_passed = False
    
    return all_checks_passed


def transcribe_audio_groq(audio_path):
    """
    Transcribe audio using the Groq API.
    
    Args:
        audio_path (str): Path to the audio file to transcribe
        
    Returns:
        tuple: (transcription_text, segments) where segments is a list of 
               (text, start_ms, end_ms) tuples
    """
    client = Groq(api_key=GROQ_API_KEY)
    filename = audio_path  # Path to your audio file

    print(f"Transcribing audio file: {filename}")
    try:
        with open(filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
            file=(filename, file.read()),
            model="distil-whisper-large-v3-en",
                response_format="verbose_json",
            )
    except Exception as e:
        print(f"An error occurred during transcription with Groq: {e}", file=sys.stderr)
        return None, []
    
    # Process the transcription result
    transcription_text = transcription.text
    segments = []
    for segment in transcription.segments:
        text = segment['text'].strip().lower()
        start_ms = int(float(segment['start']) * 1000)
        end_ms = int(float(segment['end']) * 1000)
        segments.append((text, start_ms, end_ms))

    return transcription_text, segments


def find_phrases_in_transcription(phrases, timestamps):
    """
    Find phrases in the transcription timestamps.
    
    Args:
        phrases (list): List of phrases to find
        timestamps (list): List of (text, start, end) tuples from transcription
        
    Returns:
        dict: Dictionary mapping phrases to their start and end timestamps
    """
    phrase_segments = {}
    for phrase in phrases:
        phrase_lower = phrase.lower()
        for text, start, end in timestamps:
            if phrase_lower in text:
                if phrase not in phrase_segments or start < phrase_segments[phrase]['start']:
                    phrase_segments[phrase] = {'start': start, 'end': end}
    return phrase_segments


def assemble_audio_segments(audio_path, phrase_segments, phrase_order):
    """
    Assemble audio segments from the original audio based on phrase segments.
    
    Args:
        audio_path (str): Path to the original audio file
        phrase_segments (dict): Dictionary mapping phrases to their start and end timestamps
        phrase_order (list): List of phrases in the order they should appear in the output
        
    Returns:
        AudioSegment: The assembled audio segments
    """
    audio = AudioSegment.from_file(audio_path)
    output_audio = AudioSegment.silent(duration=0)

    for phrase in phrase_order:
        if phrase in phrase_segments:
            segment_info = phrase_segments[phrase]
            segment = audio[segment_info['start']:segment_info['end']]
            output_audio += segment
        else:
            print(f"Phrase '{phrase}' not found in transcription.")

    return output_audio


def tokenize_input_text(input_text):
    """
    Tokenize input text into words.
    
    Args:
        input_text (str): The input text to tokenize
        
    Returns:
        set: Set of tokens (words) from the input text
    """
    import re
    # Remove punctuation and convert to lowercase
    tokens = re.findall(r'\b\w+\b', input_text.lower())
    return set(tokens)


# This function has been moved below with improved caching


def extract_audio_segments_by_words(audio_path, matching_segments):
    """
    Extract audio segments from the original audio based on word-level timestamps.
    
    Args:
        audio_path (str): Path to the original audio file
        matching_segments (list): List of dictionaries with start and end timestamps
        
    Returns:
        AudioSegment: The extracted audio segments
    """
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)
    output_audio = AudioSegment.silent(duration=0)
    crossfade_duration = 50  # milliseconds

    for i, segment in enumerate(matching_segments):
        start_ms = segment['start']
        end_ms = segment['end']
        audio_segment = audio[start_ms:end_ms]

        if i > 0:
            output_audio = output_audio.append(audio_segment, crossfade=crossfade_duration)
        else:
            output_audio += audio_segment

    return output_audio

def find_matching_word_segments(input_phrases, transcription_words):
    """
    Find segments in the transcription that match the input phrases.
    
    Args:
        input_phrases (list): List of phrases to find
        transcription_words (list): List of dictionaries with word, start, and end timestamps
        
    Returns:
        list: List of dictionaries with phrase, start, and end timestamps
    """
    # Concatenate transcription text and keep track of positions
    full_text = ''
    positions = []  # List of tuples (start_time, end_time, text)
    for word_info in transcription_words:
        start_time = int(word_info['start'] * 1000)
        end_time = int(word_info['end'] * 1000)
        text = word_info['word']
        full_text += text + ' '
        positions.append({
            'start': start_time,
            'end': end_time,
            'text': text
        })
    full_text = full_text.strip().lower()
    
    matching_segments = []
    for phrase in input_phrases:
        phrase_lower = phrase.lower()
        index = full_text.find(phrase_lower)
        if index != -1:
            # Find corresponding timestamps
            accumulated_length = 0
            segment_start = None
            segment_end = None
            for pos in positions:
                text = pos['text'].lower()
                text_length = len(text) + 1  # +1 for the space added during concatenation
                if accumulated_length <= index < accumulated_length + text_length:
                    segment_start = pos['start']
                if accumulated_length < index + len(phrase_lower) <= accumulated_length + text_length:
                    segment_end = pos['end']
                    break
                accumulated_length += text_length
            if segment_start is not None and segment_end is not None:
                matching_segments.append({
                    'phrase': phrase,
                    'start': segment_start,
                    'end': segment_end
                })
    return matching_segments


CACHE_DIR = 'cache'

def get_file_checksum(file_path):
    """
    Calculate the MD5 checksum of a file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: MD5 checksum of the file
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()

def transcribe_audio_with_word_timestamps(audio_path):
    """
    Transcribe audio with word-level timestamps using the Groq API.
    Uses caching to avoid re-transcribing the same audio file.
    
    Args:
        audio_path (str): Path to the audio file to transcribe
        
    Returns:
        list: List of dictionaries with word, start, and end timestamps
    """
    client = Groq(api_key=GROQ_API_KEY)

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # Use the file checksum for the cache filename
    checksum = get_file_checksum(audio_path)
    cache_filename = os.path.join(CACHE_DIR, f"{checksum}_transcription.json")

    # Check if the cache file exists
    if os.path.exists(cache_filename):
        print(f"Loading cached transcription for '{audio_path}'")
        with open(cache_filename, 'r', encoding='utf-8') as cache_file:
            transcription_words = json.load(cache_file)
        return transcription_words

    print(f"Transcribing audio file: {audio_path}")
    with open(audio_path, "rb") as file:
        try:
            transcription = client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
        except Exception as e:
            print(f"An error occurred during transcription with Groq: {e}", file=sys.stderr)
            return []

    # Process the transcription result
    transcription_words = []
    for segment in transcription.segments:
        transcription_words.append({
            'word': segment['text'].strip().lower(),
            'start': float(segment['start']),
            'end': float(segment['end'])
        })

    # Save the transcription to the cache file
    with open(cache_filename, 'w', encoding='utf-8') as cache_file:
        json.dump(transcription_words, cache_file, ensure_ascii=False, indent=2)

    return transcription_words

def extract_audio_segments_by_phrases(audio_path, matching_segments):
    """
    Extract audio segments from the original audio based on phrase-level timestamps.
    
    Args:
        audio_path (str): Path to the original audio file
        matching_segments (list): List of dictionaries with start and end timestamps
        
    Returns:
        AudioSegment: The extracted audio segments
    """
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)
    output_audio = AudioSegment.silent(duration=0)
    crossfade_duration = 50  # milliseconds

    for i, segment in enumerate(matching_segments):
        start_ms = segment['start']
        end_ms = segment['end']
        audio_segment = audio[start_ms:end_ms]

        if i > 0:
            output_audio = output_audio.append(audio_segment, crossfade=crossfade_duration)
        else:
            output_audio += audio_segment

    return output_audio

def generate_audio_from_input(input_text):
    """
    Generate audio from input text by finding matching phrases in songs,
    downloading the audio, transcribing it, and extracting the relevant segments.
    
    Args:
        input_text (str): The input text to generate audio from
        
    Returns:
        str: Path to the generated audio file, or None if generation failed
    """
    # Step 1: Tokenize input text into phrases
    input_phrases = input_text.lower().split(' and ')
    print(f"Input phrases: {input_phrases}")

    # Step 2: Find matches for the phrases in the input text
    matches = find_longest_phrase_matches(input_text)
    if not matches:
        print("No matches found in Genius.")
        return None

    # Collect unique songs to process
    unique_songs = set()
    for phrase, song_info_list in matches:
        for song_info in song_info_list:
            unique_songs.add((song_info['title'], song_info['artist']))

    # Step 3: Process each song
    final_audio = AudioSegment.silent(duration=0)
    for title, artist in unique_songs:
        print(f"\nProcessing song '{title}' by '{artist}'...")
        youtube_url = search_youtube_video(title, artist)
        if youtube_url:
            print(f"Found YouTube video: {youtube_url}")

            # Sanitize the filename
            song_key = f"{title} - {artist}"
            safe_song_key = sanitize_filename(song_key)
            audio_file = f"{safe_song_key}.mp3"

            # Check if the audio file already exists
            if os.path.exists(audio_file):
                print(f"Audio file '{audio_file}' already exists. Skipping download.")
            else:
                # Download audio
                audio_file = download_audio(youtube_url, audio_file)
                if audio_file is None or not os.path.exists(audio_file):
                    print(f"Failed to download audio for '{title}' by '{artist}'.")
                    continue

            # Transcribe audio with word-level timestamps (with caching)
            try:
                transcription_words = transcribe_audio_with_word_timestamps(audio_file)
            except Exception as e:
                print(f"An error occurred during transcription: {e}")
                continue

            # Find matching phrases in the transcription
            matching_segments = find_matching_word_segments(input_phrases, transcription_words)
            if not matching_segments:
                print(f"No matching phrases found in the transcription of '{title}' by '{artist}'.")
                continue

            # Extract and concatenate audio segments
            song_audio = extract_audio_segments_by_phrases(audio_file, matching_segments)
            final_audio += song_audio
        else:
            print(f"No suitable YouTube video found for '{title}' by '{artist}'.")

    if len(final_audio) == 0:
        print("Failed to generate audio from the provided input.")
        return None

    # Step 4: Export the final audio file
    output_filename = "final_output.mp3"
    final_audio.export(output_filename, format="mp3")
    print(f"\nGenerated audio saved as '{output_filename}'")
    return output_filename

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Hip-Hop Voice Sampler: Create audio from matching phrases in hip-hop songs."
    )
    parser.add_argument(
        "--text", "-t", 
        type=str, 
        help="Input text to generate audio from"
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Check environment variables and FFmpeg
    if not check_environment():
        print("Environment check failed. Please fix the issues above and try again.")
        sys.exit(1)
    
    # Use command-line input if provided, otherwise use default
    user_input = args.text if args.text else "we gon be alright and we gon be together"
    print(f"Using input text: '{user_input}'")
    
    # Generate audio from input
    output_file = generate_audio_from_input(user_input)
    
    if output_file:
        print(f"Success! Audio generated and saved as '{output_file}'")
    else:
        print("Failed to generate audio. Please check the error messages above.")
        sys.exit(1)
