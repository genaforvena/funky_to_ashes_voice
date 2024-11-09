import re
import os
import sys
import time
import lyricsgenius
import yt_dlp
from groq import Groq
from pydub import AudioSegment

# Initialize Genius API
GENIUS_API_KEY = os.getenv("GENIUS_TOKEN")
genius = lyricsgenius.Genius(GENIUS_API_KEY, timeout=15)

def find_longest_phrase_matches(input_text):
    words = input_text.split()
    matches = []
    cache = {}

    # Generate all possible phrases, starting with the longest
    for length in range(len(words), 0, -1):
        for start in range(len(words) - length + 1):
            end = start + length
            phrase = " ".join(words[start:end])

            if phrase in cache:
                continue  # Skip if already searched
            cache[phrase] = True

            try:
                song = genius.search_song(phrase)
                if song:
                    matches.append({
                        'phrase': phrase,
                        'title': song.title,
                        'artist': song.artist,
                        'url': song.url,
                        'start_index': start,
                        'end_index': end
                    })
            except Exception as e:
                print(f"An error occurred while searching for '{phrase}': {e}")
            time.sleep(1)  # Delay to prevent rate limiting

    # Sort matches by their position in the input text
    matches.sort(key=lambda x: x['start_index'])
    return matches

def search_youtube_video(title, artist):
    query = f"{title} {artist}"
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'ignoreerrors': True,
        'no_warnings': True,
        # Remove or comment out the 'extract_flat' option
        # 'extract_flat': 'in_playlist',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        search_url = f"ytsearch5:{query}"  # Search for top 5 results
        info = ydl.extract_info(search_url, download=False)
        if 'entries' not in info or not info['entries']:
            print("No videos found on YouTube for this song.")
            return None

        videos = info['entries']
        for video in videos:
            if video is None:
                continue
            # Now 'webpage_url' should be available
            return video['webpage_url']

    print("No suitable YouTube video found.")
    return None


def sanitize_filename(name):
    # Remove invalid characters for filenames
    return re.sub(r'[\\/*?:"<>|]', "", name)


def download_audio(youtube_url, output_audio):
    output_audio = sanitize_filename(output_audio)
    # Remove the .mp3 extension if present
    if output_audio.lower().endswith('.mp3'):
        output_audio = output_audio[:-4]
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_audio,  # No .mp3 extension here
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,  # Set to False to see output for debugging
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        # After download, append .mp3 to the filename
        output_audio += '.mp3'
        return output_audio
    except Exception as e:
        print(f"An error occurred during audio download: {e}")
        return None

def transcribe_audio_groq(audio_path):
    client = Groq()
    filename = audio_path  # Path to your audio file

    print(f"Transcribing audio file: {filename}")
    try:
        with open(filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
            file=(filename, file.read()),
            model="whisper-large-v3-turbo",
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
    phrase_segments = {}
    for phrase in phrases:
        phrase_lower = phrase.lower()
        for text, start, end in timestamps:
            if phrase_lower in text:
                if phrase not in phrase_segments or start < phrase_segments[phrase]['start']:
                    phrase_segments[phrase] = {'start': start, 'end': end}
    return phrase_segments

def assemble_audio_segments(audio_path, phrase_segments, phrase_order):
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
    import re
    # Remove punctuation and convert to lowercase
    tokens = re.findall(r'\b\w+\b', input_text.lower())
    return set(tokens)

def transcribe_audio_with_word_timestamps(audio_path):
    client = Groq()

    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(audio_path, file.read()),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )

    # Extract word-level timestamps
    words = []
    for segment in transcription.segments:
        print(segment)
        words.append({
            'word': segment['text'].strip().lower(),
            'start': float(segment['start']),
            'end': float(segment['end'])
        })

    return words

def extract_audio_segments_by_words(audio_path, matching_segments):
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

def generate_audio_from_input(input_text):
    # Step 1: Tokenize input text
    input_words = tokenize_input_text(input_text)
    print(f"Input words: {input_words}")

    # Step 2: Find matches for the phrases in the input text
    matches = find_longest_phrase_matches(input_text)
    if not matches:
        print("No matches found in Genius.")
        return

    # Collect unique songs to process
    unique_songs = {(match['title'], match['artist']) for match in matches}

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
                # Proceed to transcription
            else:
                # Download audio
                audio_file = download_audio(youtube_url, audio_file)
                if audio_file is None or not os.path.exists(audio_file):
                    print(f"Failed to download audio for '{title}' by '{artist}'.")
                    continue

            # Transcribe audio with word-level timestamps
            try:
                transcription_words = transcribe_audio_with_word_timestamps(audio_file)
            except Exception as e:
                print(f"An error occurred during transcription: {e}")
                continue

            print(f"Transcription words: {transcription_words}")
            # Find matching word segments
            matching_segments = find_matching_word_segments(input_words, transcription_words)
            if not matching_segments:
                print(f"No matching words found in the transcription of '{title}' by '{artist}'.")
                continue

            print(f"Matching segments: {matching_segments}")
            # Extract and concatenate audio segments
            song_audio = extract_audio_segments_by_words(audio_file, matching_segments)
            final_audio += song_audio
        else:
            print(f"No suitable YouTube video found for '{title}' by '{artist}'.")

    if len(final_audio) == 0:
        print("Failed to generate audio from the provided input.")
        return

    # Step 4: Export the final audio file
    output_filename = "final_output.mp3"
    final_audio.export(output_filename, format="mp3")
    print(f"\nGenerated audio saved as '{output_filename}'")

if __name__ == "__main__":
    user_input = "we gon be alright and we gon be together"
    generate_audio_from_input(user_input)