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

def generate_audio_from_input(input_text):
    # Step 1: Find matches for the phrases in the input text
    matches = find_longest_phrase_matches(input_text)
    if not matches:
        print("No matches found in Genius.")
        return

    print("Matches found:")
    for match in matches:
        print(f"- Phrase: '{match['phrase']}', Song: '{match['title']}' by '{match['artist']}'")

    # Collect phrases in order
    phrases = [match['phrase'] for match in matches]
    phrase_order = phrases.copy()

    # Step 2: For each unique song, download and process audio
    songs_processed = {}

    for match in matches:
        song_key = f"{match['title']} - {match['artist']}"
        if song_key in songs_processed:
            continue  # Skip if already processed

        print(f"\nProcessing song '{match['title']}' by '{match['artist']}'...")
        youtube_url = search_youtube_video(match['title'], match['artist'])
        if youtube_url:
            print(f"Found YouTube video: {youtube_url}")

            # Sanitize the filename
            safe_song_key = sanitize_filename(song_key)
            audio_file = f"{safe_song_key}.mp3"

            audio_file = download_audio(youtube_url, audio_file)
            if audio_file is None or not os.path.exists(audio_file):
                print(f"Failed to download audio for '{match['title']}' by '{match['artist']}'.")
                continue

            # Transcribe audio using Groq
            try:
                transcription_text, segments = transcribe_audio_groq(audio_file)
            except Exception as e:
                print(f"An error occurred during transcription with Groq: {e}")
                continue

            # Find phrases in transcription
            song_phrases = [m['phrase'] for m in matches if m['title'] == match['title'] and m['artist'] == match['artist']]
            phrase_segments = find_phrases_in_transcription(song_phrases, segments)

            songs_processed[song_key] = {
                'audio_file': audio_file,
                'phrase_segments': phrase_segments
            }
        else:
            print(f"No suitable YouTube video found for '{match['title']}' by '{match['artist']}'.")

    # Step 3: Assemble the final audio
    final_audio = AudioSegment.silent(duration=0)
    for phrase in phrase_order:
        found = False
        for song_key, data in songs_processed.items():
            if phrase in data['phrase_segments']:
                audio_segment = assemble_audio_segments(
                    data['audio_file'],
                    {phrase: data['phrase_segments'][phrase]},
                    [phrase]
                )
                final_audio += audio_segment
                found = True
                break
        if not found:
            print(f"Phrase '{phrase}' could not be assembled.")

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