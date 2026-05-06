import json
import hashlib
import os
import sys
from groq import Groq
from pydub import AudioSegment
from quotes_extractor import find_longest_phrase_matches
from audio_downloader import search_youtube_video, download_audio, sanitize_filename


CACHE_DIR = 'cache'


def get_file_checksum(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()


def transcribe_audio_with_word_timestamps(audio_path):
    client = Groq()

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    checksum = get_file_checksum(audio_path)
    cache_filename = os.path.join(CACHE_DIR, f"{checksum}_transcription.json")

    if os.path.exists(cache_filename):
        print(f"Loading cached transcription for '{audio_path}'")
        with open(cache_filename, 'r', encoding='utf-8') as cache_file:
            return json.load(cache_file)

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

    transcription_words = []
    for segment in transcription.segments:
        transcription_words.append({
            'word': segment['text'].strip().lower(),
            'start': float(segment['start']),
            'end': float(segment['end'])
        })

    with open(cache_filename, 'w', encoding='utf-8') as cache_file:
        json.dump(transcription_words, cache_file, ensure_ascii=False, indent=2)

    return transcription_words


def find_matching_word_segments(input_phrases, transcription_words):
    full_text = ''
    positions = []
    for word_info in transcription_words:
        start_time = int(word_info['start'] * 1000)
        end_time = int(word_info['end'] * 1000)
        text = word_info['word']
        full_text += text + ' '
        positions.append({'start': start_time, 'end': end_time, 'text': text})
    full_text = full_text.strip().lower()

    matching_segments = []
    for phrase in input_phrases:
        phrase_lower = phrase.lower()
        index = full_text.find(phrase_lower)
        if index != -1:
            accumulated_length = 0
            segment_start = None
            segment_end = None
            for pos in positions:
                text = pos['text'].lower()
                text_length = len(text) + 1
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


def extract_audio_segments_by_phrases(audio_path, matching_segments):
    audio = AudioSegment.from_file(audio_path)
    output_audio = AudioSegment.silent(duration=0)
    crossfade_duration = 50

    for i, segment in enumerate(matching_segments):
        audio_segment = audio[segment['start']:segment['end']]
        if i > 0:
            output_audio = output_audio.append(audio_segment, crossfade=crossfade_duration)
        else:
            output_audio += audio_segment

    return output_audio


def generate_audio_from_input(input_text):
    # matches: list of (phrase, [song_info, ...]) sorted longest phrase first
    matches = find_longest_phrase_matches(input_text)
    if not matches:
        print("No matches found in Genius.")
        return

    # Map each phrase to the first matching song
    phrase_song_pairs = []
    for phrase, song_info_list in matches:
        if song_info_list:
            song_info = song_info_list[0]
            phrase_song_pairs.append((phrase, song_info['title'], song_info['artist']))

    if not phrase_song_pairs:
        print("No usable matches found.")
        return

    # Download each unique song once
    unique_songs = {(title, artist) for _, title, artist in phrase_song_pairs}
    song_audio_files = {}

    for title, artist in unique_songs:
        print(f"\nProcessing song '{title}' by '{artist}'...")
        youtube_url = search_youtube_video(title, artist)
        if not youtube_url:
            print(f"No suitable YouTube video found for '{title}' by '{artist}'.")
            continue

        print(f"Found YouTube video: {youtube_url}")
        safe_name = sanitize_filename(f"{title} - {artist}")
        audio_file = f"{safe_name}.mp3"

        if os.path.exists(audio_file):
            print(f"Audio file '{audio_file}' already exists. Skipping download.")
        else:
            audio_file = download_audio(youtube_url, audio_file)
            if audio_file is None or not os.path.exists(audio_file):
                print(f"Failed to download audio for '{title}' by '{artist}'.")
                continue

        song_audio_files[(title, artist)] = audio_file

    # Assemble final audio in phrase order
    final_audio = AudioSegment.silent(duration=0)

    for phrase, title, artist in phrase_song_pairs:
        key = (title, artist)
        if key not in song_audio_files:
            print(f"Skipping phrase '{phrase}' — song not downloaded.")
            continue

        audio_file = song_audio_files[key]

        try:
            transcription_words = transcribe_audio_with_word_timestamps(audio_file)
        except Exception as e:
            print(f"Transcription error for '{title}': {e}")
            continue

        matching_segments = find_matching_word_segments([phrase], transcription_words)
        if not matching_segments:
            print(f"Phrase '{phrase}' not found in transcription of '{title}'.")
            continue

        segment_audio = extract_audio_segments_by_phrases(audio_file, matching_segments)
        final_audio += segment_audio

    if len(final_audio) == 0:
        print("Failed to generate audio from the provided input.")
        return

    output_filename = "final_output.mp3"
    final_audio.export(output_filename, format="mp3")
    print(f"\nGenerated audio saved as '{output_filename}'")


if __name__ == "__main__":
    user_input = "Be like a black hole never giving yourself away."
    generate_audio_from_input(user_input)
