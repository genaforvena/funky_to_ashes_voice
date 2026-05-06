# funky to ashes voice

Takes a text phrase, finds hip-hop songs whose lyrics contain it, downloads the audio, and stitches together the exact moments those words are sung into a new MP3.

## How it works

1. Searches [Genius](https://genius.com) for songs whose lyrics contain the longest possible substrings of your input text.
2. Downloads audio for each matched song from YouTube via `yt-dlp`.
3. Transcribes the audio with Groq (Whisper) to get word-level timestamps. Transcriptions are cached by file checksum so re-runs are fast.
4. Extracts the audio segments where the matched phrases are spoken and crossfades them together.
5. Exports the result as `final_output.mp3`.

## Requirements

- Python 3.7+
- FFmpeg on your PATH
- A [Genius API key](https://genius.com/api-clients)
- A [Groq API key](https://console.groq.com)

## Setup

```bash
git clone https://github.com/genaforvena/funky_to_ashes_voice.git
cd funky_to_ashes_voice
pip install -r requirements.txt
```

Set environment variables:

```bash
export GENIUS_TOKEN=your_genius_api_key
export GROQ_API_KEY=your_groq_api_key
```

On Windows:

```cmd
set GENIUS_TOKEN=your_genius_api_key
set GROQ_API_KEY=your_groq_api_key
```

## Usage

Edit the `user_input` line at the bottom of `main.py` and run:

```bash
python main.py
```

The output is saved as `final_output.mp3` in the current directory.

## License

[CC0 1.0 Universal](LICENSE) — public domain.
