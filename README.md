# Hip-Hop Voice Sampler

## Features
- **Phrase Matching**: Identifies songs whose lyrics match phrases in the input text using the Genius API.
- **Audio Downloading**: Searches for and downloads audio from YouTube using `yt-dlp`.
- **Transcription**: Transcribes audio using the Groq client for word-level timestamps.
- **Audio Processing**: Extracts and assembles audio segments corresponding to matching words using `pydub`.
- **Output Generation**: Creates a final audio file (`final_output.mp3`) containing the stitched audio segments.

## Prerequisites
- Python 3.7 or higher
- **Genius API Key**: Obtain from [Genius API](https://genius.com/api-clients).
- **Groq API Access**: Access to the Groq transcription service (contact Groq for API details).
- **FFmpeg**: Required by `yt-dlp` and `pydub` for audio processing.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/audio-phrase-stitcher.git
cd audio-phrase-stitcher
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

   ### 3. Install Required Python Packages
   Install the required packages using `pip`:
   ```bash
   pip install -r requirements.txt
   ```
   If a `requirements.txt` file is not provided, install the packages individually:
   ```bash
   pip install lyricsgenius yt-dlp groq pydub
   ```

   ### 4. Install FFmpeg
   - For Windows:
     - Download FFmpeg from [FFmpeg Windows Builds](https://ffmpeg.org/download.html).
     - Add the `bin` directory of FFmpeg to your system's PATH environment variable.
   - For macOS (using Homebrew):
     ```bash
     brew install ffmpeg
     ```
   - For Linux (Debian/Ubuntu):
     ```bash
     sudo apt-get install ffmpeg
     ```

   ## Configuration

   ### 1. Set Environment Variables
   **Genius API Key**
   ```bash
   export GENIUS_TOKEN='your_genius_api_key'
   ```
   On Windows Command Prompt:
   ```cmd
   set GENIUS_TOKEN=your_genius_api_key
   ```

   **Groq API Key**
   If the Groq client requires an API key:
   ```bash
   export GROQ_API_KEY='your_groq_api_key'
   ```
   On Windows Command Prompt:
   ```cmd
   set GROQ_API_KEY=your_groq_api_key
   ```

   ### 2. Verify FFmpeg Installation
   ```bash
   ffmpeg -version
   ```

   ## Usage

   ### 1. Run the Script
   ```bash
   python script_name.py
   ```
   Replace `script_name.py` with the actual filename.

   ### 2. Provide Input Text
   By default, the script uses:
   ```python
   user_input = "we gon be alright and we gon be together"
   ```
   To use custom input, modify this line or adapt the script to accept command-line arguments or user input.

   ### 3. Output
   The final audio file `final_output.mp3` will be saved in the current directory.

   ## How It Works
   1. **Tokenization**: The input text is tokenized into words.
   2. **Phrase Matching**: Searches for the longest phrases in the input text that match song lyrics using the Genius API.
   3. **YouTube Search and Download**: For each matched song, searches YouTube and downloads the audio using `yt-dlp`.
   4. **Transcription**: Transcribes the audio using the Groq client to obtain word-level timestamps.
   5. **Matching Word Segments**: Identifies timestamps of matching words in the transcription.
   6. **Audio Extraction and Assembly**: Extracts the corresponding audio segments and assembles them using `pydub`.
   7. **Output Generation**: The final audio is exported as `final_output.mp3`.

   ## Troubleshooting
   - **Genius API Errors**: Ensure your Genius API key is correct and you haven't exceeded rate limits.
   - **YouTube Download Issues**: Check your internet connection and update `yt-dlp` if necessary:
     ```bash
     pip install -U yt-dlp
     ```
   - **Transcription Errors**: Verify your Groq API key and ensure you have access to the required transcription models.
   - **FFmpeg Not Found**: Make sure FFmpeg is installed and added to your system's PATH.
   - **Module Import Errors**: Ensure all dependencies are installed in your Python environment.

   ## Dependencies
   - `lyricsgenius`: For accessing the Genius API.
   - `yt-dlp`: For downloading audio from YouTube.
   - `groq`: For audio transcription services.
   - `pydub`: For audio manipulation.
   - **FFmpeg**: Required by `pydub` and `yt-dlp` for processing audio files.
   
`
