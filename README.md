# Hip-Hop Voice Sampler

A Python tool that extracts individual words from hip-hop songs using AI transcription, creating a personal "speech synthesizer" that talks using actual rap samples. Think of it as creating your own vocabulary of words spoken by your favorite artists.

## 🎯 Project Goal

Create a text-to-speech system that:
1. Takes a hip-hop song and its lyrics as input
2. Cuts out individual words using AI transcription for timing
3. Creates a database of word samples that can be used to "speak" new sentences using the artist's voice

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/hip-hop-voice-sampler.git
cd hip-hop-voice-sampler

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the sampler
python sampler.py --audio song.mp3 --lyrics lyrics.txt --output samples/
```

## 📋 Requirements

- Python 3.8+
- FFmpeg (for audio processing)
- GPU recommended but not required

Required Python packages (see requirements.txt):
```
torch
torchaudio
transformers
pydub
numpy
librosa
```

## 🎵 Usage

1. Prepare your input files:
   - An MP3 file of the song
   - A text file with the lyrics

2. Run the sampler:
```python
from hip_hop_sampler import LLMWordSampler

sampler = LLMWordSampler(
    audio_path="your_song.mp3",
    lyrics_path="your_lyrics.txt",
    output_dir="samples"
)
sampler.process()
```

3. Find your samples in the output directory:
   - Individual MP3 files for each word
   - metadata.json with timing information

## 🛠 How It Works

1. **Transcription**: Uses Whisper ASR model to transcribe the audio and get word timestamps
2. **Matching**: Matches transcribed words with provided lyrics
3. **Extraction**: Cuts out word samples with a small padding to preserve natural sound
4. **Storage**: Saves individual words as MP3 files with metadata

## 📂 Project Structure

```
hip-hop-voice-sampler/
├── sampler/
│   ├── __init__.py
│   ├── transcriber.py    # ASR and word timing
│   ├── extractor.py      # Audio extraction
│   └── utils.py          # Helper functions
├── examples/             # Example scripts
├── tests/               # Test files
├── requirements.txt     # Dependencies
├── setup.py            # Package setup
└── README.md           # This file
```

## ⚠️ Limitations

- Works best with clear vocal tracks
- ASR may struggle with very fast rap sections
- Background music affects sample quality
- No cleaning/isolation of vocals from music

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 TODO

- [ ] Add vocal isolation option
- [ ] Improve word boundary detection
- [ ] Add a simple GUI
- [ ] Create a word sample database format
- [ ] Add playback functionality

## 📄 License

MIT License - see LICENSE file for details