import sys
from groq import Groq

def transcribe_audio(audio_path):
    client = Groq()
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(audio_path, file.read()),
            model="whisper-large-v3",
            response_format="text",
        )
    return transcription.strip().lower()

if __name__ == "__main__":
    input_text = " ".join(sys.argv[1:])
    output_transcription = transcribe_audio("final_output.mp3")

    print(f"Original input: '{input_text}'")
    print(f"Transcription of output: '{output_transcription}'")

    if input_text in output_transcription:
        print("Verification successful: Input text found in output transcription.")
    else:
        print("Verification failed: Input text not found in output transcription.")
