import re
import yt_dlp
import sys


def search_youtube_video(title, artist):
    """
    Search YouTube for a video matching the given song title and artist.
    
    Args:
        title (str): The song title to search for
        artist (str): The artist name to search for
        
    Returns:
        str: URL of the matching YouTube video, or None if no match is found
    """
    query = f"{title} {artist}"
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'ignoreerrors': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        search_url = f"ytsearch5:{query}"  # Search for top 5 results
        try:
            info = ydl.extract_info(search_url, download=False)
        except Exception as e:
            print(f"An error occurred during YouTube search: {e}", file=sys.stderr)
            return None

        if 'entries' not in info or not info['entries']:
            print("No videos found on YouTube for this song.")
            return None

        videos = info['entries']
        for video in videos:
            if video is None:
                continue
            video_title = video.get('title', '').lower()
            if title.lower() in video_title and artist.lower() in video_title:
                return video['webpage_url']

        # If no exact match is found, return the first video
        return videos[0]['webpage_url']


def sanitize_filename(name):
    """
    Remove invalid characters from a filename.
    
    Args:
        name (str): The filename to sanitize
        
    Returns:
        str: The sanitized filename
    """
    # Remove invalid characters for filenames
    return re.sub(r'[\\/*?:"<>|]', "", name)


def download_audio(youtube_url, output_audio):
    """
    Download audio from a YouTube URL.
    
    Args:
        youtube_url (str): The YouTube URL to download from
        output_audio (str): The desired output filename
        
    Returns:
        str: Path to the downloaded audio file, or None if download failed
    """
    output_audio = sanitize_filename(output_audio)
    # Remove the .mp3 extension if present
    if output_audio.lower().endswith('.mp3'):
        output_audio = output_audio[:-4]
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_audio}.%(ext)s',  # Include extension placeholder
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
        output_file = output_audio + '.mp3'
        return output_file
    except Exception as e:
        print(f"An error occurred during audio download: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    title = input("Enter the song title: ").strip()
    artist = input("Enter the artist name: ").strip()
    output_audio = input("Enter the desired output filename (without extension): ").strip()

    youtube_url = search_youtube_video(title, artist)
    if youtube_url:
        print(f"Found YouTube URL: {youtube_url}")
        downloaded_file = download_audio(youtube_url, output_audio)
        if downloaded_file:
            print(f"Audio downloaded successfully: {downloaded_file}")
        else:
            print("Failed to download audio.")
    else:
        print("Could not find the video on YouTube.")
