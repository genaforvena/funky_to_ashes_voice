from typing import List, Tuple
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def search_youtube_video_ids(titles_and_artists: List[Tuple[str, str]], api_key: str) -> List[str]:
    """
    Search YouTube for video IDs based on song titles and artists.
    
    Args:
        titles_and_artists: List of tuples containing (title, artist)
        api_key: YouTube Data API key
    
    Returns:
        List of YouTube video IDs
    """
    youtube = build('youtube', 'v3', developerKey=api_key)
    video_ids = []
    
    for title, artist in titles_and_artists:
        try:
            # Search for the exact song using both title and artist
            query = f"{title} {artist} official"
            request = youtube.search().list(
                part="id",
                q=query,
                type="video",
                maxResults=3,
                videoCategoryId="10"  # Music category
            )
            response = request.execute()
            
            # Extract video IDs from response
            for item in response['items']:
                video_ids.append(item['id']['videoId'])
                
        except HttpError as e:
            print(f"Error searching YouTube for {title} - {artist}: {str(e)}")
            continue
    
    return video_ids 