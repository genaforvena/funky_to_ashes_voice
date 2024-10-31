import logging
from googleapiclient.discovery import build

def verify_youtube_video(video_id: str, expected_title: str, youtube_api_key: str) -> bool:
    """
    Verify that the YouTube video contains the expected track title.
    """
    youtube = build('youtube', 'v3', developerKey=youtube_api_key)
    
    try:
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()
        
        if not response['items']:
            logging.warning(f"No video found for ID: {video_id}")
            return False
        
        video_title = response['items'][0]['snippet']['title']
        video_description = response['items'][0]['snippet']['description']
        
        logging.info(f"Video title: {video_title}")
        logging.info(f"Video description: {video_description}")
        
        # Check if the expected title is in the video title or description
        if expected_title.lower() in video_title.lower() or expected_title.lower() in video_description.lower():
            logging.info("Video matches the expected track title.")
            return True
        
        logging.info("Video does not match the expected track title.")
        return False
    
    except Exception as e:
        logging.error(f"Error verifying video: {e}")
        return False
