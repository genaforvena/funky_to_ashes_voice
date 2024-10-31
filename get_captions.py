from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from pytube import YouTube
import logging
import time
from youtube_cache import YouTubeCache

@dataclass
class CaptionResult:
    video_id: str
    captions: Optional[List[Dict]]
    source: str  # 'api_direct', 'api_manual', 'fallback'
    error: Optional[str] = None

class EnhancedCaptionFetcher:
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = self._setup_logger()
        self.cache = YouTubeCache()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('CaptionFetcher')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _try_direct_api(self, video_id: str) -> Optional[List[Dict]]:
        """Attempt to fetch captions directly using the API."""
        try:
            captions = YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=['en', 'en-GB', 'en-US'],
                preserve_formatting=True
            )
            return captions
        except Exception as e:
            self.logger.debug(f"Direct API attempt failed: {str(e)}")
            return None

    def _try_manual_language_selection(self, video_id: str) -> Optional[List[Dict]]:
        """Try to fetch captions by manually finding available transcripts."""
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # First try English variants
            for lang in ['en', 'en-GB', 'en-US']:
                try:
                    transcript = transcript_list.find_generated_transcript([lang])
                    return transcript.fetch()
                except:
                    continue

            # If no English auto-generated, try any English
            for lang in ['en', 'en-GB', 'en-US']:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    return transcript.fetch()
                except:
                    continue

            # Last resort: try to get any transcript and translate to English
            try:
                transcript = transcript_list.find_transcript(['en'])
                translated = transcript.translate('en')
                return translated.fetch()
            except:
                return None

        except Exception as e:
            self.logger.debug(f"Manual language selection failed: {str(e)}")
            return None

    def _try_pytube_fallback(self, video_id: str) -> Optional[List[Dict]]:
        """Try to fetch captions using pytube as a fallback."""
        try:
            yt = YouTube(f'https://youtube.com/watch?v={video_id}')
            captions_dict = yt.captions
            
            # Try to get English captions using dictionary access
            caption_track = None
            for code, caption in captions_dict.items():
                if code.startswith('en'):
                    caption_track = caption
                    break
            
            if caption_track:
                # Convert pytube captions to youtube_transcript_api format
                xml_captions = caption_track.xml_captions
                srt_captions = caption_track.generate_srt_captions()
                
                # Parse SRT format into list of dictionaries
                parsed_captions = []
                current_time = 0.0
                
                for line in srt_captions.split('\n\n'):
                    if line.strip():
                        parts = line.split('\n')
                        if len(parts) >= 3:
                            timing = parts[1].split(' --> ')
                            start_time = self._convert_timestamp_to_seconds(timing[0])
                            text = ' '.join(parts[2:])
                            
                            parsed_captions.append({
                                'text': text,
                                'start': start_time,
                                'duration': 0  # We'll calculate this in post-processing
                            })
                
                # Post-process to add durations
                for i in range(len(parsed_captions) - 1):
                    parsed_captions[i]['duration'] = (
                        parsed_captions[i + 1]['start'] - parsed_captions[i]['start']
                    )
                
                # Set a default duration for the last caption
                if parsed_captions:
                    parsed_captions[-1]['duration'] = 5.0
                
                return parsed_captions
            
            self.logger.debug(f"No English captions found for video {video_id}")
            return None
                
        except Exception as e:
            self.logger.debug(f"Pytube fallback failed: {str(e)}")
            return None

    def _convert_timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert SRT timestamp to seconds."""
        hours, minutes, seconds = timestamp.replace(',', '.').split(':')
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

    def get_captions(self, video_id: str) -> CaptionResult:
        """
        Main method to fetch captions using multiple strategies.
        Returns a CaptionResult with the captions and metadata about how they were obtained.
        """
        self.logger.info(f"Fetching captions for video {video_id}")

        # Check cache first
        cached_result = self.cache.get_cached_data(video_id, 'captions')
        if cached_result:
            self.logger.info(f"Found cached captions for video {video_id}")
            return CaptionResult(
                video_id=video_id,
                captions=cached_result.get('captions'),
                source=cached_result.get('source', 'cache')
            )

        # Try each method in sequence
        for attempt in range(self.max_retries):
            # 1. Try direct API access
            captions = self._try_direct_api(video_id)
            if captions:
                # Cache successful result
                self.cache.save_to_cache(video_id, 'captions', {
                    'captions': captions,
                    'source': 'api_direct'
                })
                return CaptionResult(video_id, captions, 'api_direct')

            # 2. Try manual language selection
            captions = self._try_manual_language_selection(video_id)
            if captions:
                # Cache successful result
                self.cache.save_to_cache(video_id, 'captions', {
                    'captions': captions,
                    'source': 'api_manual'
                })
                return CaptionResult(video_id, captions, 'api_manual')

            # 3. Try pytube fallback
            captions = self._try_pytube_fallback(video_id)
            if captions:
                # Cache successful result
                self.cache.save_to_cache(video_id, 'captions', {
                    'captions': captions,
                    'source': 'fallback'
                })
                return CaptionResult(video_id, captions, 'fallback')

            # If all methods failed, wait before retrying
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        # If all attempts failed
        return CaptionResult(
            video_id=video_id,
            captions=None,
            source='none',
            error="Failed to fetch captions using all available methods"
        )

    def debug_available_captions(self, video_id: str) -> None:
        """
        Debug method to print information about available captions.
        """
        try:
            yt = YouTube(f'https://youtube.com/watch?v={video_id}')
            captions_dict = yt.captions
            
            self.logger.info(f"Available caption tracks for video {video_id}:")
            for code, caption in captions_dict.items():
                self.logger.info(f"Language code: {code}")
                
        except Exception as e:
            self.logger.error(f"Error checking available captions: {str(e)}")

def get_captions(video_id: str) -> List[Dict]:
    """
    Drop-in replacement for the original get_captions function.
    Returns captions in the same format as the original function or None if not found.
    """
    fetcher = EnhancedCaptionFetcher()
    
    # Debug available captions
    fetcher.debug_available_captions(video_id)
    
    result = fetcher.get_captions(video_id)
    return result.captions


if __name__ == "__main__":
    video_id = "ZM5_6js19eM"
    captions = get_captions(video_id)
    print(captions)
