from typing import List, Dict, Tuple, Set
import os
import logging
from pydub import AudioSegment
from youtube_cache import YouTubeCache
from phrase_extractor import PhraseExtractor
from phrase_extractor import download_audio
from get_captions import get_captions

class ClipProcessor:
    def __init__(self, output_dir: str = 'output', temp_dir: str = 'temp'):
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.youtube_cache = YouTubeCache()
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

    def process_videos(self, video_ids: List[str], phrases: List[str]) -> List[Dict]:
        """
        Process videos to find and extract clips with enhanced matching and audio handling
        
        Args:
            video_ids: List of YouTube video IDs
            phrases: List of phrases to search for
            
        Returns:
            List of dictionaries containing match information and clip paths
        """
        # Initialize tracking variables
        extractor = PhraseExtractor(
            phrases=phrases,
            lead_seconds=0.5,
            trail_seconds=1.0,
            min_match_ratio=0.8
        )
        best_matches: Dict[str, Dict] = {}
        processed_phrases: Set[str] = set()

        # First process cached videos
        self._process_cached_videos(
            extractor=extractor,
            best_matches=best_matches,
            processed_phrases=processed_phrases
        )
        # Then process new videos
        remaining_videos = [vid for vid in video_ids 
                          if not self.youtube_cache.get_cached_data(vid, 'captions')]
        self._process_new_videos(
            video_ids=remaining_videos,
            extractor=extractor,
            best_matches=best_matches,
            processed_phrases=processed_phrases
        )

        return list(best_matches.values())

    def _process_cached_videos(self, extractor: PhraseExtractor, 
                             best_matches: Dict[str, Dict], 
                             processed_phrases: Set[str]) -> None:
        """Process all videos from cache"""
        cached_videos = self.youtube_cache.get_all_cached_data('captions')
        
        for video_id, cached_data in cached_videos.items():
            captions = cached_data.get('captions')
            if not captions:
                continue

            unmatched_phrases = [p for p in extractor.phrases if str(p) not in processed_phrases]
            if not unmatched_phrases:
                break

            matches = extractor.process_captions(captions)
            if matches:
                self._process_matches(
                    video_id=video_id,
                    matches=matches,
                    best_matches=best_matches,
                    processed_phrases=processed_phrases
                )

    def _process_new_videos(self, video_ids: List[str], extractor: PhraseExtractor,
                          best_matches: Dict[str, Dict], 
                          processed_phrases: Set[str]) -> None:
        """Process new videos not in cache"""
        for video_id in video_ids:
            if len(processed_phrases) == len(extractor.phrases):
                break

            try:
                captions = get_captions(video_id)  # Import this from your captions module
                if not captions:
                    continue

                matches = extractor.process_captions(captions)
                if matches:
                    self._process_matches(
                        video_id=video_id,
                        matches=matches,
                        best_matches=best_matches,
                        processed_phrases=processed_phrases
                    )

            except Exception as e:
                logging.error(f"Error processing video {video_id}: {str(e)}")

    def _process_matches(self, video_id: str, matches: List[Dict],
                        best_matches: Dict[str, Dict],
                        processed_phrases: Set[str]) -> None:
        """Process matches for a single video"""
        audio_path = None
        try:
            audio_path = download_audio(video_id, self.temp_dir)
            if not audio_path:
                return

            audio = AudioSegment.from_file(audio_path)
            
            for match in matches:
                phrase = match['phrase']
                similarity = match['similarity']
                
                clip_path = self._create_clip(
                    audio=audio,
                    match=match,
                    video_id=video_id
                )
                
                if clip_path:
                    match['clip_path'] = clip_path
                    match['video_id'] = video_id
                    
                    if (phrase not in best_matches or 
                        similarity > best_matches[phrase]['similarity']):
                        
                        # Remove old clip if exists
                        if (phrase in best_matches and 
                            'clip_path' in best_matches[phrase] and
                            os.path.exists(best_matches[phrase]['clip_path'])):
                            os.remove(best_matches[phrase]['clip_path'])
                        
                        best_matches[phrase] = match
                        processed_phrases.add(str(phrase))
                        logging.info(f"Saved best match for '{phrase}' from video {video_id}")

        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logging.error(f"Error removing audio file: {str(e)}")

    def _create_clip(self, audio: AudioSegment, match: Dict, video_id: str) -> str:
        """Create audio clip with enhanced audio processing"""
        try:
            # Create safe filename
            safe_phrase = "".join(x for x in match['phrase'] if x.isalnum() or x.isspace())
            safe_phrase = safe_phrase.strip().replace(' ', '_').lower()[:30]
            clip_path = os.path.join(self.output_dir, f"clip_{video_id}_{safe_phrase}.mp3")

            # Extract clip with precise timing
            start_ms = int(match['start_time'] * 1000)
            end_ms = int(match['end_time'] * 1000)
            clip = audio[start_ms:end_ms]

            # Apply audio enhancements
            clip = clip.normalize()  # Normalize volume
            
            # Add small fade in/out
            fade_ms = 50
            clip = clip.fade_in(fade_ms).fade_out(fade_ms)

            # Export with high quality
            clip.export(
                clip_path,
                format="mp3",
                parameters=[
                    "-q:a", "0",  # Highest quality
                    "-filter:a", "loudnorm",  # Normalize audio levels
                ]
            )
            
            return clip_path

        except Exception as e:
            logging.error(f"Error creating clip: {str(e)}")
            return ""

    def combine_clips(self, results: List[Dict], output_path: str) -> bool:
        """Combine clips with enhanced audio processing"""
        try:
            if not results:
                return False

            # Collect and verify clips
            clips = []
            for result in results:
                clip_path = result.get('clip_path')
                if clip_path and os.path.exists(clip_path):
                    clip = AudioSegment.from_file(clip_path)
                    clips.append(clip)
                else:
                    logging.warning(f"Missing clip for phrase: '{result['phrase']}'")

            if not clips:
                return False

            # Create crossfade silence
            silence = AudioSegment.silent(duration=100)  # 100ms silence
            
            # Combine clips with crossfade
            final_audio = clips[0]
            for clip in clips[1:]:
                final_audio = final_audio.append(silence, crossfade=50)
                final_audio = final_audio.append(clip, crossfade=50)

            # Normalize final audio
            final_audio = final_audio.normalize()

            # Export with high quality
            final_audio.export(
                output_path,
                format="mp3",
                parameters=[
                    "-q:a", "0",
                    "-filter:a", "loudnorm"
                ]
            )
            
            logging.info(f"Combined audio saved to: {output_path}")
            return True

        except Exception as e:
            logging.error(f"Error combining clips: {str(e)}")
            return False