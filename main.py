import sys
import os
import argparse
import logging
from get_genius_lyrics import LyricsSplitter
from clip_processor import ClipProcessor
from youtube_search import search_youtube_video_ids

def combine_quotes_to_audio(input_text: str, genius_token: str, youtube_api_key: str, output_dir: str = 'output'):
    # Initialize components
    splitter = LyricsSplitter(genius_token)
    clip_processor = ClipProcessor(output_dir=output_dir)
    remaining_text = input_text
    all_results = []
    
    while remaining_text:
        score, phrases = splitter.split_lyrics(remaining_text)
        phrases_str = [str(phrase) for phrase in phrases]
        
        print(f"Attempting split: Score {score}: {' | '.join(phrases_str)}")
        
        # Get titles and artists with improved matching
        all_titles_and_artists = []
        unmatched_phrases = []
        
        for phrase in phrases:
            matches = splitter.get_title_and_artist(phrase)
            if matches:
                all_titles_and_artists.extend(matches)
                logging.info(f"Found {len(matches)} potential matches for phrase '{phrase}'")
            else:
                # Try breaking down the phrase into smaller segments
                words = str(phrase).split()
                if len(words) > 2:
                    # Try sliding window of 2-3 words
                    for i in range(len(words) - 1):
                        sub_phrase = ' '.join(words[i:i+2])
                        sub_matches = splitter.get_title_and_artist(sub_phrase)
                        if sub_matches:
                            all_titles_and_artists.extend(sub_matches)
                            logging.info(f"Found {len(sub_matches)} potential matches for sub-phrase '{sub_phrase}'")
                            break
                    else:
                        unmatched_phrases.append(str(phrase))
                else:
                    unmatched_phrases.append(str(phrase))
                    
        if not all_titles_and_artists:
            if len(phrases) > 1:
                splitter.reduce_chunk_size()
                print("No matches found, reducing chunk size and retrying...")
                continue
            else:
                print(f"Warning: Could not find any matches for: {remaining_text}")
                break

        # Search YouTube
        video_ids = search_youtube_video_ids(all_titles_and_artists, youtube_api_key)
        
        if video_ids:
            # Process videos with improved processor
            results = clip_processor.process_videos(video_ids, phrases)
            
            if results:
                all_results.extend(results)
                matched_phrases = {result['phrase'] for result in results}
                remaining_phrases = [phrase for phrase in unmatched_phrases 
                                  if phrase not in matched_phrases]
                remaining_text = ' '.join(remaining_phrases)
                
                if remaining_text:
                    print(f"Remaining text to process: {remaining_text}")
            else:
                if len(phrases) > 1:
                    splitter.reduce_chunk_size()
                    print("No matches found, reducing chunk size and retrying...")
                else:
                    print(f"Warning: Could not find match for: {remaining_text}")
                    break
        else:
            print(f"No YouTube videos found for phrases")
            break
    
    # Sort and combine results
    if all_results:
        all_results.sort(key=lambda x: input_text.find(x['phrase']))
        combined_audio_path = os.path.join(output_dir, 'combined_audio.mp3')
        
        if clip_processor.combine_clips(all_results, combined_audio_path):
            print(f"Combined audio saved to: {combined_audio_path}")
        else:
            print("Error combining audio clips")
    else:
        print("No matches found for any phrases")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine quotes to audio.')
    parser.add_argument('input_text', type=str, help='The input text containing song quotes.')
    parser.add_argument('--output-dir', type=str, default='output', 
                      help='Directory to save output files')
    args = parser.parse_args()

    genius_token = os.getenv('GENIUS_TOKEN')
    youtube_api_key = os.getenv('YOUTUBE_API_KEY')
    
    if not genius_token or not youtube_api_key:
        print("Error: Missing API tokens. Please set GENIUS_TOKEN and YOUTUBE_API_KEY environment variables.")
        sys.exit(1)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        results = combine_quotes_to_audio(
            args.input_text, 
            genius_token, 
            youtube_api_key,
            args.output_dir
        )
        
        if results:
            print("\nProcessed clips:")
            print("-" * 50)
            for result in results:
                print(f"\nPhrase: '{result['phrase']}'")
                print(f"Video ID: {result['video_id']}")
                print(f"Similarity: {result['similarity']:.2f}")
                print(f"Clip path: {result['clip_path']}")
                print("-" * 30)
    except Exception as e:
        print(f"Error processing quotes: {str(e)}")
        sys.exit(1)