import time
import logging
import os
import subprocess
import moviepy as mp
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_audio_segment_ffmpeg(video_path, start_time, duration, output_path):
    """Extract audio segment using ffmpeg directly to avoid MoviePy issues"""
    cmd = [
        'ffmpeg',
        '-ss', str(start_time),
        '-t', str(duration),
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',
        '-ar', '44100',
        '-ac', '2',
        '-y',  # Overwrite output
        output_path
    ]
    
    # Run ffmpeg quietly
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def process_video_segments_safe(video_path, segment_duration=300, hop_length=512):
    """
    Process video in segments using ffmpeg for extraction to avoid MoviePy audio issues
    
    Args:
        video_path: Path to video file
        segment_duration: Duration of each segment in seconds (default 5 minutes)
        hop_length: Hop length for librosa analysis
    """
    logger.info(f"Processing video: {video_path}")
    
    # Get video duration using MoviePy
    with mp.VideoFileClip(video_path) as video_clip:
        total_duration = video_clip.duration
        logger.info(f"Video duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
        
        if video_clip.audio is None:
            logger.error("No audio track found")
            return []
    
    all_boundaries = []
    
    # Process in segments
    num_segments = int(np.ceil(total_duration / segment_duration))
    logger.info(f"Processing in {num_segments} segments of {segment_duration}s each")
    
    # Limit to first 5 segments for testing (remove this line for full processing)
    num_segments = min(num_segments, 20)
    
    for i in range(num_segments):
        start_time = i * segment_duration
        duration = min(segment_duration, total_duration - start_time)
        end_time = start_time + duration
        
        logger.info(f"\nProcessing segment {i+1}/{num_segments}: {start_time:.1f}s - {end_time:.1f}s")
        
        # Create temporary file for audio segment
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Extract audio segment using ffmpeg
            logger.info(f"  Extracting audio with ffmpeg...")
            extract_audio_segment_ffmpeg(video_path, start_time, duration, temp_path)
            
            # Load with librosa
            y, sr = librosa.load(temp_path, sr=None)
            logger.info(f"  Audio loaded: {len(y)/sr:.1f}s at {sr}Hz")
            
            # Compute features
            chromagram = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
            novelty = librosa.onset.onset_strength(sr=sr, S=chromagram, hop_length=hop_length)
            
            # Detect peaks
            peaks = librosa.util.peak_pick(
                novelty, 
                pre_max=10, 
                post_max=10, 
                pre_avg=20, 
                post_avg=20, 
                delta=0.2, 
                wait=10
            )
            
            # Convert to absolute times
            segment_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
            absolute_times = segment_times + start_time
            
            all_boundaries.extend(absolute_times)
            logger.info(f"  Found {len(peaks)} boundaries in this segment")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"  FFmpeg error processing segment: {e}")
        except Exception as e:
            logger.error(f"  Error processing segment: {e}")
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    # Sort and merge close boundaries
    all_boundaries = sorted(all_boundaries)
    merged_boundaries = merge_close_boundaries(all_boundaries, min_distance=2.0)
    
    return merged_boundaries

def merge_close_boundaries(boundaries, min_distance=2.0):
    """Merge boundaries that are too close together"""
    if len(boundaries) == 0:
        return []
    
    merged = [boundaries[0]]
    for b in boundaries[1:]:
        if b - merged[-1] >= min_distance:
            merged.append(b)
    
    return merged

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    start_time = time.time()
    
    # Check ffmpeg availability
    if not check_ffmpeg():
        logger.error("FFmpeg not found! Please install ffmpeg to use this script.")
        logger.error("On Ubuntu/Debian: sudo apt install ffmpeg")
        logger.error("On macOS: brew install ffmpeg")
        return
    
    # Process video
    video_path = 'vid.mp4'
    boundaries = process_video_segments_safe(video_path, segment_duration=300)  # 5-minute segments
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Total boundaries found: {len(boundaries)}")
    logger.info("Segment boundaries (seconds):")
    for i, t in enumerate(boundaries):
        logger.info(f"  {i+1}: {t:.2f}s ({t/60:.2f}m)")
    
    # Save results
    with open('segment_boundaries_fixed.txt', 'w') as f:
        f.write(f"Video segmentation results\n")
        f.write(f"Total boundaries: {len(boundaries)}\n\n")
        for i, t in enumerate(boundaries):
            f.write(f"Boundary {i+1}: {t:.2f}s ({t/60:.2f}m)\n")
    
    logger.info(f"\nResults saved to segment_boundaries_fixed.txt")
    
    # Create a simple visualization
    if len(boundaries) > 0:
        plt.figure(figsize=(15, 3))
        plt.vlines(boundaries, 0, 1, colors='red', linestyles='dashed')
        plt.xlim(0, max(boundaries) + 10)
        plt.xlabel('Time (seconds)')
        plt.title(f'Detected Segment Boundaries ({len(boundaries)} total)')
        plt.savefig('boundaries_fixed.png')
        plt.close()
        logger.info("Visualization saved to boundaries_fixed.png")
    
    total_time = time.time() - start_time
    logger.info(f"\nTotal processing time: {total_time:.2f}s ({total_time/60:.2f}m)")

if __name__ == "__main__":
    main()
