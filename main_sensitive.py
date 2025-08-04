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
    """Extract audio segment using ffmpeg directly"""
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
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def analyze_music_structure(y, sr, hop_length=512):
    """
    Perform multiple analyses to detect interesting musical moments
    """
    logger.info("  Analyzing musical structure...")
    
    # 1. Beat tracking for rhythm changes
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    # Handle tempo as numpy array or scalar
    if isinstance(tempo, np.ndarray):
        tempo_value = float(tempo[0]) if tempo.size > 0 else 0.0
    else:
        tempo_value = float(tempo)
    logger.info(f"    Tempo: {tempo_value:.1f} BPM")
    
    # 2. Spectral features for timbral changes (drops, builds)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    
    # 3. RMS energy for volume/intensity changes
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    
    # 4. Onset detection for percussive events
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    # 5. Chroma for harmonic changes
    chromagram = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    
    # 6. Tempogram for tempo variations
    tempogram = librosa.feature.tempogram(y=y, sr=sr, hop_length=hop_length)
    
    # Combine multiple novelty functions
    novelty_scores = []
    
    # Spectral novelty (for drops/builds)
    spectral_novelty = np.abs(np.diff(spectral_centroids, prepend=spectral_centroids[0]))
    spectral_novelty = spectral_novelty / np.max(spectral_novelty) if np.max(spectral_novelty) > 0 else spectral_novelty
    novelty_scores.append(('spectral', spectral_novelty, 0.3))
    
    # Energy novelty (for intensity changes)
    rms_novelty = np.abs(np.diff(rms, prepend=rms[0]))
    rms_novelty = rms_novelty / np.max(rms_novelty) if np.max(rms_novelty) > 0 else rms_novelty
    novelty_scores.append(('energy', rms_novelty, 0.3))
    
    # Onset strength (for rhythmic changes)
    onset_novelty = onset_env / np.max(onset_env) if np.max(onset_env) > 0 else onset_env
    novelty_scores.append(('onset', onset_novelty, 0.2))
    
    # Harmonic novelty
    chroma_novelty = np.sum(np.abs(np.diff(chromagram, axis=1)), axis=0)
    chroma_novelty = chroma_novelty / np.max(chroma_novelty) if np.max(chroma_novelty) > 0 else chroma_novelty
    novelty_scores.append(('harmonic', chroma_novelty, 0.2))
    
    return novelty_scores, beats, tempo

def detect_interesting_moments(novelty_scores, sr, hop_length, 
                             sensitivity=0.5, min_distance=1.0, max_segments=30):
    """
    Detect interesting moments with adjustable sensitivity
    
    Args:
        sensitivity: 0.0 (least sensitive) to 1.0 (most sensitive)
        min_distance: Minimum time between segments in seconds
        max_segments: Maximum number of segments to return per chunk
    """
    # Combine novelty scores with weights
    combined_novelty = np.zeros_like(novelty_scores[0][1])
    for name, score, weight in novelty_scores:
        # Ensure arrays are same length
        if len(score) < len(combined_novelty):
            score = np.pad(score, (0, len(combined_novelty) - len(score)), mode='constant')
        elif len(score) > len(combined_novelty):
            score = score[:len(combined_novelty)]
        combined_novelty += score * weight
    
    # Apply smoothing to reduce noise
    from scipy.ndimage import gaussian_filter1d
    combined_novelty = gaussian_filter1d(combined_novelty, sigma=2)
    
    # Dynamic threshold based on sensitivity
    # Higher sensitivity = lower threshold = more boundaries
    threshold_percentile = 95 - (sensitivity * 45)  # Range: 95th to 50th percentile
    threshold = np.percentile(combined_novelty, threshold_percentile)
    
    # Peak picking with adjusted parameters
    delta = threshold * 0.3  # Minimum peak prominence
    pre_max = max(3, int(10 * (1 - sensitivity)))  # Fewer samples for higher sensitivity
    post_max = pre_max
    wait = int(min_distance * sr / hop_length)  # Convert min_distance to frames
    
    peaks = librosa.util.peak_pick(
        combined_novelty,
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_max * 2,
        post_avg=post_max * 2,
        delta=delta,
        wait=wait
    )
    
    # Sort by novelty strength and limit number
    if len(peaks) > max_segments:
        peak_strengths = [(p, combined_novelty[p]) for p in peaks]
        peak_strengths.sort(key=lambda x: x[1], reverse=True)
        peaks = sorted([p[0] for p in peak_strengths[:max_segments]])
    
    return peaks, combined_novelty

def process_video_segments_sensitive(video_path, segment_duration=300, 
                                   sensitivity=0.7, min_clip_distance=1.5,
                                   max_clips_per_segment=20):
    """
    Process video to find interesting musical moments
    
    Args:
        video_path: Path to video file
        segment_duration: Duration of each processing segment in seconds
        sensitivity: Detection sensitivity (0.0 to 1.0, higher = more clips)
        min_clip_distance: Minimum time between clips in seconds
        max_clips_per_segment: Maximum clips to detect per segment
    """
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Settings: sensitivity={sensitivity}, min_distance={min_clip_distance}s")
    
    # Get video duration
    with mp.VideoFileClip(video_path) as video_clip:
        total_duration = video_clip.duration
        logger.info(f"Video duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
        
        if video_clip.audio is None:
            logger.error("No audio track found")
            return []
    
    all_boundaries = []
    hop_length = 512
    
    # Process in segments
    num_segments = int(np.ceil(total_duration / segment_duration))
    logger.info(f"Processing in {num_segments} segments of {segment_duration}s each")
    
    # Limit for testing
    num_segments = min(num_segments, 5)
    
    for i in range(num_segments):
        start_time = i * segment_duration
        duration = min(segment_duration, total_duration - start_time)
        end_time = start_time + duration
        
        logger.info(f"\nProcessing segment {i+1}/{num_segments}: {start_time:.1f}s - {end_time:.1f}s")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Extract audio segment
            extract_audio_segment_ffmpeg(video_path, start_time, duration, temp_path)
            
            # Load with librosa
            y, sr = librosa.load(temp_path, sr=None)
            logger.info(f"  Audio loaded: {len(y)/sr:.1f}s at {sr}Hz")
            
            # Analyze musical structure
            novelty_scores, beats, tempo = analyze_music_structure(y, sr, hop_length)
            
            # Detect interesting moments
            peaks, combined_novelty = detect_interesting_moments(
                novelty_scores, sr, hop_length,
                sensitivity=sensitivity,
                min_distance=min_clip_distance,
                max_segments=max_clips_per_segment
            )
            
            # Convert to absolute times
            segment_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
            absolute_times = segment_times + start_time
            
            # Add beat-aligned boundaries for better musical timing
            if len(beats) > 0 and len(peaks) > 0:
                beat_times = librosa.frames_to_time(beats, sr=sr)
                aligned_times = []
                for t in segment_times:
                    # Snap to nearest beat
                    nearest_beat_idx = np.argmin(np.abs(beat_times - t))
                    aligned_times.append(beat_times[nearest_beat_idx] + start_time)
                absolute_times = np.array(aligned_times)
            
            all_boundaries.extend(absolute_times)
            logger.info(f"  Found {len(peaks)} interesting moments in this segment")
            
        except Exception as e:
            logger.error(f"  Error processing segment: {e}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    # Sort and merge close boundaries
    all_boundaries = sorted(all_boundaries)
    merged_boundaries = merge_close_boundaries(all_boundaries, min_distance=min_clip_distance)
    
    return merged_boundaries

def merge_close_boundaries(boundaries, min_distance=1.5):
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
    
    if not check_ffmpeg():
        logger.error("FFmpeg not found! Please install ffmpeg.")
        return
    
    # Process video with high sensitivity for interesting moments
    video_path = 'vid.mp4'
    
    # Adjust these parameters to control detection
    SENSITIVITY = 0.7  # 0.0-1.0, higher = more clips detected
    MIN_CLIP_DISTANCE = 2.0  # Minimum seconds between clips
    MAX_CLIPS_PER_SEGMENT = 30  # Maximum clips per 5-minute segment
    
    boundaries = process_video_segments_sensitive(
        video_path, 
        segment_duration=300,
        sensitivity=SENSITIVITY,
        min_clip_distance=MIN_CLIP_DISTANCE,
        max_clips_per_segment=MAX_CLIPS_PER_SEGMENT
    )
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Total interesting moments found: {len(boundaries)}")
    logger.info("Clip boundaries (seconds):")
    for i, t in enumerate(boundaries[:20]):  # Show first 20
        logger.info(f"  {i+1}: {t:.2f}s ({t/60:.2f}m)")
    if len(boundaries) > 20:
        logger.info(f"  ... and {len(boundaries) - 20} more")
    
    # Save results with suggested clip durations
    with open('interesting_moments.txt', 'w') as f:
        f.write(f"Interesting moments in video\n")
        f.write(f"Sensitivity: {SENSITIVITY}\n")
        f.write(f"Total moments: {len(boundaries)}\n\n")
        
        # Suggest clip durations (2-5 seconds after each boundary)
        default_clip_duration = 3.0
        for i, t in enumerate(boundaries):
            end_time = t + default_clip_duration
            f.write(f"Clip {i+1}: {t:.2f}s - {end_time:.2f}s (duration: {default_clip_duration}s)\n")
    
    logger.info(f"\nResults saved to interesting_moments.txt")
    
    # Create visualization
    if len(boundaries) > 0:
        plt.figure(figsize=(15, 4))
        plt.vlines(boundaries, 0, 1, colors='red', linestyles='dashed', alpha=0.7)
        plt.xlim(0, min(max(boundaries) + 10, 1800))  # Show first 30 minutes max
        plt.xlabel('Time (seconds)')
        plt.title(f'Detected Interesting Moments ({len(boundaries)} total, sensitivity={SENSITIVITY})')
        plt.savefig('interesting_moments.png')
        plt.close()
        logger.info("Visualization saved to interesting_moments.png")
    
    total_time = time.time() - start_time
    logger.info(f"\nTotal processing time: {total_time:.2f}s ({total_time/60:.2f}m)")
    
    # Show example ffmpeg commands for extracting clips
    if len(boundaries) > 0:
        logger.info("\nExample commands to extract clips:")
        for i in range(min(3, len(boundaries))):
            start = boundaries[i]
            duration = 3.0
            logger.info(f"  ffmpeg -ss {start:.2f} -t {duration} -i {video_path} -c copy clip_{i+1}.mp4")

if __name__ == "__main__":
    main()