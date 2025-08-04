import time
import logging
import os
import subprocess
import moviepy as mp
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks

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

def compute_self_similarity_matrix(features, metric='cosine'):
    """
    Compute self-similarity matrix from features
    This helps identify repeating sections in music
    """
    # Normalize features
    features_norm = features / (np.linalg.norm(features, axis=0) + 1e-6)
    
    # Compute pairwise distances
    distance_matrix = cdist(features_norm.T, features_norm.T, metric=metric)
    
    # Convert to similarity (1 - distance for cosine)
    similarity_matrix = 1 - distance_matrix
    
    return similarity_matrix

def find_musical_sections(y, sr, hop_length=512):
    """
    Find complete musical sections suitable for looping
    """
    logger.info("  Analyzing musical structure for sections...")
    
    # 1. Beat tracking for tempo and downbeats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
    
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if tempo.size > 0 else 120.0
    else:
        tempo = float(tempo) if tempo is not None else 120.0
    
    logger.info(f"    Tempo: {tempo:.1f} BPM")
    
    # 2. Compute multiple features for section detection
    # Chroma for harmonic content
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    
    # MFCC for timbral content
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    
    # Tonnetz for harmonic relationships
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr, hop_length=hop_length)
    
    # 3. Stack features and compute self-similarity
    features = np.vstack([chroma, mfcc[:, :chroma.shape[1]], tonnetz[:, :chroma.shape[1]]])
    ssm = compute_self_similarity_matrix(features)
    
    # 4. Enhance diagonal structures (repeating sections)
    # Apply Gaussian checkerboard kernel for better section detection
    from scipy.ndimage import gaussian_filter
    ssm_enhanced = gaussian_filter(ssm, sigma=2)
    
    # 5. Detect section boundaries using novelty from SSM
    # Compute checkerboard kernel convolution
    kernel_size = 32  # Adjust based on expected section length
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[:kernel_size//2, :kernel_size//2] = 1
    kernel[kernel_size//2:, kernel_size//2:] = 1
    kernel[:kernel_size//2, kernel_size//2:] = -1
    kernel[kernel_size//2:, :kernel_size//2] = -1
    
    # Convolve along diagonal
    novelty = np.zeros(ssm.shape[0])
    for i in range(kernel_size//2, ssm.shape[0] - kernel_size//2):
        submatrix = ssm_enhanced[i-kernel_size//2:i+kernel_size//2, i-kernel_size//2:i+kernel_size//2]
        novelty[i] = np.sum(submatrix * kernel)
    
    # Smooth and normalize novelty
    novelty = gaussian_filter(novelty, sigma=4)
    novelty = novelty / (np.max(np.abs(novelty)) + 1e-6)
    
    # 6. Find peaks in novelty (section boundaries)
    peaks, properties = find_peaks(novelty, 
                                  height=np.percentile(novelty, 75),
                                  distance=int(sr * 2 / hop_length))  # Min 2 seconds between sections
    
    # Convert to time
    boundary_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    
    # 7. Snap boundaries to nearest downbeat for better loops
    if len(beat_times) > 0 and len(boundary_times) > 0:
        # Find downbeats (every 4 beats assuming 4/4 time)
        bars_per_second = tempo / 60 / 4  # bars per second
        bar_duration = 4 * 60 / tempo  # seconds per bar
        
        # Estimate downbeats
        downbeat_times = []
        if len(beat_times) > 4:
            for i in range(0, len(beat_times), 4):
                downbeat_times.append(beat_times[i])
        
        if len(downbeat_times) > 0:
            # Snap section boundaries to nearest downbeat
            snapped_boundaries = []
            for boundary in boundary_times:
                nearest_idx = np.argmin(np.abs(np.array(downbeat_times) - boundary))
                snapped_boundaries.append(downbeat_times[nearest_idx])
            boundary_times = np.array(snapped_boundaries)
    
    # 8. Create sections from boundaries
    sections = []
    all_times = np.concatenate([[0], boundary_times, [len(y) / sr]])
    all_times = np.unique(np.sort(all_times))
    
    for i in range(len(all_times) - 1):
        start = all_times[i]
        end = all_times[i + 1]
        duration = end - start
        
        # Filter sections by duration (good loops are typically 4-32 bars)
        min_duration = 4 * 60 / tempo * 2  # At least 2 bars
        max_duration = 4 * 60 / tempo * 16  # At most 16 bars
        
        if min_duration <= duration <= max_duration:
            # Check if duration is close to a multiple of bars
            bars = duration / (4 * 60 / tempo)
            if np.abs(bars - np.round(bars)) < 0.1:  # Within 10% of exact bars
                sections.append({
                    'start': start,
                    'end': end,
                    'duration': duration,
                    'bars': int(np.round(bars))
                })
    
    return sections, ssm, novelty, tempo

def process_video_for_loops(video_path, segment_duration=300, 
                           min_section_bars=2, max_section_bars=16):
    """
    Process video to find loopable sections
    """
    logger.info(f"Processing video for loops: {video_path}")
    
    # Get video duration
    with mp.VideoFileClip(video_path) as video_clip:
        total_duration = video_clip.duration
        logger.info(f"Video duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
        
        if video_clip.audio is None:
            logger.error("No audio track found")
            return []
    
    all_sections = []
    hop_length = 512
    
    # Process in segments
    num_segments = int(np.ceil(total_duration / segment_duration))
    logger.info(f"Processing in {num_segments} segments of {segment_duration}s each")
    
    # Limit for testing
    num_segments = min(num_segments, 3)
    
    for seg_idx in range(num_segments):
        start_time = seg_idx * segment_duration
        duration = min(segment_duration, total_duration - start_time)
        end_time = start_time + duration
        
        logger.info(f"\nProcessing segment {seg_idx+1}/{num_segments}: {start_time:.1f}s - {end_time:.1f}s")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Extract audio segment
            extract_audio_segment_ffmpeg(video_path, start_time, duration, temp_path)
            
            # Load with librosa
            y, sr = librosa.load(temp_path, sr=None)
            logger.info(f"  Audio loaded: {len(y)/sr:.1f}s at {sr}Hz")
            
            # Find musical sections
            sections, ssm, novelty, tempo = find_musical_sections(y, sr, hop_length)
            
            # Adjust section times to absolute video time
            for section in sections:
                section['start'] += start_time
                section['end'] += start_time
                all_sections.append(section)
            
            logger.info(f"  Found {len(sections)} loopable sections")
            for i, section in enumerate(sections[:5]):  # Show first 5
                logger.info(f"    Section {i+1}: {section['start']:.2f}s - {section['end']:.2f}s ({section['bars']} bars)")
            
            # Save debug visualizations for first segment
            if seg_idx == 0 and len(sections) > 0:
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
                
                # Self-similarity matrix
                ax1.imshow(ssm, cmap='hot', aspect='auto', origin='lower')
                ax1.set_title('Self-Similarity Matrix')
                ax1.set_xlabel('Time (frames)')
                ax1.set_ylabel('Time (frames)')
                
                # Novelty curve
                frames = np.arange(len(novelty))
                time_axis = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
                ax2.plot(time_axis, novelty)
                for section in sections:
                    ax2.axvline(x=section['start'] - start_time, color='r', linestyle='--', alpha=0.7)
                    ax2.axvline(x=section['end'] - start_time, color='r', linestyle='--', alpha=0.7)
                ax2.set_title('Section Boundary Detection')
                ax2.set_xlabel('Time (seconds)')
                ax2.set_ylabel('Novelty')
                
                # Detected sections
                ax3.barh(range(len(sections)), 
                        [s['duration'] for s in sections],
                        left=[s['start'] - start_time for s in sections],
                        height=0.8)
                ax3.set_title('Detected Loopable Sections')
                ax3.set_xlabel('Time (seconds)')
                ax3.set_ylabel('Section')
                ax3.set_xlim(0, duration)
                
                plt.tight_layout()
                plt.savefig('loop_analysis.png')
                plt.close()
                logger.info("  Debug visualization saved to loop_analysis.png")
            
        except Exception as e:
            logger.error(f"  Error processing segment: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    # Sort sections by start time
    all_sections.sort(key=lambda x: x['start'])
    
    # Merge or filter overlapping sections
    filtered_sections = []
    for section in all_sections:
        if not filtered_sections or section['start'] >= filtered_sections[-1]['end']:
            filtered_sections.append(section)
    
    return filtered_sections

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
    
    # Process video for loops
    video_path = 'vid.mp4'
    
    sections = process_video_for_loops(
        video_path, 
        segment_duration=300,
        min_section_bars=2,
        max_section_bars=16
    )
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Total loopable sections found: {len(sections)}")
    logger.info("\nLoopable sections:")
    
    # Save results
    with open('loopable_sections.txt', 'w') as f:
        f.write(f"Loopable sections in video\n")
        f.write(f"Total sections: {len(sections)}\n\n")
        
        for i, section in enumerate(sections):
            start_min = section['start'] / 60
            end_min = section['end'] / 60
            logger.info(f"  Section {i+1}: {section['start']:.2f}s - {section['end']:.2f}s ({section['bars']} bars, {section['duration']:.2f}s)")
            f.write(f"Section {i+1}:\n")
            f.write(f"  Start: {section['start']:.2f}s ({start_min:.2f}m)\n")
            f.write(f"  End: {section['end']:.2f}s ({end_min:.2f}m)\n")
            f.write(f"  Duration: {section['duration']:.2f}s\n")
            f.write(f"  Bars: {section['bars']}\n")
            f.write(f"  FFmpeg command: ffmpeg -ss {section['start']:.2f} -t {section['duration']:.2f} -i {video_path} -c copy loop_{i+1}.mp4\n\n")
    
    logger.info(f"\nResults saved to loopable_sections.txt")
    
    # Create visualization
    if len(sections) > 0:
        plt.figure(figsize=(15, 6))
        for i, section in enumerate(sections[:50]):  # First 50 sections
            plt.barh(0, section['duration'], left=section['start'], height=0.5, 
                    alpha=0.7, label=f"{section['bars']} bars" if i < 5 else "")
        
        plt.xlim(0, min(max([s['end'] for s in sections]), 1800))  # First 30 minutes max
        plt.xlabel('Time (seconds)')
        plt.title(f'Loopable Sections ({len(sections)} total)')
        plt.yticks([])
        if len(sections) > 0:
            plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('loopable_sections.png')
        plt.close()
        logger.info("Visualization saved to loopable_sections.png")
    
    total_time = time.time() - start_time
    logger.info(f"\nTotal processing time: {total_time:.2f}s ({total_time/60:.2f}m)")
    
    # Show example commands
    if len(sections) > 0:
        logger.info("\nExample loop extraction commands:")
        for i in range(min(3, len(sections))):
            section = sections[i]
            logger.info(f"  ffmpeg -ss {section['start']:.2f} -t {section['duration']:.2f} -i {video_path} -c copy loop_{i+1}.mp4")

if __name__ == "__main__":
    main()
