import time
import logging
import psutil
import os
import sys
import traceback
from datetime import datetime
import moviepy as mp
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def log_memory_usage(checkpoint=""):
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory usage {checkpoint}: RSS={mem_info.rss/1024/1024:.2f}MB, VMS={mem_info.vms/1024/1024:.2f}MB")

def get_file_info(file_path):
    """Get file size and other info"""
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        return size_mb
    return None

def main():
    start_time = time.time()
    logger.info("Starting music-video-editing script")
    log_memory_usage("at start")
    
    try:
        # Check video file
        video_path = 'vid.mp4'
        logger.info(f"Checking video file: {video_path}")
        
        file_size = get_file_info(video_path)
        if file_size:
            logger.info(f"Video file size: {file_size:.2f} MB")
        else:
            logger.error(f"Video file not found: {video_path}")
            return
        
        # Load video with progress tracking
        logger.info("Loading video file with MoviePy...")
        log_memory_usage("before video load")
        
        video_clip = mp.VideoFileClip(video_path)
        
        log_memory_usage("after video load")
        logger.info(f"Video loaded successfully: duration={video_clip.duration:.2f}s, fps={video_clip.fps}")
        
        # Extract audio
        logger.info("Extracting audio from video...")
        audio_clip = video_clip.audio
        
        if audio_clip is None:
            logger.error("No audio track found in video")
            return
        
        logger.info(f"Audio extracted: duration={audio_clip.duration:.2f}s")
        log_memory_usage("after audio extraction")
        
        # Export audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_buffer:
            temp_audio_path = audio_buffer.name
            
        try:
            logger.info(f"Writing audio to temporary file: {temp_audio_path}")
            write_start = time.time()
            
            # Write with progress callback
            audio_clip.write_audiofile(
                temp_audio_path, 
                fps=44100, 
                codec='pcm_s16le', 
                logger=None
            )
            
            write_duration = time.time() - write_start
            temp_size = get_file_info(temp_audio_path)
            logger.info(f"Audio written successfully in {write_duration:.2f}s, size: {temp_size:.2f} MB")
            log_memory_usage("after audio write")
            
            # Load audio with librosa
            logger.info("Loading audio with librosa...")
            load_start = time.time()
            
            y, sr = librosa.load(temp_audio_path, sr=None)
            
            load_duration = time.time() - load_start
            logger.info(f"Audio loaded by librosa in {load_duration:.2f}s: shape={y.shape}, sr={sr}")
            log_memory_usage("after librosa load")
            
            # Compute chromagram
            logger.info("Computing chromagram...")
            chroma_start = time.time()
            
            chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            chroma_duration = time.time() - chroma_start
            logger.info(f"Chromagram computed in {chroma_duration:.2f}s: shape={chromagram.shape}")
            log_memory_usage("after chromagram")
            
            # Compute novelty function
            logger.info("Computing novelty function...")
            novelty_start = time.time()
            
            novelty = librosa.onset.onset_strength(sr=sr, S=chromagram)
            
            novelty_duration = time.time() - novelty_start
            logger.info(f"Novelty computed in {novelty_duration:.2f}s: shape={novelty.shape}")
            log_memory_usage("after novelty")
            
            # Detect peaks
            logger.info("Detecting peaks in novelty function...")
            peaks_start = time.time()
            
            peaks = librosa.util.peak_pick(
                novelty, 
                pre_max=10, 
                post_max=10, 
                pre_avg=20, 
                post_avg=20, 
                delta=0.2, 
                wait=10
            )
            
            peaks_duration = time.time() - peaks_start
            logger.info(f"Peaks detected in {peaks_duration:.2f}s: found {len(peaks)} peaks")
            
            # Convert to times
            logger.info("Converting frames to time...")
            times = librosa.frames_to_time(np.arange(len(novelty)), sr=sr)
            seg_times = librosa.frames_to_time(peaks, sr=sr)
            
            logger.info(f"Segment boundaries (seconds): {seg_times}")
            
            # Plot results
            logger.info("Creating visualization...")
            plt.figure(figsize=(12, 4))
            plt.plot(times, novelty, label='Novelty Curve')
            plt.vlines(seg_times, 0, novelty.max(), color='r', linestyle='dashed', label='Segment Boundaries')
            plt.legend()
            plt.title('Novelty-based Segment Boundaries on Video Audio')
            plt.xlabel('Time (s)')
            plt.ylabel('Novelty')
            
            # Save plot instead of showing (non-blocking)
            plot_path = 'novelty_plot.png'
            plt.savefig(plot_path)
            logger.info(f"Plot saved to {plot_path}")
            plt.close()
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                logger.info("Temporary audio file deleted")
            
            # Clean up video clip
            video_clip.close()
            logger.info("Video clip closed")
        
        total_duration = time.time() - start_time
        logger.info(f"Script completed successfully in {total_duration:.2f}s")
        log_memory_usage("at end")
        
    except Exception as e:
        logger.error(f"Error occurred: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        log_memory_usage("at error")
        raise

if __name__ == "__main__":
    main()