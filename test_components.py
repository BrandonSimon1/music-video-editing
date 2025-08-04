#!/usr/bin/env python3
"""
Test script to isolate potential issues by testing components individually
"""
import time
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required libraries can be imported"""
    logger.info("Testing imports...")
    try:
        import moviepy
        logger.info(f"✓ moviepy version: {moviepy.__version__}")
    except Exception as e:
        logger.error(f"✗ Failed to import moviepy: {e}")
        return False
    
    try:
        import librosa
        logger.info(f"✓ librosa version: {librosa.__version__}")
    except Exception as e:
        logger.error(f"✗ Failed to import librosa: {e}")
        return False
    
    try:
        import numpy as np
        logger.info(f"✓ numpy version: {np.__version__}")
    except Exception as e:
        logger.error(f"✗ Failed to import numpy: {e}")
        return False
    
    try:
        import matplotlib
        logger.info(f"✓ matplotlib version: {matplotlib.__version__}")
    except Exception as e:
        logger.error(f"✗ Failed to import matplotlib: {e}")
        return False
    
    try:
        import soundfile
        logger.info(f"✓ soundfile version: {soundfile.__version__}")
    except Exception as e:
        logger.error(f"✗ Failed to import soundfile: {e}")
        return False
    
    try:
        import psutil
        logger.info(f"✓ psutil version: {psutil.__version__}")
    except Exception as e:
        logger.error(f"✗ Failed to import psutil: {e}")
        return False
    
    return True

def test_video_file():
    """Test video file existence and basic properties"""
    logger.info("\nTesting video file...")
    video_path = 'vid.mp4'
    
    if not os.path.exists(video_path):
        logger.error(f"✗ Video file not found: {video_path}")
        return False
    
    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    logger.info(f"✓ Video file exists: {size_mb:.2f} MB")
    
    return True

def test_video_load():
    """Test loading video with MoviePy"""
    logger.info("\nTesting video loading...")
    try:
        import moviepy.editor as mp
        video = mp.VideoFileClip('vid.mp4')
        logger.info(f"✓ Video loaded: duration={video.duration:.2f}s, fps={video.fps}")
        
        # Test audio extraction
        if video.audio is not None:
            logger.info(f"✓ Audio track found: duration={video.audio.duration:.2f}s")
        else:
            logger.error("✗ No audio track found")
            
        video.close()
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load video: {e}")
        return False

def test_audio_extraction(duration_limit=30):
    """Test audio extraction with limited duration"""
    logger.info(f"\nTesting audio extraction (first {duration_limit}s)...")
    try:
        import moviepy.editor as mp
        import tempfile
        
        # Load only first N seconds
        video = mp.VideoFileClip('vid.mp4').subclipfx(t_end=duration_limit)
        logger.info(f"✓ Loaded {duration_limit}s clip")
        
        if video.audio is None:
            logger.error("✗ No audio track")
            return False
        
        # Extract and save audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_path = tmp.name
        
        start = time.time()
        video.audio.write_audiofile(temp_path, logger=None)
        duration = time.time() - start
        
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        logger.info(f"✓ Audio extracted in {duration:.2f}s: {size_mb:.2f} MB")
        
        # Test librosa load
        import librosa
        y, sr = librosa.load(temp_path, sr=None)
        logger.info(f"✓ Librosa loaded audio: shape={y.shape}, sr={sr}")
        
        # Cleanup
        os.unlink(temp_path)
        video.close()
        
        return True
    except Exception as e:
        logger.error(f"✗ Audio extraction failed: {e}")
        return False

def test_memory_limits():
    """Check system memory"""
    logger.info("\nChecking system resources...")
    try:
        import psutil
        
        # System memory
        mem = psutil.virtual_memory()
        logger.info(f"Total RAM: {mem.total / (1024**3):.2f} GB")
        logger.info(f"Available RAM: {mem.available / (1024**3):.2f} GB")
        logger.info(f"Used RAM: {mem.percent:.1f}%")
        
        # Current process
        process = psutil.Process()
        mem_mb = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Current process memory: {mem_mb:.2f} MB")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to check memory: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting component tests...")
    
    tests = [
        ("Imports", test_imports),
        ("Video File", test_video_file),
        ("Memory Check", test_memory_limits),
        ("Video Load", test_video_load),
        ("Audio Extraction (30s)", test_audio_extraction),
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"\n{'='*50}")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY:")
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"  {name}: {status}")
    
    # Suggestions
    if not all(r[1] for r in results):
        logger.info("\nSUGGESTIONS:")
        logger.info("1. Try processing a shorter video clip first")
        logger.info("2. Monitor system resources during execution")
        logger.info("3. Use the debug script (main_debug.py) for detailed logging")
        logger.info("4. Consider processing audio in chunks for large files")

if __name__ == "__main__":
    main()