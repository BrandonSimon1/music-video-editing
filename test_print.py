#!/usr/bin/env python3
"""
Test script to diagnose why prints aren't showing
"""
import sys
import io
import moviepy as mp
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tempfile

def main():
    print("=== START OF SCRIPT ===", flush=True)
    
    # Test basic printing
    print("Test 1: Basic print", flush=True)
    sys.stdout.write("Test 2: sys.stdout.write\n")
    sys.stdout.flush()
    
    try:
        # Load video
        print("\nLoading video...", flush=True)
        video_path = 'vid.mp4'
        video_clip = mp.VideoFileClip(video_path)
        print(f"Video loaded: duration={video_clip.duration:.2f}s", flush=True)
        
        # Extract audio
        print("\nExtracting audio...", flush=True)
        audio_clip = video_clip.audio
        
        # Write audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_buffer:
            print(f"Writing audio to: {audio_buffer.name}", flush=True)
            audio_clip.write_audiofile(audio_buffer.name, fps=44100, codec='pcm_s16le', logger=None)
            
            # Load with librosa
            print("\nLoading audio with librosa...", flush=True)
            y, sr = librosa.load(audio_buffer.name, sr=None)
            print(f"Audio shape: {y.shape}, SR: {sr}", flush=True)
            
            # Compute features
            print("\nComputing chromagram...", flush=True)
            chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            print("Computing novelty...", flush=True)
            novelty = librosa.onset.onset_strength(sr=sr, S=chromagram)
            
            print("Detecting peaks...", flush=True)
            peaks = librosa.util.peak_pick(novelty, pre_max=10, post_max=10, 
                                         pre_avg=20, post_avg=20, delta=0.2, wait=10)
            
            # Convert to times
            seg_times = librosa.frames_to_time(peaks, sr=sr)
            
            # MULTIPLE WAYS TO OUTPUT RESULTS
            print("\n=== RESULTS ===", flush=True)
            print(f"Found {len(peaks)} segment boundaries", flush=True)
            print("Segment boundaries (seconds):", seg_times, flush=True)
            
            # Also write to stderr (always visible)
            print("\n--- Output to stderr ---", file=sys.stderr)
            print(f"Peaks found: {len(peaks)}", file=sys.stderr)
            print(f"Boundaries: {seg_times}", file=sys.stderr)
            
            # Save results to file
            with open('results.txt', 'w') as f:
                f.write(f"Segment boundaries found: {len(peaks)}\n")
                f.write(f"Times (seconds): {seg_times}\n")
                for i, t in enumerate(seg_times):
                    f.write(f"  Boundary {i+1}: {t:.2f}s\n")
            print("\nResults also saved to results.txt", flush=True)
            
            # Save plot without showing
            times = librosa.frames_to_time(np.arange(len(novelty)), sr=sr)
            plt.figure(figsize=(12, 4))
            plt.plot(times, novelty, label='Novelty Curve')
            plt.vlines(seg_times, 0, novelty.max(), color='r', linestyle='dashed', label='Segment Boundaries')
            plt.legend()
            plt.title('Novelty-based Segment Boundaries on Video Audio')
            plt.xlabel('Time (s)')
            plt.ylabel('Novelty')
            plt.savefig('novelty_plot.png')
            plt.close()
            print("Plot saved to novelty_plot.png", flush=True)
            
            # Clean up
            video_clip.close()
            import os
            os.unlink(audio_buffer.name)
            
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    
    print("\n=== END OF SCRIPT ===", flush=True)
    
    # Ensure all output is flushed
    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == "__main__":
    main()