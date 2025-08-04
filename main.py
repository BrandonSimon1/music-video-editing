def main():
    print("Hello from music-video-editing!")
    import moviepy as mp
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    import io
    import soundfile as sf
    import tempfile

    # Load your video file (replace 'your_video.mp4' with your file)
    video_path = 'vid.mp4'
    video_clip = mp.VideoFileClip(video_path)

    # Extract audio as a numpy array and sampling rate
    audio_clip = video_clip.audio
    # Export audio to a bytes buffer (wav format) in-memory
    with tempfile.NamedTemporaryFile(suffix='.wav') as audio_buffer:
        audio_clip.write_audiofile(audio_buffer.name, fps=44100, codec='pcm_s16le', logger=None)

        # Load the audio from the buffer into librosa
        y, sr = librosa.load(audio_buffer.name, sr=None)  # Use native sampling rate of extracted audio

        # Compute chromagram
        chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)

        # Compute novelty function
        novelty = librosa.onset.onset_strength(sr=sr, S=chromagram)

        # Detect peaks in the novelty function
        peaks = librosa.util.peak_pick(novelty, pre_max=10, post_max=10, pre_avg=20, post_avg=20, delta=0.2, wait=10)

        # Convert frame indices to time in seconds
        times = librosa.frames_to_time(np.arange(len(novelty)), sr=sr)
        seg_times = librosa.frames_to_time(peaks, sr=sr)

        # Plot the novelty curve and detected segment boundaries
        plt.figure(figsize=(12, 4))
        plt.plot(times, novelty, label='Novelty Curve')
        plt.vlines(seg_times, 0, novelty.max(), color='r', linestyle='dashed', label='Segment Boundaries')
        plt.legend()
        plt.title('Novelty-based Segment Boundaries on Video Audio')
        plt.xlabel('Time (s)')
        plt.ylabel('Novelty')
        plt.show()

        # Print segment boundaries relative to the video timeline
        print("Segment boundaries (seconds):", seg_times)
        print(f"Found {len(peaks)} segment boundaries")
        
        # Also save results to file
        with open('segment_boundaries.txt', 'w') as f:
            f.write(f"Found {len(peaks)} segment boundaries\n")
            f.write(f"Times (seconds): {list(seg_times)}\n")
            for i, t in enumerate(seg_times):
                f.write(f"Boundary {i+1}: {t:.2f}s\n")
        print("Results saved to segment_boundaries.txt")



if __name__ == "__main__":
    main()
