import moviepy as mp

# Quick test to check MoviePy API
video = mp.VideoFileClip('vid.mp4')

print("Available methods on VideoFileClip:")
methods = [m for m in dir(video) if not m.startswith('_')]
subclip_methods = [m for m in methods if 'sub' in m.lower() or 'clip' in m.lower()]
print("\nMethods with 'sub' or 'clip':")
for m in sorted(subclip_methods):
    print(f"  - {m}")

# Check if it's a function property
if hasattr(video, 'subclip'):
    print(f"\nsubclip exists: {type(video.subclip)}")
if hasattr(video, 'subclipped'):
    print(f"subclipped exists: {type(video.subclipped)}")

video.close()