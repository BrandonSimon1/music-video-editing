# Algorithm plugins for process_video.py.
#
# Each algorithm is a module in this package that exposes:
#
#   ALGORITHM_NAME: str
#     Short identifier used in folder names and clips.json ("allin1", "librosa", …).
#
#   ALGORITHM_VERSION: str
#     Model/version string for provenance ("harmonix-all", "0.10.1", …).
#
#   def add_args(parser: argparse.ArgumentParser) -> None:
#     Register algorithm-specific CLI flags on the parser.
#     Use a dedicated argument group to avoid name collisions.
#
#   def run(video_file: Path, cache_dir: Path, args: argparse.Namespace) -> list[dict]:
#     Run the algorithm and return a list of clip dicts.
#     Required keys per clip: start_time (float), end_time (float), duration (float).
#     Optional keys (used by downstream filters): beat_density, beat_regularity_cv,
#     num_measures, segment_labels, beats.
#     cache_dir is <video_folder>/analysis/ — use it for expensive intermediates,
#     or ignore it if the algorithm has no cacheable analysis step.
