"""
Recovery script: encode unencoded image frames into videos for a LeRobotDataset.

Run this from your lerobot repo root after a recording session was interrupted
before the final batch encoding could complete.

Usage:
    python encode_remaining_episodes.py \
        --dataset-path data/test1_dataset553 \
        --start-episode 16        # first episode that wasn't encoded
        --vcodec libsvtav1        # optional, matches what you used during recording
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def get_num_episodes(dataset_path: Path) -> int:
    """Read the total episode count from meta/info.json."""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    return info["total_episodes"]


def find_unencoded_start(dataset_path: Path, camera_keys: list[str]) -> int:
    """
    Auto-detect the first episode whose images haven't been encoded yet by
    checking which episode video files are missing in videos/chunk-000/.
    """
    videos_dir = dataset_path / "videos" / "chunk-000"
    num_episodes = get_num_episodes(dataset_path)

    for ep_idx in range(num_episodes):
        for cam_key in camera_keys:
            # LeRobot video naming: <cam_key>/episode_<06d>.mp4
            cam_folder = cam_key.replace(".", "/")
            video_file = videos_dir / cam_folder / f"episode_{ep_idx:06d}.mp4"
            if not video_file.exists():
                logging.info(f"First missing video: {video_file}")
                return ep_idx

    return num_episodes  # all encoded


def encode(dataset_path: str, start_episode: int | None, vcodec: str) -> None:
    # Import here so the script gives a clean error if lerobot isn't installed
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    root = Path(dataset_path)
    num_episodes = get_num_episodes(root)

    # Load in "create" / local mode so we can access the internal API.
    # `root` tells it to look at the local directory instead of the Hub.
    dataset = LeRobotDataset(
        repo_id=root.name,   # used as identifier; doesn't need to match Hub
        root=root,
    )

    if start_episode is None:
        # Try to auto-detect by inspecting which videos already exist
        cam_keys = [k for k in dataset.features if "image" in k or "observation" in k]
        if not cam_keys:
            # Fall back to listing the images/ subdirectories
            images_dir = root / "images"
            cam_keys = [p.name for p in images_dir.iterdir() if p.is_dir()]
        start_episode = find_unencoded_start(root, cam_keys)

    end_episode = num_episodes

    if start_episode >= end_episode:
        logging.info("No unencoded episodes found — nothing to do.")
        return

    logging.info(
        f"Encoding episodes {start_episode} to {end_episode - 1} "
        f"({end_episode - start_episode} episode(s)) with vcodec={vcodec}"
    )

    # Patch episodes_since_last_encoding so batch_encode_videos knows the range
    dataset.episodes_since_last_encoding = end_episode - start_episode

    dataset.batch_encode_videos(start_episode, end_episode)

    logging.info("Done! All episodes encoded.")
    logging.info(
        "You can now delete the images/ directory if you no longer need the raw PNGs:\n"
        f"  rm -rf {root / 'images'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode unencoded LeRobot image episodes to video.")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to the dataset root, e.g. data/test1_dataset553",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=None,
        help=(
            "Index of the first episode to encode (0-based). "
            "If omitted, the script auto-detects by checking which video files are missing."
        ),
    )
    parser.add_argument(
        "--vcodec",
        default="libsvtav1",
        help="FFmpeg video codec to use (default: libsvtav1). Use libx264 if svtav1 isn't available.",
    )
    args = parser.parse_args()

    encode(
        dataset_path=args.dataset_path,
        start_episode=args.start_episode,
        vcodec=args.vcodec,
    )