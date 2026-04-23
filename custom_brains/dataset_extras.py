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
import shutil
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

def merge_datasets(out_dir, *dataset_dirs):
    """
    Used to combine two identically formatted datasets, likely because a recording session was interrupted. 
    Kind of a force. Don't use without editing.
    """
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    # merged metadata
    merged_episodes = []
    merged_stats = []
    merged_tasks = {}
    
    global_episode_index_episodes = 0
    global_episode_index_stats = 0
    video_onboard_index = 0
    video_behind_index = 0
    global_episode_index_data = 0
    global_task_index = 0

    (onboard_dir := out_dir / "videos/chunk-000/observation.images.onboard").mkdir(parents=True, exist_ok=True)
    (behind_dir := out_dir / "videos/chunk-000/observation.images.behind").mkdir(parents=True, exist_ok=True)
    chunk_data_dir = out_dir / "data" / "chunk-000"
    chunk_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting merge of {len(dataset_dirs)} datasets into {out_dir}\n")

    for d_idx, d in enumerate(dataset_dirs):
        d = Path(d)
        print(f"Processing dataset {d_idx + 1}/{len(dataset_dirs)}: {d}")
        data_chunk = d / "data/chunk-000"
        video_chunk = d / "videos/chunk-000"
        
        tasks_file = d / "meta/tasks.jsonl"
        with open(tasks_file) as f:
            tasks = [json.loads(l) for l in f]
        print(f"  Found {len(tasks)} tasks in this dataset")

        for episode_file in sorted(data_chunk.glob("episode_*.parquet")):
            new_data_name = f"episode_{global_episode_index_data:06d}.parquet"
            shutil.copy(episode_file, out_dir / "data/chunk-000" / new_data_name)
            global_episode_index_data += 1

        src_onboard_dir = video_chunk / "observation.images.onboard"
        for video_file in sorted(src_onboard_dir.glob("episode_*.mp4")):
            new_onboard_name = f"episode_{video_onboard_index:06d}.mp4"
            print("  Copying video to onboard_dir at", onboard_dir / new_onboard_name)
            shutil.copy(video_file, onboard_dir / new_onboard_name)
            video_onboard_index += 1

        src_behind_dir = video_chunk / "observation.images.behind"
        for video_file in sorted(src_behind_dir.glob("episode_*.mp4")):
            new_behind_name = f"episode_{video_behind_index:06d}.mp4"
            print("  Copying video to behind_dir at", behind_dir / new_behind_name)
            shutil.copy(video_file, behind_dir / new_behind_name)
            video_behind_index += 1

        task_index_local_to_global = {}
        with open(d / "meta" / "tasks.jsonl") as f:
            print("  Opening tasks")
            for line in f:
                task_entry = json.loads(line)
                task_index = task_entry["task_index"]
                task = task_entry["task"]
                if task in merged_tasks:
                    task_index_local_to_global[task_index] = merged_tasks[task]          
                else:
                    merged_tasks[task] = global_task_index
                    task_index_local_to_global[task_index] = global_task_index
                    global_task_index += 1
                
        with open(d / "meta/episodes.jsonl") as f:
            print("  Copying episodes.jsonl from", d)
            for line in f:
                ep = json.loads(line)
                ep["episode_index"] = global_episode_index_episodes
                merged_episodes.append(ep)
                global_episode_index_episodes += 1
                
        with open(d / "meta/episodes_stats.jsonl") as f:
            print("  Copying episodes_stats.jsonl from", d)
            for line in f:
                stat = json.loads(line)
                stat["episode_index"] = global_episode_index_stats
                merged_stats.append(stat)
                
                
                new_index = task_index_local_to_global[stat["stats"]["task_index"]["min"][0]] 
                old_count = stat["stats"]["task_index"]["count"][0]
                stat["stats"]["episode_index"] = {"min": [global_episode_index_stats], "max": [global_episode_index_stats], "mean": [float(global_episode_index_stats)], "std": [float(global_episode_index_stats)], "count": [old_count]}
                stat["stats"]["task_index"] = {"min": [new_index], "max": [new_index], "mean": [float(new_index)], "std": [float(new_index)], "count": [old_count]}
                global_episode_index_stats += 1
        # Cumulative totals after each dataset
        print(f"  Cumulative totals after dataset {d_idx + 1}:")
        print(f"    Total episodes merged: {global_episode_index_episodes}")
        print(f"    Total stats merged: {global_episode_index_stats}")
        print(f"    Total videos copied (onboard): {video_onboard_index}, (behind): {video_behind_index}")
        print(f"    Total unique tasks: {len(merged_tasks)}\n")

    # write merged metadata
    with open(out_dir / "meta/episodes.jsonl", "w") as f:
        for ep in merged_episodes:
            f.write(json.dumps(ep) + "\n")

    with open(out_dir / "meta/episodes_stats.jsonl", "w") as f:
        for stat in merged_stats:
            f.write(json.dumps(stat) + "\n")

    with open(out_dir / "meta/tasks.jsonl", "w") as f:
        for task in merged_tasks.keys():
            task_entry = {"task_index": merged_tasks[task], "task": task}
            f.write(json.dumps(task_entry) + "\n")

    info = {
        "codebase_version": "v2.1",
        "robot_type": "so101_follower",
        "total_episodes": len(merged_episodes),
        "total_frames": sum(ep["length"] for ep in merged_episodes),
        "total_tasks": len(merged_tasks),
        "total_videos": len(merged_episodes) * 2,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {"train": f"0:{len(merged_episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet", 
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    }

    first_info_path = Path(dataset_dirs[0]) / "meta" / "info.json"
    if first_info_path.exists():
        with open(first_info_path) as f:
            first_info = json.load(f)
            info["features"] = first_info.get("features", {})

    with open(out_dir / "meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    print(f"Merging complete. Total episodes: {len(merged_episodes)}, total tasks: {len(merged_tasks)}")

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
