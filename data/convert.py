import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME


# ========= 你要改的地方 =========
SRC_DIR = Path("/home/flow/code/libero/LIBERO/libero/datasets/libero_goal")
REPO_ID = "flow929/ledataset_libero_spatial"
FPS = 10
IMAGE_SIZE = 256
OVERWRITE = True

# 写图片阶段的并发参数
IMAGE_WRITER_PROCESSES = 4
IMAGE_WRITER_THREADS = 8
# ==============================


def resize_rgb(img: np.ndarray, size: int = 256) -> np.ndarray:
    """Resize HWC uint8 RGB image to (size, size, 3)."""
    out = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    if out.dtype != np.uint8:
        out = out.astype(np.uint8)
    return out


def build_state(ee_states_t: np.ndarray, gripper_states_t: np.ndarray) -> np.ndarray:
    """Concatenate ee_states and gripper_states into 1D float32 state."""
    state = np.concatenate(
        [
            ee_states_t.reshape(-1),
            gripper_states_t.reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)
    return state


def infer_feature_shapes(first_hdf5: Path):
    """Infer dataset feature shapes from the first demo of the first file."""
    with h5py.File(first_hdf5, "r") as f:
        demo_names = sorted(f["data"].keys())
        if not demo_names:
            raise ValueError(f"No demos found in {first_hdf5}")

        demo = f["data"][demo_names[0]]

        image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
        wrist_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

        state_dim = (
            demo["obs"]["ee_states"][0].reshape(-1).shape[0]
            + demo["obs"]["gripper_states"][0].reshape(-1).shape[0]
        )
        action_dim = demo["actions"][0].reshape(-1).shape[0]

    return image_shape, wrist_shape, state_dim, action_dim


def create_lerobot_dataset(
    repo_id: str,
    image_shape: tuple[int, int, int],
    wrist_shape: tuple[int, int, int],
    state_dim: int,
    action_dim: int,
) -> LeRobotDataset:
    """Create a LeRobot image-based dataset."""
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type="panda",
        features={
            "image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": wrist_shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        },
        use_videos=True,
        image_writer_processes=IMAGE_WRITER_PROCESSES,
        image_writer_threads=IMAGE_WRITER_THREADS,
    )
    return dataset


def convert_one_demo(dataset: LeRobotDataset, demo, task_name: str) -> int:
    """
    Convert one demo into one LeRobot episode.
    Returns the number of frames written.
    """
    agentview = demo["obs"]["agentview_rgb"]
    wristview = demo["obs"]["eye_in_hand_rgb"]
    ee_states = demo["obs"]["ee_states"]
    gripper_states = demo["obs"]["gripper_states"]
    actions = demo["actions"]

    T = actions.shape[0]

    # 一些基本一致性检查
    assert agentview.shape[0] == T
    assert wristview.shape[0] == T
    assert ee_states.shape[0] == T
    assert gripper_states.shape[0] == T

    for t in range(T):
        image_t = resize_rgb(agentview[t], IMAGE_SIZE)
        wrist_t = resize_rgb(wristview[t], IMAGE_SIZE)
        state_t = build_state(ee_states[t], gripper_states[t])
        action_t = actions[t].reshape(-1).astype(np.float32)

        frame = {
            "image": image_t,
            "wrist_image": wrist_t,
            "state": state_t,
            "actions": action_t,
            "task": task_name,
        }
        dataset.add_frame(frame)

    dataset.save_episode(parallel_encoding=False)
    return T


def main():
    hdf5_files = sorted(SRC_DIR.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in: {SRC_DIR}")

    output_path = HF_LEROBOT_HOME / REPO_ID
    if OVERWRITE and output_path.exists():
        shutil.rmtree(output_path)

    image_shape, wrist_shape, state_dim, action_dim = infer_feature_shapes(hdf5_files[0])

    dataset = create_lerobot_dataset(
        repo_id=REPO_ID,
        image_shape=image_shape,
        wrist_shape=wrist_shape,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    total_files = 0
    total_episodes = 0
    total_frames = 0

    for hdf5_path in hdf5_files:
        total_files += 1
        print(f"\nProcessing file: {hdf5_path.name}")

        with h5py.File(hdf5_path, "r") as f:
            demo_names = sorted(f["data"].keys())
            print(f"  demos: {len(demo_names)}")

            for demo_name in demo_names:
                demo = f["data"][demo_name]
                task_name = hdf5_path.stem

                num_frames = convert_one_demo(dataset, demo, task_name)

                total_episodes += 1
                total_frames += num_frames
                # print(f"  saved {demo_name}: {num_frames} frames")

    print("\nDone.")
    print(f"Total files    : {total_files}")
    print(f"Total episodes : {total_episodes}")
    print(f"Total frames   : {total_frames}")
    print(f"Output path    : {output_path}")


if __name__ == "__main__":
    main()