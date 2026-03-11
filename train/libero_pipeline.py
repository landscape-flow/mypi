# libero_torch_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import einops
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lerobot.datasets.lerobot_dataset as lerobot_dataset


@dataclass
class LiberoTorchConfig:
    repo_id: str
    action_horizon: int = 10
    batch_size: int = 8
    num_workers: int = 0
    shuffle: bool = True
    prompt_from_task: bool = True

def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def parse_image(image: Any) -> np.ndarray:
    """
    对齐 libero policy / preprocess 的图像预期：
    - float 图像 -> uint8
    - CHW -> HWC
    """
    image = _to_numpy(image)

    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)

    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    return image



class LiberoPolicyStyleTransform:
    """
    单条样本整理成中间 batch 格式：
    {
        "state": ...,
        "image": {
            "base_0_rgb": ...,
            "left_wrist_0_rgb": ...,
            "right_wrist_0_rgb": ...,
        },
        "image_mask": {
            "base_0_rgb": ...,
            "left_wrist_0_rgb": ...,
            "right_wrist_0_rgb": ...,
        },
        "prompt": ...,
        "actions": ...,
    }
    """

    def __init__(self, use_pi0_fast: bool = False):
        self.use_pi0_fast = use_pi0_fast

    def __call__(self, data: dict) -> dict:
        base_image = parse_image(data["observation/image"])
        wrist_image = parse_image(data["observation/wrist_image"])
        state = _to_numpy(data["observation/state"]).astype(np.float32)

        out = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.array(True),
                "left_wrist_0_rgb": np.array(True),
                "right_wrist_0_rgb": np.array(True if self.use_pi0_fast else False),
            },
        }

        if "actions" in data:
            out["actions"] = _to_numpy(data["actions"]).astype(np.float32)

        if "prompt" in data:
            out["prompt"] = data["prompt"]

        return out


class LiberoLeRobotDataset(Dataset):
    """
    从 LeRobotDataset 读取 LIBERO，并整理成符合你模型输入习惯的单样本。
    """

    def __init__(self, cfg: LiberoTorchConfig):
        self.cfg = cfg
        self.meta = lerobot_dataset.LeRobotDatasetMetadata(cfg.repo_id)
        self.dataset = lerobot_dataset.LeRobotDataset(
            cfg.repo_id,
            delta_timestamps={
                "actions": [t / self.meta.fps for t in range(cfg.action_horizon)]
            },
        )
        self.transform = LiberoPolicyStyleTransform(use_pi0_fast=cfg.use_pi0_fast)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        raw = self.dataset[idx]

        sample = {
            "observation/image": raw["observation"]["image"],
            "observation/wrist_image": raw["observation"]["wrist_image"],
            "observation/state": raw["observation"]["state"],
        }

        if "actions" in raw:
            sample["actions"] = raw["actions"]

        if self.cfg.prompt_from_task and "task_index" in raw:
            task_index = int(raw["task_index"])
            sample["prompt"] = self.meta.tasks[task_index]
        elif "prompt" in raw:
            sample["prompt"] = raw["prompt"]

        return self.transform(sample)


def _collate_nested(items: list[Any]) -> Any:
    first = items[0]

    if isinstance(first, dict):
        return {k: _collate_nested([item[k] for item in items]) for k in first}

    if isinstance(first, str):
        return items

    if isinstance(first, torch.Tensor):
        return torch.stack(items, dim=0)

    if isinstance(first, np.ndarray):
        return torch.stack([torch.from_numpy(x) for x in items], dim=0)

    if isinstance(first, (bool, np.bool_)):
        return torch.tensor(items, dtype=torch.bool)

    if isinstance(first, (int, np.integer)):
        return torch.tensor(items, dtype=torch.long)

    if isinstance(first, (float, np.floating)):
        return torch.tensor(items, dtype=torch.float32)

    raise TypeError(f"Unsupported type in batch collation: {type(first)}")


def libero_collate_fn(items: list[dict]) -> dict:
    return _collate_nested(items)


def build_libero_dataloader(cfg: LiberoTorchConfig) -> DataLoader:
    dataset = LiberoLeRobotDataset(cfg)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        drop_last=True,
        collate_fn=libero_collate_fn,
    )


def move_batch_to_device(batch: Any, device: torch.device | str) -> Any:
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return batch
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch


class SimpleObservation:
    """
    与 preprocess_observation_pytorch 的输入字段严格对齐。
    """
    def __init__(
        self,
        *,
        images,
        image_masks,
        state,
        tokenized_prompt,
        tokenized_prompt_mask,
        token_ar_mask,
        token_loss_mask,
    ):
        self.images = images
        self.image_masks = image_masks
        self.state = state
        self.tokenized_prompt = tokenized_prompt
        self.tokenized_prompt_mask = tokenized_prompt_mask
        self.token_ar_mask = token_ar_mask
        self.token_loss_mask = token_loss_mask


def tokenize_prompts_for_openpi(prompts: list[str], tokenizer, device: torch.device | str):
    """
    生成 preprocess_observation_pytorch 所需字段：
    - tokenized_prompt
    - tokenized_prompt_mask
    - token_ar_mask
    - token_loss_mask
    """
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    tokenized_prompt = tokenized["input_ids"].to(device)
    tokenized_prompt_mask = tokenized["attention_mask"].to(device).bool()

    # 先给最小可跑版本：
    # prompt token 不参与 action loss，因此全 False
    token_ar_mask = torch.zeros_like(tokenized_prompt_mask, dtype=torch.bool, device=device)
    token_loss_mask = torch.zeros_like(tokenized_prompt_mask, dtype=torch.bool, device=device)

    return tokenized_prompt, tokenized_prompt_mask, token_ar_mask, token_loss_mask


def pack_observation(batch: dict, tokenizer, device: torch.device | str) -> SimpleObservation:
    """
    把 dataloader 输出的 batch，整理成你的 model.forward(observation, actions) 需要的 observation。
    """
    batch = move_batch_to_device(batch, device)

    if "prompt" not in batch:
        raise ValueError("batch 中缺少 prompt，无法构造 tokenized_prompt。")

    (
        tokenized_prompt,
        tokenized_prompt_mask,
        token_ar_mask,
        token_loss_mask,
    ) = tokenize_prompts_for_openpi(batch["prompt"], tokenizer, device)

    observation = SimpleObservation(
        images=batch["image"],
        image_masks=batch["image_mask"],
        state=batch["state"],
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        token_ar_mask=token_ar_mask,
        token_loss_mask=token_loss_mask,
    )
    return observation


def pack_observation_and_actions(batch: dict, tokenizer, device: torch.device | str):
    observation = pack_observation(batch, tokenizer, device)

    if "actions" not in batch:
        raise ValueError("batch 中缺少 actions。")

    actions = move_batch_to_device(batch["actions"], device).to(torch.float32)
    return observation, actions


def parse_model_outputs(outputs: dict) -> dict:
    """
    对齐 libero policy 的输出约定：
    只取前 7 维动作。
    """
    return {"actions": outputs["actions"][:, :7]}


if __name__ == "__main__":
    cfg = LiberoTorchConfig(
        repo_id="your_lerobot_libero_repo_id",
        action_horizon=10,
        batch_size=2,
        num_workers=0,
        shuffle=True,
        prompt_from_task=True,
        use_pi0_fast=False,
    )

    loader = build_libero_dataloader(cfg)
    batch = next(iter(loader))

    print("batch keys:", batch.keys())
    print("state shape:", batch["state"].shape)
    print("image keys:", batch["image"].keys())
    print("base image shape:", batch["image"]["base_0_rgb"].shape)
    print("actions shape:", batch["actions"].shape if "actions" in batch else None)
    print("prompt sample:", batch["prompt"][0] if "prompt" in batch else None)