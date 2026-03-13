import torch
import numpy as np
import os
import logging
import wandb
import dataclasses
from config import TrainConfig
from pathlib import Path
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import Dataset, DataLoader, default_collate
from typing import Any
import einops
import torch.nn.functional as F
from transformers import AutoTokenizer
import utils.image_tools as image_tools


# wandb sync xxx  later
os.environ["WANDB_MODE"] = "offline"

def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)

def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)

def init_wandb(config: TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def task_to_prompt(task: str) -> str:
    s = str(task).strip()
    if s.endswith("_demo"):
        s = s[:-5]
    return s

class RawPiDataset(Dataset):
    """
    只负责:
    - 从 base_ds 取样本
    - 基础字段整理
    - 图像转成 HWC uint8
    - task -> prompt
    不做 resize / tokenize / pad
    """

    def parse_image(image: Any) -> np.ndarray:
        """
        输入可能是 torch.Tensor / np.ndarray
        统一转成 uint8, HWC
        """
        image = np.asarray(image)

        # float图像 -> uint8
        if np.issubdtype(image.dtype, np.floating):
            image = (255 * image).clip(0, 255).astype(np.uint8)

        # CHW -> HWC
        if image.ndim == 3 and image.shape[0] == 3:
            image = einops.rearrange(image, "c h w -> h w c")

        return image

    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]

        base_image = self.parse_image(sample["image"])
        left_wrist_image = self.parse_image(sample["wrist_image"])

        out = {
            "state": sample["state"],          # torch tensor / numpy 都行，default_collate 会处理
            "actions": sample["actions"],      # 训练时需要
            "prompt": task_to_prompt(sample["task"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": True,
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": False,    # PI0
            },
        }
        return out


class PaligemmaTokenizer:
    def __init__(self, tokenizer_path: str | Path, max_len: int = 48):
        self._max_len = max_len
        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)

    def _clean_text(self, prompt: str) -> str:
        return prompt.strip().replace("_", " ").replace("\n", " ")

    def tokenize(
        self,
        prompts: list[str],
        state: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is not None:
            raise AssertionError("PI0 不把 state 拼进文本，tokenize 不接受 state 参数")

        texts = [self._clean_text(str(p)) + "\n" for p in prompts]

        enc = self._tokenizer(
            texts,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self._max_len,
            return_attention_mask=True,
            return_tensors="pt",
        )

        tokens = enc["input_ids"].long()          # [B, L]
        mask = enc["attention_mask"].bool()       # [B, L]

        if (mask.sum(dim=1) >= self._max_len).any():
            logging.warning(
                "Some prompts reached/exceeded max length (%d), text may be truncated.",
                self._max_len,
            )

        return tokens, mask


def pad_last_dim_torch(x: torch.Tensor, target_dim: int, value: float = 0.0) -> torch.Tensor:
    cur_dim = x.shape[-1]
    if cur_dim > target_dim:
        raise ValueError(f"current dim {cur_dim} > target dim {target_dim}")
    if cur_dim == target_dim:
        return x
    return F.pad(x, (0, target_dim - cur_dim), value=value)

def process_pi0_batch(
    batch: dict,
    tokenizer: PaligemmaTokenizer,
    *,
    image_size: int = 224,
    action_dim: int = 32
):
    """
    输入 batch:
      state:   [B, 8]
      actions: [B, 7]
      image/*: [B, H, W, 3]
      prompt:  list[str] 或 task

    输出 batch:
      state:   [B, 32]
      actions: [B, 32]
      image/*: [B, 224, 224, 3]
      tokenized_prompt:      [B, 48]
      tokenized_prompt_mask: [B, 48]
    """
    out = {}

    # state / actions pad
    out["state"] = pad_last_dim_torch(batch["state"], action_dim)

    if "actions" in batch:
        out["actions"] = pad_last_dim_torch(batch["actions"], action_dim)

    # image resize + pad
    out["image"] = {
        "base_0_rgb": image_tools.resize_with_pad_torch(batch["image"]["base_0_rgb"], image_size, image_size),
        "left_wrist_0_rgb": image_tools.resize_with_pad_torch(batch["image"]["left_wrist_0_rgb"], image_size, image_size),
        "right_wrist_0_rgb": image_tools.resize_with_pad_torch(batch["image"]["right_wrist_0_rgb"], image_size, image_size),
    }

    # image_mask 保留
    if "image_mask" in batch:
        out["image_mask"] = batch["image_mask"]

    # prompt -> tokens
    if "prompt" not in batch:
        raise KeyError("batch 中缺少 'prompt' 字段")

    prompts = batch["prompt"]
    prompt_ids, prompt_mask = tokenizer.tokenize(prompts)

    out["tokenized_prompt"] = prompt_ids
    out["tokenized_prompt_mask"] = prompt_mask

    return out

def make_pi0_collate_fn(
    tokenizer: PaligemmaTokenizer,
    *,
    image_size: int = 224,
    action_dim: int = 32,
    prompt_key: str = "prompt",
    fallback_task_key: str = "task",
):
    def collate_fn(samples: list[dict]) -> dict:
        # 先用默认 collate 拼成 batch
        batch = default_collate(samples)

        # 再做 PI0 前处理
        pi0_batch = process_pi0_batch(
            batch,
            tokenizer,
            image_size=image_size,
            action_dim=action_dim,
            prompt_key=prompt_key,
            fallback_task_key=fallback_task_key,
        )
        return pi0_batch

    return collate_fn

def train_loop(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = True
    set_seed(config.seed)

    # tmp config -------------------------------------------------------------------------
    repo_id = "flow929/ledataset_libero_spatial"
    root = HF_LEROBOT_HOME / repo_id


    # -------------------------------------------------------------------------
    ds = LeRobotDataset(repo_id)

    # 1. base dataset: 你已有的 ds
    raw_ds = RawPiDataset(ds)

    # 2. tokenizer
    tokenizer = PaligemmaTokenizer(
        tokenizer_model_path="/你的路径/paligemma_tokenizer.model",
        max_len=48,
    )

    # 3. dataloader
    loader = DataLoader(
        raw_ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=make_pi0_collate_fn(
            tokenizer,
            image_size=224,
            action_dim=32,
            prompt_key="prompt",
            fallback_task_key="task",
        ),
    )

    # 4. 直接拿最终 pi0 batch
    pi0_batch = next(iter(loader))




def main():
    init_logging()



    train_loop(config)





























if __name__ == "__main__":
    main()
