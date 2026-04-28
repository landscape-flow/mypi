import torch
import numpy as np
import logging
import wandb
import dataclasses
import sys
import os

from pathlib import Path
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import Dataset, DataLoader, default_collate
from typing import Any
import einops
import torch.nn.functional as F
from transformers import AutoTokenizer
import tqdm, time

# 获取当前文件的上一级目录（即 项目根目录）
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils.image_tools as image_tools
from models.pio import PI0Pytorch


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
    @staticmethod
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


def train_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = True
    # set_seed(config.seed)

    # tmp config -------------------------------------------------------------------------
    repo_id = "flow929/ledataset_libero_spatial"
    root = HF_LEROBOT_HOME / repo_id


    # -------------------------------------------------------------------------
    ds = LeRobotDataset(repo_id)

    # 1. base dataset: 你已有的 ds
    raw_ds = RawPiDataset(ds)

    # 2. tokenizer
    tokenizer = PaligemmaTokenizer(
        tokenizer_path="/home/flow/code/mypi/models/paligemma_tokenizer",
        max_len=48,
    )

    def pi0_collate_fn(samples):
        batch = default_collate(samples)

        batch = process_pi0_batch(
            batch,
            tokenizer,
            image_size=224,
            action_dim=32
        )

        # ===== 转成 PI0 期望格式 =====
        images = list(batch["image"].values())             # list of [B,H,W,3]
        image_masks = list(batch["image_mask"].values())   # list of bool

        observation = (
            images,
            image_masks,
            batch["tokenized_prompt"],
            batch["tokenized_prompt_mask"],
            batch["state"],
        )

        actions = batch["actions"]

        return observation, actions

    # 3. dataloader
    loader = DataLoader(
        raw_ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=pi0_collate_fn
    )

    
    model = PI0Pytorch().to(device)
    model.train()

    # 4. DEBUG pi0 batch
    def inspect_batch(batch, name="batch"):
        print(f"\n===== {name} =====")
        for k, v in batch.items():
            if isinstance(v, dict):
                print(f"{k}:")
                for kk, vv in v.items():
                    shape = tuple(vv.shape) if hasattr(vv, "shape") else type(vv)
                    dtype = vv.dtype if hasattr(vv, "dtype") else type(vv)
                    print(f"  {kk:<20} shape={shape}, dtype={dtype}")
            else:
                shape = tuple(v.shape) if hasattr(v, "shape") else type(v)
                dtype = v.dtype if hasattr(v, "dtype") else type(v)
                print(f"{k:<24} shape={shape}, dtype={dtype}")
    pi0_batch = next(iter(loader))
    inspect_batch(pi0_batch, "pi0_batch from loader")

    # 5. optimizer（对齐原 AdamW）----------
    peak_lr = 2.5e-5            # CosineDecaySchedule.peak_lr
    warmup_steps = 1000         # CosineDecaySchedule.warmup_steps
    decay_steps = 30000         # CosineDecaySchedule.decay_steps
    end_lr = 2.5e-6             # CosineDecaySchedule.decay_lr

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,               # 初始占位，实际会被 lr_schedule 覆盖
        betas=(0.9, 0.95),        # b1, b2
        eps=1e-8,
        weight_decay=1e-10,       # 原配置中 weight_decay = 1e-10
    )
    grad_clip_norm = 1.0          # clip_gradient_norm

    # 6. LR schedule（完全对齐 warmup_cosine_decay_schedule）----------
    def lr_schedule(step: int):
        if step < warmup_steps:
            # 线性 warmup：从 peak_lr/(warmup_steps+1) 到 peak_lr
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    # 7. 训练参数 ----------
    num_train_steps = decay_steps  
    log_interval = 100
    save_interval = 1000
    wandb_enabled = False   

    global_step = 0
    start_time = time.time()
    infos = []      # 存储最近 log_interval 步的统计

    pbar = tqdm(total=num_train_steps, desc="Training", initial=0) if is_main else None
    #
    def to_device(batch, device):
        if isinstance(batch, dict):
            return {k: to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(to_device(x, device) for x in batch)
        elif isinstance(batch, torch.Tensor):
            return batch.to(device)
        else:
            return batch
        
    # ---------- 训练循环 ----------
    while global_step < num_train_steps:
        for observation, actions in loader:
            if global_step >= num_train_steps:
                break
            
            # Move to device
            observation = to_device(observation, device)
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # ===== 3. LR 更新 =====
            lr = lr_schedule(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            # Forward pass
            losses = model(observation, actions)

            if isinstance(losses, (list, tuple)):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            #===== 5. backward =====
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip_norm
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # ===== 6. logging =====
            infos.append({
                "loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
            })

            if global_step % log_interval == 0 and len(infos) > 0:
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                if wandb_enabled and len(infos) > 0:
                    pass            

                start_time = time.time()
                infos = []  # Reset stats collection

            
            # save checkpoint
            if global_step % save_interval == 0:
                torch.save({
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "step": global_step,
                }, f"ckpt_{global_step}.pt")

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

            global_step += 1

    # Close progress bar
    if pbar is not None:
        pbar.close()

    if wandb_enabled:
        pass    



def main():
    # init_logging()
    train_loop()





























if __name__ == "__main__":
    main()
