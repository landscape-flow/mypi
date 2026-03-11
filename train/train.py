import torch
import numpy as np
import os
import logging
import wandb
import dataclasses



class TrainConfig:


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



def train_loop(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = True
    set_seed(config.seed)








    cfg = LiberoTorchConfig(
        repo_id="physical-intelligence/libero",
        action_horizon=50,
        batch_size=4,
        num_workers=0,
    )

    loader = build_libero_dataloader(cfg)








def main():
    init_logging()
    config = TrainConfig.cli()

    train_loop(config)


if __name__ == "__main__":
    main()


























    return 0

def main():
    init_logging()
    config = _config.cli()
    train_loop(config)

if __name__ == "__main__":
    main()
