import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
import preprocessing_pytorch as _preprocessing
from typing import Literal, TypeAlias
import dataclasses


Variant = Literal["dummy", "gemma_300m",  "gemma_2b"]

@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: bool


gemma_config_dict = {
    "dummy": Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        ),
    "gemma_300m": Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        ),
    "gemma_2b": Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
}


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        paligemma_config = gemma_config_dict[config.paligemma_variant]
        action_expert_config = gemma_config_dict[config.action_expert_variant]



        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)
        self.state_proj = nn.Linear(32, action_expert_config.width)
        self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
        self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None


    def _preprocess_observation(self, observation, *,train=True):

        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)

        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )


    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        # obs: 
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)




