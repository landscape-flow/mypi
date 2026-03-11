import torch
from torch import nn
from transformers import PaliGemmaForConditionalGeneration
from transformers import GemmaForCausalLM
from transformers.models.auto import CONFIG_MAPPING
from typing import Literal

'''
PI0Pytorch
├─ embed_prefix: 图像+语言 -> prefix_embs
├─ embed_suffix: 状态+动作+时间 -> suffix_embs
└─ PaliGemmaWithExpertModel
   ├─ prefix分支: PaliGemma.language_model
   ├─ suffix分支: GemmaForCausalLM.model
   └─ 每层把 prefix/suffix 的 QKV 拼起来做联合 attention

'''

class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",

    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"


        action_expert_config_hf = CONFIG_MAPPING["gemma"]()
        action_expert_config_hf.head_dim=action_expert_config.head_dim,
        action_expert_config_hf.hidden_size=action_expert_config.width,
        action_expert_config_hf.intermediate_size=action_expert_config.mlp_dim,
        action_expert_config_hf.num_attention_heads=action_expert_config.num_heads,
        action_expert_config_hf.num_hidden_layers=action_expert_config.depth,
        action_expert_config_hf.num_key_value_heads=action_expert_config.num_kv_heads,
        action_expert_config_hf.vocab_size=257152,
        action_expert_config_hf.hidden_activation="gelu_pytorch_tanh",
        action_expert_config_hf.torch_dtype="float32",
        action_expert_config_hf.use_adarms=use_adarms[1],
        action_expert_config_hf.adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,

        self.vlm = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.expert.model.embed_tokens = None  # 不需要输入 token id 的 embedding 层

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.vlm.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.vlm.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
