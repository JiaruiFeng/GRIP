from typing import Optional

from transformers import PreTrainedTokenizer, PreTrainedModel

from constants import TORCH_DTYPE, HF_DECODER_ONLY_LLMS
from models.utils import get_hf_llm_tokenizer


def get_hf_ft_model(
        model_name: str,
        load_dir: Optional[str] = None,
        padding_side: str = "left",
        truncation_side: str = "left",
        tokenize_max_length: int = 4096,
        quantization: bool = False,
        dtype: str = "bfloat16",
        use_prefix: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        dropout: float = 0.0,
        num_virtual_tokens: int = 20,
        encoder_hidden_size: int = 128,
        **kwargs,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    return get_hf_llm_tokenizer(
        model_name=HF_DECODER_ONLY_LLMS[model_name],
        load_dir=load_dir,
        padding_side=padding_side,
        truncation_side=truncation_side,
        tokenize_max_length=tokenize_max_length,
        use_fast=True,
        flash_attention=False,
        dtype=TORCH_DTYPE[dtype],
        quantization=quantization,
        peft=True,
        use_prefix=use_prefix,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        dropout=dropout,
        num_virtual_tokens=num_virtual_tokens,
        encoder_hidden_size=encoder_hidden_size,
        **kwargs,
    )
