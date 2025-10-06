from typing import Optional, Union

import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PrefixTuningConfig, PeftMixedModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer, PreTrainedModel


def get_hf_llm_tokenizer(
        model_name: str,
        load_dir: Optional[str] = None,
        device_map: Optional[str] = "auto",
        padding_side: str = "left",
        truncation_side: str = "left",
        tokenize_max_length: int = 4096,
        use_fast: bool = True,
        flash_attention: bool = False,
        quantization: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        peft: bool = False,
        use_prefix: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        target_modules: Optional[list[str]] = None,
        dropout: float = 0.0,
        num_virtual_tokens: int = 20,
        encoder_hidden_size: int = 128,
        **kwargs,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    if peft:
        device_map = None

    if flash_attention:
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"

    if quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        bnb_config = None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        quantization_config=bnb_config,
        attn_implementation=attn_implementation,
    )

    if load_dir is not None:
        model = PeftModel.from_pretrained(
            model,
            load_dir,
            torch_dtype=dtype,
            **kwargs,
        )
        model = model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(
            load_dir,
            use_fast=use_fast,
            padding=True,
            truncation=True,
            padding_side=padding_side,
            truncation_side=truncation_side,
            max_length=tokenize_max_length,
            **kwargs,
        )

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast,
            padding=True,
            truncation=True,
            padding_side=padding_side,
            truncation_side=truncation_side,
            max_length=tokenize_max_length,
            **kwargs,
        )
    pad_token = "[PAD]"
    tokenizer.pad_token = pad_token
    if pad_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token": pad_token})
        model.resize_token_embeddings(len(tokenizer))

    if peft:
        model = load_peft_model(
            model,
            use_prefix=use_prefix,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            dropout=dropout,
            num_virtual_tokens=num_virtual_tokens,
            encoder_hidden_size=encoder_hidden_size,
            **kwargs,
        )
    return model, tokenizer


def load_peft_model(
        model: PreTrainedModel,
        use_prefix: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32,
        target_modules: Optional[list[str]] = None,
        dropout: float = 0.0,
        num_virtual_tokens: int = 20,
        encoder_hidden_size: int = 128,
        **kwargs
) -> PeftModel:
    if use_prefix:
        print("Load prefix tuning model......")
        model = get_prefix_model(
            model,
            num_virtual_tokens=num_virtual_tokens,
            encoder_hidden_size=encoder_hidden_size,
            **kwargs,
        )
    else:
        print("Load LoRA model......")
        model = get_lora_model(
            model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            dropout=dropout,
            **kwargs,
        )
    return model


def get_lora_model(
        model: Union[PreTrainedModel, PeftModel],
        lora_r: int = 8,
        lora_alpha: int = 32,
        dropout: float = 0.0,
        target_modules: Optional[list[str]] = None,
        **kwargs) -> PeftModel:
    if target_modules is None:
        target_modules = ["k_proj", "v_proj", "q_proj"]
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    # Remove previous lora modules.
    while isinstance(model, PeftModel):
        if hasattr(model, "unload"):
            model = model.unload()
        else:
            model = model.base_model
        
    model = get_peft_model(model, lora_config, adapter_name="mylora")
    model.print_trainable_parameters()

    return model


def get_prefix_model(
        model: Union[PreTrainedModel, PeftModel],
        num_virtual_tokens: int = 20,
        encoder_hidden_size: int = 128,
        **kwargs) -> PeftModel:
    r"""Create prefix model.
    """
    prefix_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        num_virtual_tokens=num_virtual_tokens,
        encoder_hidden_size=encoder_hidden_size,
        prefix_projection=True,

    )
    # Remove previous lora modules.
    while isinstance(model, PeftModel) or isinstance(model, PeftMixedModel):
        if hasattr(model, "unload"):
            model = model.unload()
        else:
            model = model.base_model
    model = get_peft_model(model, prefix_config, adapter_name="myprefix")
    model.print_trainable_parameters()

    return model
