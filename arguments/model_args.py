from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name: str = field(
        default="qwen-7b",
        metadata={"help": "The name of the model."}
    )

    load_dir: str = field(
        default=None,
        metadata={"help": "The directory to load the model from."}
    )

    tokenize_max_length: int = field(
        default=4096,
        metadata={"help": "The maximum length of the tokenized input."}
    )

    padding_side: str = field(
        default="right",
        metadata={"help": "The padding side of the tokenizer."}
    )

    truncation_side: str = field(
        default="right",
        metadata={"help": "The truncation side of the tokenizer."}
    )

    dtype: str = field(
        default="bfloat16",
        metadata={"help": "The dtype of the model."}
    )

    quantization: bool = field(
        default=False,
        metadata={"help": "Whether to quantize the model."}
    )
