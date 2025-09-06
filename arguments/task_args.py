from dataclasses import dataclass, field


@dataclass
class TaskArguments:
    dataset_name: str = field(
        default="scene_graph",
        metadata={"help": "The name of the dataset to perform"}
    )

    task_generator_model_name: str = field(
        default="qwen-32b",
        metadata={"help": "The model name of the task generator."}
    )

    num_qa: int = field(
        default=20,
        metadata={"help": "The number of QA pairs to generate."}
    )

    k_shot: int = field(
        default=5,
        metadata={"help": "The number example questions to use."}
    )

    num_summarization: int = field(
        default=10,
        metadata={"help": "The number of subgraph summarization to generate."}
    )

    task_gen_max_length: int = field(
        default=1000,
        metadata={"help": "The maximum length for task generation."}
    )