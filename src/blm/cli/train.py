"""
CLI to run training on a model
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from transformers.hf_argparser import HfArgumentParser
from transformers import TrainingArguments
from huggingface_hub import login
import logging
from blm.utils.train import train
from blm.utils.helpers import merge_object_properties, logging_config, save_package_versions

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    attn_implementation: str = field(
        default=None,
        metadata={"help": "Attention mechanism"}
    )
    quantize: bool = field(
        default=False,
        metadata={"help": "Quantize model"}
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )


@dataclass
class LoRaArguments:
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )


def do_train():
    parser = HfArgumentParser((DataArguments, TrainingArguments, ModelArguments, LoRaArguments))
    data_args, train_args, model_args, lora_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = merge_object_properties(data_args, train_args, model_args, lora_args)
    logging_config(os.path.join(args.output_dir, "train.log"))

    if args.token:
        logger.info(f"Logging into the Hugging Face Hub with token {args.token[:10]}...")
        login(token=args.token)
    
    save_package_versions(os.path.join(args.output_dir, "requirements.txt"))
    train(args)


if __name__ == "__main__":
    do_train()