import torch
import os
import json
import argparse
import logging
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from blm.utils.helpers import logging_config

logger = logging.getLogger(__name__)


def main(args):
    os.makedirs(args.merged_path)

    logger.info(f"LoRA checkcpoint: {args.lora_path}")
    device = None if torch.cuda.is_available() else "cpu"

    adapter_config_file = os.path.join(args.lora_path, "adapter_config.json")

    with open(adapter_config_file, "r") as fh:
        adapter_config = json.load(fh)

    model_id = adapter_config["base_model_name_or_path"]
    
    # Load base model, model ID is in the adapter configuration
    logger.info(f"Loading base model: {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )

    # Load the PEFT model
    logger.info(f"Loading PEFT model")
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model = model.to(device)
    
    # Merge PEFT with base model
    logger.info(f"Merge weights")
    model = model.merge_and_unload()
    
    # Save model and tokenizer
    logger.info("Save model")
    model.save_pretrained(args.merged_path, safe_serialization=True, max_shard_size="2GB")

    logger.info("Save tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(args.merged_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, help="Path to LoRA adapters")
    parser.add_argument("--merged_path", type=str, help="Local path where the merged LoRA adapters are saved")
    parser.add_argument("--hf_token", type=str, help="Huggingface token")
    args = parser.parse_args()

    if args.hf_token:
        logger.info(f"Logging into the Hugging Face Hub with token {args.hf_token[:10]}...")
        login(token=args.hf_token)

    logging_config("merge_lora.log")

    main(args)