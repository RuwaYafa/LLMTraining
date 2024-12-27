import torch
import os
import json
import argparse
import logging
import natsort
import tarfile
import shutil
from glob import glob
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from blm.utils.helpers import download_and_untar_model_from_s3, logging_config

logger = logging.getLogger(__name__)


def main(args):
    root_dir = args.merged_path

    adapters_dir = os.path.join(root_dir, "adapters")
    merged_weights_dir = os.path.join(root_dir, "merged_weights")
    merged_weights_tar = os.path.join(root_dir, "merged_weights", "llm.tar.gz")

    os.makedirs(adapters_dir, exist_ok=True)
    os.makedirs(merged_weights_dir, exist_ok=True)
    
    logger.info(f"Download and untar model adapters from {args.lora_path}")
    download_and_untar_model_from_s3(args.lora_path, adapters_dir)
    
    # Locate the folder for the checkpoint, if the checkpoint is in the cli
    # argument we will use it, otherwise fine the last checkpoint to merge
    if args.checkpoint:
        checkpoint_path = os.path.join(adapters_dir, f"checkpoint-{args.checkpoint}")
    else:
        checkpoint_path = natsort.natsorted(glob(f"{adapters_dir}/checkpoint-*"))
        checkpoint_path = checkpoint_path[-1]

    logger.info(f"LoRA checkcpoint: {checkpoint_path}")
    device = None if torch.cuda.is_available() else "cpu"

    adapter_config_file = os.path.join(checkpoint_path, "adapter_config.json")

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
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model = model.to(device)
    
    # Merge PEFT with base model
    logger.info(f"Merge weights")
    model = model.merge_and_unload()
    
    # Save model and tokenizer
    logger.info("Save model")
    model.save_pretrained(merged_weights_dir, safe_serialization=True, max_shard_size="2GB")

    logger.info("Save tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(merged_weights_dir)
    
    # Delete adapters, no longer needed, it will also save disk space
    logger.info(f"Delete LoRA adapters in {adapters_dir} - no longer needed")
    shutil.rmtree(adapters_dir)
    
    # tar the model
    logger.info(f"tar model to {merged_weights_tar}")
    with tarfile.open(merged_weights_tar, "w:gz") as tar:
        for f in os.listdir(merged_weights_dir):
            tar.add(os.path.join(merged_weights_dir, f), arcname=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, help="S3 location to LoRA adapters")
    parser.add_argument("--merged_path", type=str, help="S3 bucket or local path where the merged LoRA adapters are saved")
    parser.add_argument("--checkpoint", type=int, default=None, help="Use most recent checkpoint or specify checkpoint")
    parser.add_argument("--hf_token", type=str, help="Huggingface token")
    args = parser.parse_args()

    if args.hf_token:
        logger.info(f"Logging into the Hugging Face Hub with token {args.hf_token[:10]}...")
        login(token=args.hf_token)

    logging_config("merge_lora.log")

    main(args)