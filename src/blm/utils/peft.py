import logging
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)

logger = logging.getLogger(__name__)


def create_and_prepare_model(args):
    config = transformers.AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True, token=args.token)
    quantization_config = None

    if args.quantize:
        logger.info("Load quantization config")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        attn_implementation=args.attn_implementation,
        quantization_config=quantization_config,
        token=args.token,
        device_map=None if args.deepspeed else 'auto'
    )

    # find all linear modules in model for lora
    target_modules = find_all_linear_names(model)

    # create lora config
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )

    # pre-process the model by upcasting the layer norms in float 32 for
    # Adapted from https://github.com/tmm1/axolotl/blob/2eda9e02a9d15a7a3f92b41f257d9844d72fc220/src/axolotl/utils/models.py#L338
    logger.info("pre-processing model for peft")
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.bfloat16)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                module = module.to(torch.bfloat16)

    # initialize peft model
    logger.info("initializing peft model")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    
    # enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        
    model = get_peft_model(model, peft_config)

    # logger.info parameters
    model.print_trainable_parameters()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.token)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)