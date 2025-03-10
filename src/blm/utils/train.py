from datasets import load_from_disk
import logging
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from blm.utils.peft import create_and_prepare_model
from blm.utils.prompter import Prompter

logger = logging.getLogger(__name__)


def train(args):
    # Load and create peft model
    model, peft_config, tokenizer = create_and_prepare_model(args)
    model.config.use_cache = False

    prompter = Prompter(tokenizer)
    dataset = load_from_disk(args.data_path)
    dataset = dataset.map(prompter, batched=True)

    logger.info(f"Pre-saving tokenizer to {args.output_dir}")
    tokenizer.save_pretrained(args.output_dir)

    if peft_config:
        logger.info(f"Pre-saving adapter config to {args.output_dir}")
        peft_config.save_pretrained(args.output_dir)

    if hasattr(model, "config"):
        logger.info(f"Pre-saving model config to {args.output_dir}")
        model.config.save_pretrained(args.output_dir)

    collator = DataCollatorForCompletionOnlyLM(response_template=prompter.response_template, 
                                               instruction_template=prompter.instruction_template, 
                                               tokenizer=tokenizer, 
                                               mlm=False)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        dataset_text_field="prompt",
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        args=args,
        data_collator=collator
    )

    logger.info("Model parameters...")
    trainer.model.print_trainable_parameters()

    # Start training
    trainer.train()
