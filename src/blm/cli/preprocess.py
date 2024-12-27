import logging
import argparse
import numpy as np
import random
import json
from datasets import Dataset, DatasetDict
from blm.utils.helpers import logging_config

logger = logging.getLogger(__name__)


def save_dataset(examples, output_path):
    random.shuffle(examples)
    train_dataset, eval_dataset = np.split(examples, [int(0.95 * len(examples))])
    train_dataset = train_dataset.tolist()
    eval_dataset = eval_dataset.tolist()

    ds = DatasetDict({
        "train": Dataset.from_list(train_dataset),
        "eval": Dataset.from_list(eval_dataset)
    })

    ds.save_to_disk(output_path)


def main(args):
    with open(args.input_path, "r") as fh:
        if args.n:
            examples = [json.loads(next(fh)) for _ in range(args.n)]
        else:
            examples = [json.loads(line) for line in fh.readlines()]

    save_dataset(examples, args.output_path)
    logger.info(f"Total training examples: {len(examples)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help="Data output path")
    parser.add_argument("--input_path", type=str, help="Data input path")
    parser.add_argument("--n", type=int, default=None, help="Number of examples to generate, use for testing.")
    args = parser.parse_args()

    logging_config("processing.log")

    main(args)