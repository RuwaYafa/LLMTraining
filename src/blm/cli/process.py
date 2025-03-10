import logging
import argparse
from datasets import Dataset, DatasetDict, load_dataset
from blm.utils.helpers import logging_config

logger = logging.getLogger(__name__)


def save_dataset(dataset, output_path, n):
    """
    Convert the dataset into messages as follows, each training example will
    follow the following format:
      {
        "messages": [{"content": "<INSTRUCTIONS>", "role": "user"},
                     {"content": "<RESPONSE>", "role": "assistant"}
                    ]
      }
    :param dataset: Dataset
    :param output_path: str - output path to save dataset to
    :param n: int - number of training examples to sample
    :return: Dataset
    """
    train_messages = [{"messages": [{"content": e["instruction"], "role": "user"}, {"content": e["output"], "role": "assistant"}]} for e in dataset["train"]]
    eval_messages = [{"messages": [{"content": e["instruction"], "role": "user"}, {"content": e["output"], "role": "assistant"}]} for e in dataset["test"]]

    ds = DatasetDict({
        "train": Dataset.from_list(train_messages[:n]),
        "eval": Dataset.from_list(eval_messages[:int(n*0.1)])
    })

    ds.save_to_disk(output_path)
    return ds


def main(args):
    dataset = load_dataset("arbml/CIDAR")
    dataset = dataset['train'].train_test_split(test_size=0.2)
    ds = save_dataset(dataset, args.output_path, args.n)
    logger.info(f"Total training examples: {len(ds['train'])}")
    logger.info(f"Total eval examples: {len(ds['eval'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help="Data output path")
    parser.add_argument("--n", type=int, default=500, help="Number of training examples to sample")
    args = parser.parse_args()

    logging_config("processing.log")

    main(args)