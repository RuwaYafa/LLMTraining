import sys
import argparse
import logging
import subprocess
from importlib import metadata

logger = logging.getLogger(__name__)


def save_package_versions(output_file):
    logger.info("pip freeze:")
    with open(output_file, "w") as fh:
        dists = {dist.metadata["Name"]: dist.version for dist in metadata.distributions()}
        logger.info(dists)
        fh.write("\n".join([f"{d}=={v}" for d, v in dists.items()]))

    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True).stdout
    logger.info(f"nvcc --version: {str(nvcc)}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
def merge_object_properties(train_args, data_args, model_args, lora_args):
    """
    Used to copy properties from one object to another if there isn't a naming conflict;
    """
    for property in train_args.__dict__:
        #Check to make sure it can't be called... ie a method.
        #Also make sure the object objectToMergeTo doesn't have a property of the same name.
        if not callable(train_args.__dict__[property]) and not hasattr(data_args, property):
            setattr(data_args, property, getattr(train_args, property))
    
    for property in model_args.__dict__:
        #Check to make sure it can't be called... ie a method.
        #Also make sure the objectobjectToMergeTo doesn't have a property of the same name.
        if not callable(model_args.__dict__[property]) and not hasattr(data_args, property):
            setattr(data_args, property, getattr(model_args, property))
            
    for property in lora_args.__dict__:
        #Check to make sure it can't be called... ie a method.
        #Also make sure the objectobjectToMergeTo doesn't have a property of the same name.
        if not callable(lora_args.__dict__[property]) and not hasattr(data_args, property):
            setattr(data_args, property, getattr(lora_args, property))

    return data_args


def logging_config(log_file=None):
    """
    Initialize logger configuration
    :param log_file: Path - log to this file
    :return: None
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file, "w", "utf-8"))
        print("Logging to {}".format(log_file))

    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        format="%(levelname)s\t%(name)s\t%(asctime)s\t%(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
    )
