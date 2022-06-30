import logging
import torch

# reference: https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings

# set up mult file loggers
# useful for logging in evaluation or inference
def setup_logger(name, log_file, formatter,level=logging.INFO):
    """To setup as many loggers as you want"""
    if torch.distributed.get_world_size()>1:
        handler = logging.NullHandler()
    else:
        handler = logging.FileHandler(log_file)        
    handler.setFormatter(logging.Formatter(formatter))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger