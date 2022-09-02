import logging
import torch

# reference: https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings

# set up mult file loggers
# useful for logging in evaluation or inference
def setup_logger(name, log_file, formatter):
    """To setup as many loggers as you want"""
    if torch.distributed.get_world_size()>1:
        print("multiple gpus detected. Logging disabled."
        )
        level=logging.WARNING
        handler = logging.FileHandler(log_file)      
        handler.setFormatter(logging.Formatter(formatter))
        #handler = logging.NullHandler()
    else:
        level=logging.INFO
        handler = logging.FileHandler(log_file)      
        handler.setFormatter(logging.Formatter(formatter))   
    # --changed
    # level=logging.INFO
    # handler = logging.FileHandler(log_file)      
    # handler.setFormatter(logging.Formatter(formatter))  


    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger