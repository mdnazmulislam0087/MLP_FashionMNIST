import logging
import os
import time

def get_timestamp(name):  #<< best it is
    timestamp = time.asctime().replace(" ", "_").replace(":", "_")
    unique_name = f"{name}_at_{timestamp}.log"
    return unique_name


def setup_logger(config, file_name=None):
    
    """Generel logging function

    Returns:
        logger file: Gives logger file in the desired directory
    """
    logs_dir = config["logs"]
    generel_logs_dir_path = os.path.join(logs_dir["LOGS_DIR"], logs_dir["GENERAL_LOGS_DIR"])
    os.makedirs(generel_logs_dir_path, exist_ok=True)
    
    logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
    logger=logging.basicConfig(filename=os.path.join(generel_logs_dir_path, file_name), level=logging.INFO, format=logging_str,
                    filemode="a")
    
    return logger