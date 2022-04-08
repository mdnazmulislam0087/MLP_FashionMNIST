from src.utils.common import read_config
from src.utils.logger import setup_logger
from src.utils.logger import get_timestamp


import argparse
import logging


def logger_config(config_path):
    config=read_config(config_path)
    logs_name = get_timestamp("generelLogs")
    log=setup_logger(config, file_name=logs_name)
    return log
    
    

def training(config_path):
    config= read_config(config_path)
    logging.info (config)
    



# Main 
if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--config", "-c" , default="config.yaml")
    parsed_args = args.parse_args()
    logger_config (config_path=parsed_args.config)
    training(config_path=parsed_args.config)
    