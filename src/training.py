
from src.utils.common import read_config
from src.utils.logger import setup_logger
from src.utils.logger import get_timestamp
from src.utils.data_mgmt import get_data
from src.utils.model import create_model
from src.utils.model import save_model

import tensorflow as tf
import argparse
import logging
import os

def logger_config(config_path):
    config=read_config(config_path)
    logs_name = get_timestamp("generelLogs")
    log=setup_logger(config, file_name=logs_name)
    return log
    
    

def training(config_path):
    config= read_config(config_path)
    logging.info (config)
    
    
    # Data Management
    logging.info(">>>>>>>>>>>Data Loading started<<<<<<<<<<<<<<<<<<<")
    validation_datasize= config["params"]["VALIDATION_DATASIZE"]
    try:
        (X_train, y_train), (X_valid, y_valid),(X_test, y_test)= get_data(validation_datasize=validation_datasize)
    except Exception as e:
        logging.exception(e)
        raise e
    logging.info(">>>>>>>Data Loading finished<<<<<<<<<<<<<<<<<<")  
    
    # Creating and comiling model 
    logging.info(">>>>>>>Creating and Compiling model started <<<<<<<<<<<<<<<<<")
    LOSS_FUNCTION=config["params"]["LOSS_FUNCTION"]
    OPTIMIZER=config["params"]["OPTIMIZER"]
    METRICS=config["params"]["METRICS"]    
    try:
        model_clf=create_model(LOSS_FUNCTION=LOSS_FUNCTION,OPTIMIZER=OPTIMIZER,METRICS=METRICS)
    except Exception as e:
        logging.exception(e)
        raise e
    logging.info(">>>>>>>Creating and Compiling model finished <<<<<<<<<<<<<<<<<")
    
    # TRAINING MODELS
    EPOCHS=config["params"]["EPOCHS"]
    VALIDATION_SET=(X_valid, y_valid)
    try:
        logging.info(">>>>>>>Model training started <<<<<<<<<<<<<<<<<")
        history= model_clf.fit(x=X_train,y=y_train,epochs=EPOCHS,validation_data=VALIDATION_SET)
        logging.info(">>>>>>>Model training finished <<<<<<<<<<<<<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e
    
    
    # save the model at
    logging.info(">>> Saving models >>>>")
    artifacts_dir = config["artifacts"]["ARTIFACTS_DIR"]
    model_dir = config["artifacts"]["MODEL_DIR"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    model_name=config["artifacts"]["MODEL_NAME"]
    save_model(model_clf, model_name , model_dir= model_dir_path )

    logging.info(f">>> Model saved Location: {model_dir_path}>>>>")
    
    
    



# Main 
if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--config", "-c" , default="config.yaml")
    parsed_args = args.parse_args()
    logger_config (config_path=parsed_args.config)
    training(config_path=parsed_args.config)
    