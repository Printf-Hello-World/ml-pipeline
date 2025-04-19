from src.data_ingestion import DataIngestor  
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelManager
from src.evaluation import Evaluator
import pandas as pd
from src.utils.utils import logger
from src.utils.config import ConfigReader
import os

#get all the configs
config = ConfigReader()
data_ingestion_config = config.get("data_ingestion_config")
preprocess_config = config.get("preprocess_config")
model_initialisation_config = config.get("model_initialisation_config")
training_method = config.get("training_method")
param_grid = config.get("param_grid")
mlflow_uri = config.get("mlflow_uri")


process_name = 'Data Ingestion'
try:
    logger.info(f"{process_name} in progress") 
    data_ingestor = DataIngestor(**data_ingestion_config)
    data = data_ingestor.ingest()
    logger.info(f"{process_name} done!")
except Exception as e:
        raise e


process_name = 'Data Preprocessing'
try:
    logger.info(f"{process_name} in progress") 
    data_processor = DataPreprocessor(data, **preprocess_config)
    X_train, y_train, X_test, y_test= data_processor.process_data_split()
    logger.info(f"{process_name} done!")


except Exception as e:
        raise e


if model_initialisation_config['model_path']:
    process_name = 'Load Model'
    model = ModelManager(**model_initialisation_config)
    logger.info(f"{process_name} done!")
else:
    process_name = 'Model Training'
    try:
        logger.info("Initialising Model") 
        model = ModelManager(**model_initialisation_config)
        logger.info(f"{process_name} in progress") 
        if training_method == "gridsearchCV":
            model.grid_search_cv(X_train, y_train, param_grid, cv = 2)
        elif training_method == "fit":
            model.fit(X_train, y_train)
        logger.info(f"{process_name} done!")

    except Exception as e:
            raise e


process_name = 'Evaluation'
try:
    logger.info(f"{process_name} in progress") 
    evaluator = Evaluator(model, X_test, y_test, mlflow_uri)
    evaluator.evaluate()
    logger.info(f"{process_name} done!")

except Exception as e:
        raise e

