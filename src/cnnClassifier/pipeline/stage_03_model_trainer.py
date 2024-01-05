from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import Training
from cnnClassifier import logger
import shutil
import os


STAGE_NAME ='Training'

class ModelTrainingPipeline:
    def __init__(self):
        pass 
    
    def main(self):
        config=ConfigurationManager()
        training_config=config.get_training_config()
        training=Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()
        destination_directory="model"
        file_path=r"artifacts/training/model.h5"
        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)
        shutil.copy(file_path, destination_directory)
        print("Model copied to Model directory")
        

if __name__ =='__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started<<<<<<<")
        obj=ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed successfully <<<<<< \n\n x====================x")
    except Exception as e:
        logger.exception(e)
        raise e    