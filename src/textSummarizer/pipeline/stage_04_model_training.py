from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_trainer import ModelTrainer
from textSummarizer.logging import logging

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # Initialize configuration manager to load the configuration
        config = ConfigurationManager()
        
        # Get the model trainer configuration
        model_trainer_config = config.get_model_trainer_config()
        
        # Initialize the ModelTrainer with the configuration
        model_trainer = ModelTrainer(config=model_trainer_config)
        
        # Start the training process
        model_trainer.train()
