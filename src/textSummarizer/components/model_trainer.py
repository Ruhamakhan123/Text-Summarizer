from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
from textSummarizer.entity import ModelTrainerConfig
from datasets import load_from_disk
import torch
import os

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Determine if CUDA is available and set device accordingly
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model from the pretrained checkpoint
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        
        # Set up the data collator for Seq2Seq tasks
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_pegasus)

        # Load the dataset from the specified path
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # Training arguments configuration
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, 
            num_train_epochs=1,  # You can adjust this as needed
            warmup_steps=500, 
            per_device_train_batch_size=1, 
            per_device_eval_batch_size=1,
            weight_decay=0.01, 
            logging_steps=10,
            evaluation_strategy='steps', 
            eval_steps=500, 
            save_steps=500,  # Save model more frequently
            gradient_accumulation_steps=16
        ) 

        # Initialize the Trainer with the specified arguments, model, and datasets
        trainer = Trainer(
            model=model_pegasus, 
            args=trainer_args,
            tokenizer=tokenizer, 
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["test"],  # Use the training set
            eval_dataset=dataset_samsum_pt["validation"]  # Use the validation set
        )
        
        # Start training the model
        try:
            trainer.train()
        except Exception as e:

            print(f"Training failed with error: {e}")


        # Save the trained model and tokenizer
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
