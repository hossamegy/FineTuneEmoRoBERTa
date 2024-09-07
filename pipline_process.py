import re
import pandas as pd
import numpy as np
import pickle
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

class Piplines_Prepare_preprocessing_training():
    # Initialize with column names for input text and target emotion
    def __init__(self, input_column_name='Text', traget_column_name='Emotion'):
        self.input_column_name = input_column_name
        self.traget_column_name = traget_column_name

    # Load dataset from a CSV file
    def data_importer(self, path):
        dataset = pd.read_csv(path)

        print("="*50)
        print("step 1: The data loaded successfully :)")
        print("="*50)
        print(dataset.head(5))  # Print the first 5 rows of the dataset for inspection

        return dataset

    # Clean data by applying various text processing functions
    def data_cleaner(self, dataset):
        # Convert all text in the input column to lower case
        def to_lower_case(examples):
            return [text.lower() for text in examples[self.input_column_name]]

        # Remove special characters from the text
        def remove_special_characters(examples):
            return [re.sub(r'[^\w\s]', '', text) for text in examples[self.input_column_name]]

        # Remove all numbers from the text
        def remove_numbers(examples):
            return [re.sub(r'\d+', '', text) for text in examples[self.input_column_name]]

        # Remove emojis from the text
        def remove_emojis(examples):
            return [text.encode('ascii', 'ignore').decode('ascii') for text in examples[self.input_column_name]]

        # Remove extra spaces from the text
        def remove_extra_spaces(examples):
            return [' '.join(text.split()) for text in examples[self.input_column_name]]

        # Remove rare words that occur less than min_freq times
        def remove_rare_words(examples, min_freq=10):
            all_words = [word for text in examples[self.input_column_name] for word in text.split()]
            word_counts = Counter(all_words)
            filtered_texts = [
                ' '.join(word for word in text.split() if word_counts[word] >= min_freq)
                for text in examples[self.input_column_name]
            ]
            return filtered_texts

        # Remove empty strings from the dataset
        def remove_empty_strings(examples):
            non_empty_indices = [i for i, text in enumerate(examples[self.input_column_name]) if text.strip() != '']
            return examples.iloc[non_empty_indices].reset_index(drop=True)

        # Apply all text cleaning functions to the dataset
        dataset[self.input_column_name] = to_lower_case(dataset)
        dataset[self.input_column_name] = remove_special_characters(dataset)
        dataset[self.input_column_name] = remove_numbers(dataset)
        dataset[self.input_column_name] = remove_emojis(dataset)
        dataset[self.input_column_name] = remove_rare_words(dataset, min_freq=10)
        dataset[self.input_column_name] = remove_extra_spaces(dataset)
        dataset = remove_empty_strings(dataset)
        
        print("="*50)
        print("step 2: The data was cleaned successfully :)")
        print("="*50)
        print(dataset.head(5))  # Print the first 5 rows of the cleaned dataset

        return dataset

    # Tokenize data and prepare datasets for training, validation, and testing
    def data_preprocessor(self, dataset, tokenizer):
        # Tokenize the text and return the encoding
        def tokenize_function(examples):
            encoding = tokenizer(examples[self.input_column_name], max_length=100, padding="max_length", truncation=True)
            return {self.input_column_name: encoding['input_ids'], 'attention_mask': encoding['attention_mask']}
       
        # Encode target labels into numerical values
        le = LabelEncoder()
        dataset[self.traget_column_name] = le.fit_transform(dataset[self.traget_column_name])   
        with open(r"FineTuned_roberta\label_encoder_object.pkl", 'wb') as f:
            pickle.dump(le, f)  # Save the label encoder to a file

        # Split the dataset into training, validation, and test sets
        train_data, temp_data = train_test_split(dataset, test_size=0.05, shuffle=True)
        validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        # Convert pandas DataFrames to Hugging Face Datasets
        train_dataset = Dataset.from_pandas(train_data)
        validation_dataset = Dataset.from_pandas(validation_data)
        test_dataset = Dataset.from_pandas(test_data)

        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': validation_dataset,
            'test': test_dataset})
        
        dataset_dict = dataset_dict.remove_columns('__index_level_0__')  # Remove unnecessary columns
        
        # Tokenize datasets
        tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.rename_column(self.traget_column_name, "labels")
        tokenized_datasets = tokenized_datasets.rename_column(self.input_column_name, "input_ids")
        
        # Print a sample of the tokenized data for inspection
        print(tokenized_datasets['train'][5:10]['input_ids'])
        print(tokenized_datasets['train'][5:10]['labels'])
        print(tokenized_datasets['validation'][5:10]['input_ids'])
        print(tokenized_datasets['validation'][5:10]['labels'])
        print(tokenized_datasets['test'][5:10]['input_ids'])
        print(tokenized_datasets['test'][5:10]['labels'])

        print("="*52)
        print("step 3: Data preprocessing was done successfully :)")
        print("="*52)
        print(le.classes_)  # Print the classes in the label encoder
        print(dataset_dict)  # Print the dataset dictionary
        print(tokenized_datasets)  # Print the tokenized datasets
        return tokenized_datasets, le

    # Define training parameters and train the model
    def model_trainer(self, dataset, model):
        # Function to compute metrics for evaluation
        def compute_metrics_test(p):
            pred, labels = p
            pred = np.argmax(pred, axis=1)
            accuracy = accuracy_score(y_true=labels, y_pred=pred)
            recall = recall_score(y_true=labels, y_pred=pred, average='weighted')
            precision = precision_score(y_true=labels, y_pred=pred, average='weighted')
            f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')
            print(classification_report(labels, pred))
            return {"Test accuracy": accuracy,
                    "Test precision": precision,
                    "Test recall": recall,
                    "Test f1": f1}

        # Set up training arguments
        batch_size = 32
        model_name = r"H:\finetuned-emotion"
        training_args = TrainingArguments(
            output_dir=model_name,
            num_train_epochs = 4,
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            weight_decay=0.01,
            evaluation_strategy ='epoch',
            disable_tqdm=False
        )

        # Initialize Trainer with model, arguments, and datasets
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            compute_metrics=compute_metrics_test,
        )
        print("="*52)
        print("step 4: Start training Model :)")
        print("="*52)

        trainer.train()  # Start training the model
        print("model saved in:", model_name)
        print("="*52)
        print("Finish training Model :)")
        print("="*52)

        return trainer
    
    # Evaluate the model on the test dataset
    def model_evaluation(self, trainer, dataset):
        print("="*52)
        print("step 5: Start evaluation model :)")
        print("="*52)

        test_results = trainer.evaluate(eval_dataset=dataset['test'])

        print("-"*45)
        print("Test evaluation results:")
        print("-"*45)
        print(test_results)  # Print the results of the model evaluation
        print("="*52)

    # Run the entire pipeline from data import to model evaluation
    def run_pipline(self, df_path, model, tokenizer):
        dataset = self.data_importer(df_path)
        dataset = self.data_cleaner(dataset)
        dataset, _ = self.data_preprocessor(dataset, tokenizer)
        trainer = self.model_trainer(dataset, model)
        self.model_evaluation(trainer, dataset)
