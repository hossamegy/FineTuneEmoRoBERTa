import torch
from pipline_process import Piplines_Prepare_preprocessing_training
from transformers import RobertaTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_name = "roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6).to(device)
tokenizer = RobertaTokenizer.from_pretrained(model_name)
 
Piplines_Prepare_preprocessing_training(
    input_column_name='Text', 
    traget_column_name='Emotion'
).run_pipline(r"dataset\Emotion_final.csv", model, tokenizer)                                       