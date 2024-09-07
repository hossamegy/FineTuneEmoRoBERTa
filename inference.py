import torch
from transformers import AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fineTuned_model_path="" # add your fine tuned model 

fineTuned_model = AutoModelForSequenceClassification.from_pretrained(fineTuned_model_path).to(device)

input_text = ""

with torch.no_grad():
    outputs = fineTuned_model(**input_text)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()