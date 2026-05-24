import torch
import torch.nn as nn
from transformers import AutoTokenizer, DistilBertModel

class DistilBertMultiLabel(nn.Module):
    def __init__(self, num_classes=28):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

class MLService:
    def __init__(self, weights_path, labels):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.labels = labels
        self.model = DistilBertMultiLabel(num_classes=len(labels)).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        
    def predict(self, text, threshold=0.3):
        inputs = self.tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs["input_ids"], inputs["attention_mask"])
            probabilities = torch.sigmoid(outputs).cpu().tolist()[0]
        detected = [self.labels[i] for i, prob in enumerate(probabilities) if prob > threshold]
        return detected if detected else ["neutral"]