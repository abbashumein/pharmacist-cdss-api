import torch
import torch.nn as nn
from transformers import AutoTokenizer, DistilBertModel
from app.core.config import settings, EMOTION_LABELS
import logging

logger = logging.getLogger(__name__)

class DistilBertMultiLabel(nn.Module):
    def __init__(self, num_labels: int = 28):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))

class MLService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertMultiLabel(num_labels=len(EMOTION_LABELS)).to(self.device)
        self.model.load_state_dict(
            torch.load(settings.MODEL_PATH, map_location=self.device)
        )
        self.model.eval()
        logger.info(f"MLService ready on {self.device}")

    def predict(self, text: str, threshold: float = 0.3) -> list[str]:
        inputs = self.tokenizer(
            text, max_length=128, padding="max_length",
            truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.sigmoid(logits).cpu().tolist()[0]
        detected = [EMOTION_LABELS[i] for i, p in enumerate(probs) if p > threshold]
        return detected if detected else ["neutral"]
