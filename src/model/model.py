import torch
from torch import nn
from transformers import BertModel

class NewsClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(NewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)  # استخدم الإخراج من pooler
        return logits

def create_model(num_labels=2):
    model = NewsClassifier(num_labels=num_labels)
    return model
