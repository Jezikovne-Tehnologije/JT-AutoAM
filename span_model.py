import torch
import torch.nn as nn
from transformers import AutoModel

class SpanDetector(nn.Module):
    def __init__(self, plm_name, num_labels=3, class_weights=[10.0, 10.0, 1.0]): # 0: O, 1: B, 2: I
        super().__init__()
        self.bert = AutoModel.from_pretrained(plm_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.register_buffer('weights', torch.tensor(class_weights))
        self.criterion = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss = self.criterion(logits.view(-1, 3), labels.view(-1))
            return logits, loss
        return logits

def extract_spans_from_bio(labels, offsets):
    """
    Extract [start, end] character spans from BIO labels and offsets.
    labels: list of label IDs (0: O, 1: B, 2: I)
    offsets: list of (start, end) tuples from tokenizer
    """
    spans = []
    current_span = None
    
    for label, offset in zip(labels, offsets):
        start, end = offset
        if start == end == 0: continue
        
        if label == 1: # B
            if current_span:
                spans.append(current_span)
            current_span = [start, end - 1] # Inclusive end
        elif label == 2: # I
            if current_span:
                current_span[1] = end - 1
            else:
                current_span = [start, end - 1]
        else: # O
            if current_span:
                spans.append(current_span)
                current_span = None
    
    if current_span:
        spans.append(current_span)
        
    return spans
