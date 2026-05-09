import torch
from torch.utils.data import Dataset, DataLoader
import json
import os

class SpanDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512): # 0: O, 1: B, 2: I
        self.data = []

        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample['text']
        spans = sample['ACs_span']
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        offsets = encoding['offset_mapping'].squeeze(0)
        
        labels = torch.zeros(self.max_len, dtype=torch.long)
        
        for start, end in spans:
            first_token = True
            for i, (token_start, token_end) in enumerate(offsets):
                if token_start == token_end == 0: continue
                

                if token_start >= start and token_end <= end + 1:
                    # Ignore whitespace or single punctuation
                    token_text = text[token_start:token_end]
                    if token_text.strip() == "" and first_token:
                        continue
                        
                    if first_token:
                        labels[i] = 1 # B
                        first_token = False
                    else:
                        labels[i] = 2 # I
                        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'offsets': offsets,
            'text': text
        }

def get_span_dataloader(data_path, tokenizer, batch_size, shuffle=True):
    dataset = SpanDataset(data_path, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
