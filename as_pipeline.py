import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel


PLM = 'roberta-base'
AS_LABELS = ['very weak', 'weak', 'medium', 'strong', 'very strong']


def load_json_or_jsonl(path):
    with open(path, encoding='utf-8') as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def find_relation_type(ARs, edge):
    for relation_type, edges in ARs.items():
        if edge in edges:
            return relation_type
    return None


def build_as_lookup(AS):
    lookup = {}
    for strength, edges in AS.items():
        for edge in edges:
            lookup[tuple(edge)] = strength
    return lookup


def component_text(example, component_id):
    start, end = example['ACs_span'][component_id]
    return example['text'][start:end + 1]


def make_relation_input(example, edge, relation_type):
    source, target = edge
    source_type = example['ACs'][source]
    target_type = example['ACs'][target]
    source_text = component_text(example, source).replace('\n', ' ')
    target_text = component_text(example, target).replace('\n', ' ')
    return (
        f'relation: {relation_type}. '
        f'source type: {source_type}. target type: {target_type}. '
        f'source argument: {source_text}. '
        f'target argument: {target_text}.'
    )


def build_training_samples(examples, label2id):
    samples = []
    for example in examples:
        if 'AS' not in example:
            continue
        as_lookup = build_as_lookup(example['AS'])
        for edge_tuple, strength in as_lookup.items():
            edge = list(edge_tuple)
            relation_type = find_relation_type(example['ARs'], edge)
            if relation_type == None:
                continue
            samples.append({
                'text': make_relation_input(example, edge, relation_type),
                'label': label2id[strength],
                'edge': edge,
                'id': example.get('id')
            })
    return samples


def build_prediction_samples(example):
    samples = []
    for relation_type, edges in example['ARs'].items():
        if relation_type == 'none':
            continue
        for edge in edges:
            samples.append({
                'text': make_relation_input(example, edge, relation_type),
                'edge': edge,
                'relation_type': relation_type,
                'id': example.get('id')
            })
    return samples


class RelationStrengthDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def make_collate_fn(tokenizer, device, max_len):
    def collate(batch):
        encodings = tokenizer(
            [item['text'] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        ).to(device)
        result = {'encodings': encodings, 'items': batch}
        if 'label' in batch[0]:
            result['labels'] = torch.tensor([item['label'] for item in batch], device=device)
        return result
    return collate


class RelationStrengthModel(nn.Module):
    def __init__(self, plm=PLM, num_labels=len(AS_LABELS), dropout=0.3):
        super().__init__()
        self.plm = plm
        self.encoder = AutoModel.from_pretrained(plm)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = outputs['last_hidden_state'][:, 0, :]
        logits = self.classifier(pooled)
        loss = None
        if labels != None:
            loss = nn.functional.cross_entropy(logits, labels)
        return logits, loss


def group_strength_predictions(items, pred_ids, id2label):
    grouped = {label: [] for label in id2label.values()}
    for item, pred_id in zip(items, pred_ids):
        grouped[id2label[int(pred_id)]].append(item['edge'])
    return grouped
