import argparse
import os
from collections import Counter

import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from as_pipeline import (
    AS_LABELS,
    PLM,
    RelationStrengthDataset,
    RelationStrengthModel,
    build_training_samples,
    load_json_or_jsonl,
    make_collate_fn,
)


def evaluate(model, dataloader, id2label):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in dataloader:
            logits, _ = model(
                batch['encodings']['input_ids'],
                batch['encodings']['attention_mask']
            )
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            golds.extend(batch['labels'].cpu().tolist())
    print('macro_f1', f1_score(golds, preds, average='macro'))
    print(classification_report(golds, preds, target_names=[id2label[i] for i in range(len(id2label))], digits=3))
    return f1_score(golds, preds, average='macro')


def main():
    parser = argparse.ArgumentParser(description='Train second-stage AS pipeline classifier.')
    parser.add_argument('--dataset', choices=['PE', 'CDCP'], default='PE')
    parser.add_argument('--train', help='Training JSONL path')
    parser.add_argument('--test', help='Test JSONL path')
    parser.add_argument('--output', default='models/saved/as_pipeline.pt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_path = args.train or f'data/{args.dataset}/train.jsonl'
    test_path = args.test or f'data/{args.dataset}/test.jsonl'
    id2label = dict(enumerate(AS_LABELS))
    label2id = {label: idx for idx, label in id2label.items()}

    train_samples = build_training_samples(load_json_or_jsonl(train_path), label2id)
    test_samples = build_training_samples(load_json_or_jsonl(test_path), label2id)
    print('train samples:', len(train_samples), dict(Counter(sample['label'] for sample in train_samples)))
    print('test samples:', len(test_samples), dict(Counter(sample['label'] for sample in test_samples)))

    tokenizer = AutoTokenizer.from_pretrained(PLM)
    model = RelationStrengthModel(PLM, len(AS_LABELS)).to(device)
    collate_fn = make_collate_fn(tokenizer, device, args.max_len)
    train_loader = DataLoader(RelationStrengthDataset(train_samples), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(RelationStrengthDataset(test_samples), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_f1 = -1
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f'epoch {epoch}')
        for batch in train_bar:
            optimizer.zero_grad()
            _, loss = model(
                batch['encodings']['input_ids'],
                batch['encodings']['attention_mask'],
                batch['labels']
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_bar.set_postfix(loss=total_loss / max(len(train_bar), 1))
        score = evaluate(model, test_loader, id2label)
        if score > best_f1:
            best_f1 = score
            torch.save({
                'model_state': model.state_dict(),
                'plm': PLM,
                'labels': AS_LABELS,
                'dataset': args.dataset
            }, args.output)
            print('saved best AS pipeline model:', args.output)


if __name__ == '__main__':
    main()
