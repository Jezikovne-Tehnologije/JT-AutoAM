import argparse
import json

import torch
from transformers import AutoTokenizer

from as_pipeline import (
    RelationStrengthDataset,
    RelationStrengthModel,
    build_prediction_samples,
    group_strength_predictions,
    load_json_or_jsonl,
    make_collate_fn,
)


def main():
    parser = argparse.ArgumentParser(description='Add AS predictions to AutoAM output JSON.')
    parser.add_argument('--input', required=True, help='JSON/JSONL with text, ACs_span, ACs, and ARs')
    parser.add_argument('--model', default='models/saved/as_pipeline.pt')
    parser.add_argument('--output')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    labels = checkpoint['labels']
    id2label = dict(enumerate(labels))
    model = RelationStrengthModel(
        checkpoint['plm'],
        len(labels),
        distance_loss_weight=checkpoint.get('distance_loss_weight', 0.2)
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint['plm'])
    collate_fn = make_collate_fn(tokenizer, device, args.max_len)

    outputs = []
    for example in load_json_or_jsonl(args.input):
        samples = build_prediction_samples(example)
        dataloader = torch.utils.data.DataLoader(
            RelationStrengthDataset(samples),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        pred_ids = []
        items = []
        with torch.no_grad():
            for batch in dataloader:
                logits, _ = model(
                    batch['encodings']['input_ids'],
                    batch['encodings']['attention_mask']
                )
                pred_ids.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                items.extend(batch['items'])
        result = dict(example)
        result['AS'] = group_strength_predictions(items, pred_ids, id2label)
        outputs.append(result)

    text = json.dumps(outputs[0] if len(outputs) == 1 else outputs, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
    else:
        print(text)


if __name__ == '__main__':
    main()
