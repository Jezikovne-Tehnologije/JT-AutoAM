# AS Pipeline Experiment

This branch tests argument strength as a second-stage model instead of a parallel AutoAM head.

## Pipeline

```text
raw text + ACs_span
-> AutoAM predicts ACs and ARs
-> AS pipeline model reads AutoAM output
-> AS pipeline model predicts strength for each AR edge
```

## AS Model Input

The AS model expects AutoAM-style output:

```json
{
  "id": "example001",
  "text": "Full essay text...",
  "ACs_span": [[10, 50], [80, 140]],
  "ACs": ["Claim", "Premise"],
  "ARs": {
    "support": [[1, 0]],
    "attack": []
  }
}
```

Training data also includes gold `AS`:

```json
{
  "AS": {
    "very weak": [],
    "weak": [],
    "medium": [[1, 0]],
    "strong": [],
    "very strong": []
  }
}
```

## Train

```bash
python train_as_pipeline.py --dataset PE --output models/saved/as_pipeline.pt
```

The script splits the training set internally:

```text
85% train / 15% validation
```

Every 5 epochs it evaluates on the validation split and saves the best validation checkpoint. The original test file is only used at the end as the final test check.

You can change the validation frequency:

```bash
python train_as_pipeline.py --dataset PE --val_every 2
```

For CDCP:

```bash
python train_as_pipeline.py --dataset CDCP --output models/saved/as_pipeline_cdcp.pt
```

## Predict

```bash
python predict_as_pipeline.py --input autoam_output.json --model models/saved/as_pipeline.pt
```

Optional:

```bash
python predict_as_pipeline.py --input autoam_output.json --model models/saved/as_pipeline.pt --output with_as.json
```

## Expected Output

The script returns the original AutoAM output plus:

```json
{
  "AS": {
    "very weak": [],
    "weak": [],
    "medium": [[1, 0]],
    "strong": [],
    "very strong": []
  }
}
```
