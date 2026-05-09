import torch
from transformers import AutoTokenizer
from span_model import SpanDetector
from span_dataloader import get_span_dataloader
from tqdm import tqdm
import os
from sklearn.metrics import classification_report, f1_score
import numpy as np

dataset = 'PE'
PLM = 'roberta-base'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 20
lr = 2e-5
batch_size = 8
save_path = './models/saved/span_best.pt'



def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
            
            mask = attention_mask.bool()
            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())
    
    report = classification_report(all_labels, all_preds, target_names=['O', 'B-AC', 'I-AC'], digits=4)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return report, macro_f1

def train_span_model():
    tokenizer = AutoTokenizer.from_pretrained(PLM)
    train_loader = get_span_dataloader(f'./data/{dataset}/train.jsonl', tokenizer, batch_size)
    test_loader = get_span_dataloader(f'./data/{dataset}/test.jsonl', tokenizer, batch_size, shuffle=False)

    model = SpanDetector(PLM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_f1 = 0
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits, loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss/len(train_loader))
        
        # Validation
        print(f"\nEvaluating Epoch {epoch+1}...")
        report, macro_f1 = evaluate(model, test_loader, device)
        print(report)
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), save_path)
            print(f"Model saved (best macro F1: {best_f1:.4f})!")

    print("\n" + "--------" + " BEST MODEL EVALUATION " + "--------")
    
    final_model = SpanDetector(PLM).to(device)
    final_model.load_state_dict(torch.load(save_path, map_location=device, weights_only=False))
    
    report, macro_f1 = evaluate(final_model, test_loader, device)
    print("\nPerformance on test dataset (Best Model):\n", report)
    print(f"Final F1: {macro_f1:.4f}")

if __name__ == '__main__':
    train_span_model()
