"""
ProjectD: Synthetic Data Generator & CPU Fine-Tuning Pipeline
Trains a sequence classification head on microsoft/deberta-v3-small.
"""

import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup

# ==========================================
# 1. SYNTHETIC DATASET GENERATION
# ==========================================

SYNTHETIC_CORPUS = [
    # 0: NONE (Satisfied / Natural Progression)
    ("Here is your tracking link: https://track.co/123", "Perfect, thank you so much!", 0),
    ("I have updated your shipping address to 123 Elm St.", "Got it, that works.", 0),
    ("The sum of your order is $45.20 including taxes.", "Thanks for clarifying.", 0),
    ("Here is the Python script using standard json library.", "Awesome, exactly what I needed.", 0),
    ("Your flight is confirmed for 6:00 PM tomorrow.", "Great, appreciate your help!", 0),
    ("The refund of $34.00 has been credited to your wallet.", "Thank you!", 0),
    ("Here is the documentation link for the API authentication.", "This makes sense, thanks.", 0),
    ("The package was marked delivered at your front porch.", "Found it, thanks!", 0),

    # 1: REPHRASE (User repeats request with different wording)
    ("Our standard shipping takes 3-5 business days.", "When will my package actually arrive?", 1),
    ("You can reset your password from the account security page.", "How do I change my login password?", 1),
    ("The capital of Australia is Canberra.", "What city is Australia's capital?", 1),
    ("We accept VISA, Mastercard, and American Express.", "Can I pay using my credit card?", 1),
    ("Use `git pull origin main` to sync changes.", "How to update local branch with remote main?", 1),
    ("Subscription costs $15 per month billed annually.", "What is the monthly price of the plan?", 1),
    ("Check the tracking number in your confirmation email.", "Where can I view the tracking number?", 1),

    # 2: CORRECTION (User points out an error or unwanted action)
    ("I have initiated a full cancellation and refund of your ticket.", "Wait, don't cancel it! I just wanted to change seats.", 2),
    ("Here is the solution using Pandas and NumPy.", "I told you no third-party libraries, pure Python only.", 2),
    ("Your appointment is scheduled for Wednesday at 10 AM.", "No, I said Thursday afternoon, not Wednesday.", 2),
    ("I have charged the remaining balance to your default card.", "Stop, that's the wrong card!", 2),
    ("Here is the guide for deploying to AWS EC2.", "We are running on Google Cloud Platform, not AWS.", 2),
    ("I removed all duplicate entries from table A.", "That's not what I asked, table B has the duplicates.", 2),
    ("The discount code expired yesterday.", "No it didn't, the email said valid through the weekend.", 2),

    # 3: ABANDONMENT (User gives up or exits in frustration)
    ("Let's troubleshoot your router settings starting with step 1.", "Never mind, I'll figure it out myself.", 3),
    ("Can you provide your order number, zip code, and billing phone?", "Forget it, too much hassle.", 3),
    ("Please hold while I check the server diagnostics.", "Nevermind, cancelling the request.", 3),
    ("Would you like to try clearing your browser cache?", "Leave it, this is useless.", 3),
    ("Let me transfer you to an automated phone system.", "Forget about it, bye.", 3),
    ("I could not locate your receipt. Could you re-upload?", "Just cancel the whole thing.", 3),

    # 4: DRIFT (User abruptly changes topic due to unhelpful answer)
    ("To bake sourdough, maintain your starter hydration at 100%.", "What is the weather in Seattle tomorrow?", 4),
    ("The server error code 502 indicates a bad gateway.", "Can you write a poem about the ocean?", 4),
    ("Here is how you replace the brake pads on a 2018 sedan.", "Who won the World Cup in 2022?", 4),
    ("Check DNS propagation using the dig command.", "Recommend a good coffee shop in downtown.", 4),
    ("Your invoice has been exported to PDF format.", "Translate 'hello world' to Japanese.", 4),
]


class PairwiseFeedbackDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        resp, follow, label = self.data[idx]
        encoding = self.tokenizer(
            resp,
            follow,
            truncation="longest_first",
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# ==========================================
# 2. TRAINING ROUTINE (HEAD-ONLY ON CPU)
# ==========================================

def train():
    model_id = "microsoft/deberta-v3-small"
    save_dir = "./deberta-feedback-finetuned"
    os.makedirs(save_dir, exist_ok=True)

    print(f"[*] Initializing Tokenizer & Model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    id2label = {
        0: "NONE",
        1: "REPHRASE",
        2: "CORRECTION",
        3: "ABANDONMENT",
        4: "DRIFT"
    }
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=5,
        id2label=id2label,
        label2id=label2id
    )

    # Freeze DeBERTa backbone: only train pooler & classification head
    print("[*] Freezing backbone layers for fast CPU training...")
    for name, param in model.named_parameters():
        if "classifier" not in name and "pooler" not in name:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Trainable parameters: {trainable_params:,}")

    dataset = PairwiseFeedbackDataset(SYNTHETIC_CORPUS, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-3,
        weight_decay=0.01
    )
    epochs = 8
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("\n[*] Commencing Fine-Tuning...")
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        acc = correct / total
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc*100:.1f}%")

    print(f"\n[*] Saving fine-tuned checkpoint to: {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("[✓] Model fine-tuning complete!")


if __name__ == "__main__":
    train()