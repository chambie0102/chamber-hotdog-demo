"""
Hot Dog / Not Hot Dog Classifier
Fine-tunes ViT-B/16 on Food-101 hot dog subset
All hyperparams configurable via env vars for iteration
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import wandb
from sklearn.metrics import confusion_matrix, classification_report

# ── Hyperparameters (all via env vars) ──────────────────────────
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
DROPOUT = float(os.environ.get("DROPOUT", "0.0"))
USE_CLASS_WEIGHTS = os.environ.get("USE_CLASS_WEIGHTS", "false").lower() == "true"
AUGMENTATION = os.environ.get("AUGMENTATION", "none")  # none, basic, aggressive
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0"))
SEED = int(os.environ.get("SEED", "42"))
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "chamber-hotdog-demo")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "jasonong-chamberai")
MODEL_NAME = os.environ.get("MODEL_NAME", "google/vit-base-patch16-224")
MAX_SAMPLES_PER_CLASS = int(os.environ.get("MAX_SAMPLES_PER_CLASS", "2500"))

# ── Reproducibility ─────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🌭 Hot Dog Classifier | Device: {device}")
print(f"   Batch={BATCH_SIZE}, LR={LEARNING_RATE}, Epochs={EPOCHS}, Dropout={DROPOUT}")
print(f"   ClassWeights={USE_CLASS_WEIGHTS}, Aug={AUGMENTATION}, Warmup={WARMUP_EPOCHS}")
print(f"   WeightDecay={WEIGHT_DECAY}, MaxSamples={MAX_SAMPLES_PER_CLASS}")


# ── Dataset ─────────────────────────────────────────────────────
class HotDogDataset(Dataset):
    """Binary dataset: hot_dog (1) vs not_hot_dog (0) from Food-101"""

    def __init__(self, split="train", transform=None, max_per_class=2500):
        from datasets import load_dataset

        print(f"Loading Food-101 {split} split...")
        ds = load_dataset("ethz/food101", split=split, trust_remote_code=True)

        # Food-101 label names
        label_names = ds.features["label"].names
        hotdog_idx = label_names.index("hot_dog")

        # Collect indices
        hotdog_indices = []
        other_indices = []
        for i, label in enumerate(ds["label"]):
            if label == hotdog_idx:
                hotdog_indices.append(i)
            else:
                other_indices.append(i)

        # Balance: sample equal number of non-hotdog images
        n_hotdog = min(len(hotdog_indices), max_per_class)
        n_other = min(len(other_indices), max_per_class)

        random.shuffle(hotdog_indices)
        random.shuffle(other_indices)
        hotdog_indices = hotdog_indices[:n_hotdog]
        other_indices = other_indices[:n_other]

        self.indices = hotdog_indices + other_indices
        self.labels = [1] * len(hotdog_indices) + [0] * len(other_indices)
        self.ds = ds
        self.hotdog_idx = hotdog_idx
        self.transform = transform

        # Shuffle together
        combined = list(zip(self.indices, self.labels))
        random.shuffle(combined)
        self.indices, self.labels = zip(*combined)
        self.indices = list(self.indices)
        self.labels = list(self.labels)

        print(f"  {split}: {len(hotdog_indices)} hot dogs + {len(other_indices)} not hot dogs = {len(self.indices)} total")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.ds[self.indices[idx]]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(split, augmentation="none"):
    """Get transforms based on split and augmentation level"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if split == "train":
        aug_transforms = []
        if augmentation in ("basic", "aggressive"):
            aug_transforms.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        if augmentation == "aggressive":
            aug_transforms.extend([
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomErasing(p=0.2),
            ])

        base = [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        ]
        post = [transforms.ToTensor(), normalize]

        # RandomErasing must come after ToTensor
        if augmentation == "aggressive":
            erasing = [t for t in aug_transforms if isinstance(t, transforms.RandomErasing)]
            aug_no_erasing = [t for t in aug_transforms if not isinstance(t, transforms.RandomErasing)]
            return transforms.Compose(base + aug_no_erasing + post + erasing)
        else:
            return transforms.Compose(base + aug_transforms + post)
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


# ── Model ───────────────────────────────────────────────────────
def build_model(dropout=0.0):
    from transformers import ViTForImageClassification

    print(f"Loading {MODEL_NAME} pretrained...")
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    # Replace classifier head with dropout
    if dropout > 0:
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 2),
        )

    return model.to(device)


# ── Training ────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scheduler=None, epoch=0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if scheduler and epoch >= WARMUP_EPOCHS:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = 100.0 * correct / total
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Per-class metrics
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    not_hotdog_acc = 100.0 * cm[0, 0] / max(cm[0].sum(), 1)
    hotdog_acc = 100.0 * cm[1, 1] / max(cm[1].sum(), 1)

    report = classification_report(all_labels, all_preds,
                                   target_names=["not_hot_dog", "hot_dog"],
                                   output_dict=True)

    return {
        "loss": total_loss / total,
        "accuracy": acc,
        "not_hotdog_accuracy": not_hotdog_acc,
        "hotdog_accuracy": hotdog_acc,
        "sensitivity": hotdog_acc,  # recall for hot dog class
        "specificity": not_hotdog_acc,  # recall for not-hot-dog class
        "confusion_matrix": cm.tolist(),
        "report": report,
        "predictions": all_preds,
        "labels": all_labels,
    }


# ── Main ────────────────────────────────────────────────────────
def main():
    # Init W&B
    config = {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "dropout": DROPOUT,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "augmentation": AUGMENTATION,
        "warmup_epochs": WARMUP_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "model": MODEL_NAME,
        "max_samples_per_class": MAX_SAMPLES_PER_CLASS,
        "device": str(device),
    }

    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config=config)
    print(f"\n🏃 W&B run: {wandb.run.name} ({wandb.run.url})")

    # Data
    train_transform = get_transforms("train", AUGMENTATION)
    val_transform = get_transforms("val")

    train_dataset = HotDogDataset("train", transform=train_transform, max_per_class=MAX_SAMPLES_PER_CLASS)
    test_dataset = HotDogDataset("validation", transform=val_transform, max_per_class=MAX_SAMPLES_PER_CLASS)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = build_model(DROPOUT)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss with optional class weights
    if USE_CLASS_WEIGHTS:
        n_hotdog = sum(1 for l in train_dataset.labels if l == 1)
        n_other = sum(1 for l in train_dataset.labels if l == 0)
        weight = torch.tensor([n_hotdog / (n_hotdog + n_other),
                               n_other / (n_hotdog + n_other)]).to(device)
        print(f"Class weights: not_hot_dog={weight[0]:.3f}, hot_dog={weight[1]:.3f}")
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)

    # Scheduler
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * WARMUP_EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps) if WARMUP_EPOCHS > 0 else None

    # Training loop
    best_acc = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")

        # Warmup LR
        if epoch < WARMUP_EPOCHS:
            warmup_lr = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
            print(f"  Warmup LR: {warmup_lr:.6f}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer,
                                                 criterion, scheduler, epoch)

        # Evaluate
        metrics = evaluate(model, test_loader, criterion)
        elapsed = (time.time() - start_time) / 60

        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {metrics['loss']:.4f} | Test Acc:  {metrics['accuracy']:.2f}%")
        print(f"  🌭 Hot Dog Acc: {metrics['hotdog_accuracy']:.2f}%")
        print(f"  🚫 Not Hot Dog Acc: {metrics['not_hotdog_accuracy']:.2f}%")
        print(f"  ⏱️  Elapsed: {elapsed:.1f} min")

        # Log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": metrics["loss"],
            "test_accuracy": metrics["accuracy"],
            "hotdog_accuracy": metrics["hotdog_accuracy"],
            "not_hotdog_accuracy": metrics["not_hotdog_accuracy"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": elapsed,
        })

        # Log confusion matrix
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=metrics["labels"].tolist(),
                preds=metrics["predictions"].tolist(),
                class_names=["not_hot_dog", "hot_dog"],
            )
        })

        # Save best model
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")
            print(f"  💾 New best model saved! ({best_acc:.2f}%)")

    # Final evaluation with best model
    print(f"\n{'='*60}")
    print("Final Evaluation with Best Model")
    print(f"{'='*60}")

    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    final = evaluate(model, test_loader, criterion)

    print(f"\n🏆 FINAL RESULTS:")
    print(f"   Test Accuracy:      {final['accuracy']:.2f}%")
    print(f"   🌭 Hot Dog Acc:     {final['hotdog_accuracy']:.2f}%")
    print(f"   🚫 Not Hot Dog Acc: {final['not_hotdog_accuracy']:.2f}%")
    print(f"   Confusion Matrix:   {final['confusion_matrix']}")

    # Log final metrics
    wandb.summary["test_accuracy"] = final["accuracy"]
    wandb.summary["hotdog_accuracy"] = final["hotdog_accuracy"]
    wandb.summary["not_hotdog_accuracy"] = final["not_hotdog_accuracy"]
    wandb.summary["sensitivity"] = final["sensitivity"]
    wandb.summary["specificity"] = final["specificity"]
    wandb.summary["normal_accuracy"] = final["not_hotdog_accuracy"]
    wandb.summary["pneumonia_accuracy"] = final["hotdog_accuracy"]  # for cron compat
    wandb.summary["best_accuracy"] = best_acc
    wandb.summary["total_time_minutes"] = (time.time() - start_time) / 60

    # Check if target hit
    if final["hotdog_accuracy"] >= 90 and final["not_hotdog_accuracy"] >= 90:
        print("\n🎯 TARGET HIT! Both classes ≥ 90%!")
    else:
        print("\n⚠️ Target not yet hit. Need both classes ≥ 90%.")

    wandb.finish()
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
