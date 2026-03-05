"""Hot Dog / Not Hot Dog Inference CLI
Uses the trained ViT model from train.py (HuggingFace Transformers).
"""
import os, sys, torch, torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CLASSES = ["not_hot_dog", "hot_dog"]
EMOJI = {"hot_dog": "🌭", "not_hot_dog": "❌"}
MODEL_PATH = os.environ.get("MODEL_PATH", "model/best_model.pth")
MODEL_NAME = os.environ.get("MODEL_NAME", "google/vit-base-patch16-224")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
print(f"Loading model from {MODEL_PATH}...")
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()
print("🌭 Hot Dog / Not Hot Dog Classifier ready!\n")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image):
    """Predict on a PIL Image. Returns (label, probs_dict)."""
    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor).logits
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
    return CLASSES[pred_idx], {
        CLASSES[i]: f"{probs[i].item() * 100:.1f}%" for i in range(len(CLASSES))
    }


def predict_file(path):
    img = Image.open(path)
    label, probs = predict(img)
    print(f"{EMOJI[label]} {label} — {probs}")
    return label, probs


def predict_url(url):
    resp = requests.get(url, timeout=30)
    img = Image.open(BytesIO(resp.content))
    label, probs = predict(img)
    print(f"{EMOJI[label]} {label} — {probs}")
    return label, probs


def test_dataset(n=20):
    """Test on random samples from the Food-101 validation set."""
    from datasets import load_dataset
    import random

    ds = load_dataset("ethz/food101", split="validation", trust_remote_code=True)
    label_names = ds.features["label"].names
    hotdog_idx = label_names.index("hot_dog")

    indices = random.sample(range(len(ds)), min(n, len(ds)))
    correct = 0
    for i in indices:
        item = ds[i]
        img = item["image"].convert("RGB")
        true_binary = 1 if item["label"] == hotdog_idx else 0
        true_label = CLASSES[true_binary]
        pred_label, probs = predict(img)
        match = "✅" if pred_label == true_label else "❌"
        print(f"{match} True: {true_label:<14} Pred: {pred_label:<14} {probs}")
        if pred_label == true_label:
            correct += 1
    print(f"\nAccuracy: {correct}/{n} ({100 * correct / n:.0f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python predict.py <image_path>       # classify a local image")
        print("  python predict.py <url>               # classify from URL")
        print("  python predict.py --test [N]          # test on N random Food-101 samples")
        sys.exit(0)

    arg = sys.argv[1]
    if arg == "--test":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        test_dataset(n)
    elif arg.startswith("http"):
        predict_url(arg)
    else:
        predict_file(arg)
