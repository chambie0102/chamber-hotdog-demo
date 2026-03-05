# Chamber Hot Dog Demo 🌭

Hot Dog / Not Hot Dog classification trained on Chamber GPU cluster.
Inspired by Silicon Valley S4E4.

## What it does
Classifies food images as **HOT DOG** or **NOT HOT DOG** using a fine-tuned ViT-B/16 model (HuggingFace Transformers).

## Pipeline
1. Code → GitHub → GitHub Actions builds Docker image (linux/amd64)
2. Image pushed to Docker Hub (`chambie0102/chamber-hotdog-demo`)
3. Job submitted to Chamber GPU cluster via `chamber-agent` CLI
4. Training runs on Tesla T4, metrics logged to W&B
5. Autonomous agent iterates on hyperparameters until target accuracy

## Dataset
- **Source:** [ethz/food101](https://huggingface.co/datasets/ethz/food101) — `hot_dog` class vs sampled non-hot-dog classes
- **Balanced:** Up to 2,500 samples per class (configurable via `MAX_SAMPLES_PER_CLASS`)
- **Classes:** hot_dog, not_hot_dog

## W&B Project
[chamber-hotdog-demo](https://wandb.ai/jasonong-chamberai/chamber-hotdog-demo)

## Files
| File | Purpose |
|------|---------|
| `train.py` | Training script — ViT-B/16, Food-101, full W&B logging |
| `predict.py` | CLI inference (file, URL, or Food-101 test samples) |
| `demo/app.py` | Flask web demo (port 5051) |
| `job.yaml` | K8s manifest for Chamber cluster |
| `Dockerfile` | Container build |
| `.github/workflows/docker-publish.yml` | CI/CD |

## Iteration Strategy
Start with intentionally non-optimal params to show the iteration story:

| Run | Change | Expected Outcome |
|-----|--------|------------------|
| v1 | batch=64, lr=0.001, no aug, no dropout | Baseline — may plateau ~80% |
| v2 | lr=3e-4, add class weights | Better convergence |
| v3 | lr=1e-4, warmup=2, basic augmentation | ~85-90% |
| v4 | aggressive aug, dropout=0.2 | ~90%+ |
| v5 | lr=5e-5, epochs=15, fine-tune | Target: ≥90% both classes |

## Quick Start

### Train locally (CPU/GPU)
```bash
pip install -r requirements.txt
EPOCHS=3 python train.py
```

### Submit to Chamber
```bash
chamber-agent jobs submit --manifest job.yaml --team "Recommendation Service"
```

### Run demo app
```bash
pip install flask
cd demo && MODEL_PATH=../model/best_model.pth python app.py
# Open http://localhost:5051
```

### CLI inference
```bash
python predict.py photo.jpg              # local file
python predict.py https://example.com/img.jpg  # URL
python predict.py --test 20              # test on Food-101 samples
```
