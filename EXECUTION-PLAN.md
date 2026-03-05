# Hot Dog / Not Hot Dog Training — Execution Plan

## Phase 1: Build & First Run (30 min)

### Step 1: train.py ✅
- HuggingFace Transformers `ViTForImageClassification`
- Dataset: `ethz/food101` — hot_dog class vs balanced sample of other classes
- All hyperparams via env vars
- Full W&B logging: per-class accuracy, confusion matrix, sensitivity/specificity
- Start with INTENTIONALLY non-optimal params:
  - batch_size=64, lr=0.001, no augmentation, no dropout, no warmup

### Step 2: Dockerfile ✅
- pytorch:2.1.0-cuda11.8 base + transformers + datasets

### Step 3: job.yaml ✅
- Tesla-T4, tolerations + nodeSelector + GPU limit (all three!)
- Memory: 4Gi request / 8Gi limit
- /dev/shm 2Gi volume

### Step 4: GitHub Actions ✅
- Trigger on v* tags → `chambie0102/chamber-hotdog-demo`

### Step 5: Tag + push → Docker build → Chamber submit

## Phase 2: Autonomous Iteration Loop (1-2 hours)

### Iteration Logic:
```
After each run:
1. Pull W&B metrics (per-class accuracy, loss curve, train vs test gap)
2. Diagnose:
   - OOM → reduce batch_size by half
   - Class imbalance (any class <60%) → add/adjust class weights
   - Loss oscillating → reduce LR by 5x
   - Loss plateau → reduce LR by 2x, add warmup
   - Train >> Test (>10% gap) → add dropout, augmentation, reduce epochs
   - Both classes >85% → fine-tune with lower LR for final push
3. Update env vars in job.yaml
4. Tag, push, build, submit
5. Repeat until both classes ≥ 90%
```

### Expected progression:
- **v1:** batch=64, lr=0.001 → ~75-80% (LR too high)
- **v2:** lr=3e-4, class weights → ~80-85%
- **v3:** lr=1e-4, warmup=2, basic aug → ~85-90%
- **v4:** aggressive aug, dropout=0.2 → ~90%+
- **v5:** lr=5e-5, epochs=15 → ≥90% both classes ✓

## Phase 3: Demo (15 min)

- Download best model from W&B
- Run Flask demo on port 5051 (separate from xray on 5050)
- Show W&B iteration story
- CLI inference on sample images

## Success Criteria
- 3-5 training runs showing clear progression
- Final accuracy ≥90% on both classes
- W&B dashboard showing the iteration story
- Demo app working on phone (port 5051)
