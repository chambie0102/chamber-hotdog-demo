# Hot Dog / Not Hot Dog — Chamber GPU Training

Binary image classifier using ResNet-18 fine-tuned on Food-101.

## Config (env vars — no rebuild needed)

| Variable | Default | Description |
|----------|---------|-------------|
| BATCH_SIZE | 64 | Training batch size |
| LEARNING_RATE | 1e-3 | Initial learning rate |
| EPOCHS | 5 | Number of training epochs |
| OPTIMIZER | adam | adam / adamw / sgd |
| AUGMENTATION | none | none / basic |
| WARMUP_EPOCHS | 0 | Linear warmup epochs |
| USE_CLASS_WEIGHTS | false | Enable class weighting |
| DROPOUT | 0.0 | Dropout before classifier |
