# Cat vs Dog Classifier

Trains a ResNet-18 to tell cats from dogs. Uses PyTorch and torchvision.

## Quick start

```bash
uv sync
uv run train
uv run classify best_model.pt cat_photo.jpg
```

## Data

https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

Download and place under `Data/train_dataset` and `Data/test_dataset`.

we should have used hugging face instead of github actually

```
Data/
├── train_dataset/
│   ├── Cat/
│   └── Dog/
└── test_dataset/
    ├── Cat/
    └── Dog/
```

## Options

```bash
uv run train --data-dir Data --epochs 30 --batch-size 32 --model resnet18
uv run classify best_model.pt img1.jpg img2.jpg img3.jpg --batch-size 32
```
