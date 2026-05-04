import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from .config import IMAGENET_MEAN, IMAGENET_STD, CLASSES, InferenceConfig
from .train import create_model


def load_model(
    model_path: str,
    model_name: str = "resnet18",
    num_classes: int = len(CLASSES),
    compile_model: bool = True,
) -> nn.Module:
    model = create_model(model_name, num_classes, compile_model=False)
    state = torch.load(model_path, weights_only=True)

    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()
    if compile_model and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

    return model


def _get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@torch.inference_mode()
def classify_image(
    model: nn.Module,
    image_path: str,
    classes: list[str],
    device: torch.device = torch.device("cpu"),
) -> tuple[str, float]:
    transform = _get_transform()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device, non_blocking=True)

    outputs = model(tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

    return classes[predicted.item()], confidence.item()


@torch.inference_mode()
def classify_batch(
    model: nn.Module,
    image_paths: list[str],
    classes: list[str],
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> list[tuple[str, float]]:
    transform = _get_transform()
    results: list[tuple[str, float]] = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch = torch.stack([
            transform(Image.open(p).convert("RGB")) for p in batch_paths
        ]).to(device, non_blocking=True)

        outputs = model(batch)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predicted = torch.max(probabilities, 1)

        for j in range(len(batch_paths)):
            results.append((classes[predicted[j].item()], confidences[j].item()))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify images")
    parser.add_argument("model_path", help="Path to model checkpoint (.pt)")
    parser.add_argument("image_path", nargs="+", help="Image path(s) to classify")
    parser.add_argument("--model-name", default="resnet18", choices=["resnet18", "resnet50", "efficientnet_b0", "convnext_tiny"], help="Model architecture")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for multiple images")
    args = parser.parse_args()

    model = load_model(
        args.model_path,
        model_name=args.model_name,
        compile_model=not args.no_compile,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if len(args.image_path) == 1:
        label, confidence = classify_image(model, args.image_path[0], CLASSES, device)
        print(f"{args.image_path[0]}: {label} (confidence: {confidence:.4f})")
    else:
        results = classify_batch(model, args.image_path, CLASSES, args.batch_size, device)
        for path, (label, confidence) in zip(args.image_path, results):
            print(f"{path}: {label} (confidence: {confidence:.4f})")


if __name__ == "__main__":
    main()
