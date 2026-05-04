from dataclasses import dataclass, field

CLASSES = ["Cat", "Dog"]
NUM_CLASSES = len(CLASSES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class TrainingConfig:
    data_dir: str = "Data"
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 30
    seed: int = 42
    num_workers: int = 4
    val_split: float = 0.15
    patience: int = 10
    max_keep_checkpoints: int = 3
    compile_model: bool = True
    amp: bool = True
    log_dir: str = "runs/cats_vs_dogs"
    save_dir: str = "checkpoints"
    resume: str | None = None
    model_name: str = "resnet18"


@dataclass
class InferenceConfig:
    model_path: str = "best_model.pth"
    image_path: str = ""
    classes: list[str] = field(default_factory=lambda: CLASSES)
    batch_size: int = 32
    compile_model: bool = True
