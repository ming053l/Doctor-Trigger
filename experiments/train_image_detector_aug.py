"""Retrain the passive head on canonical data WITH the deployed head's
distortion-augmentation recipe, using the deployed code's own albumentations
builder (experiments/_snapshot/src/dataset.py: build_albu_transform) driven by the
deployed model config's albumentations block. The plain train_image_detector.py
variant omits augmentation and its head collapses on lossy channels.
Writes <checkpoints.image_detector_aug>/<ds>/best_val_auc.pt.
Usage: train_image_detector_aug.py <ffpp|dfdc|celebdf>
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torchvision.transforms as T
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
from doctor_trigger import config  # noqa: E402
from experiments.shared.image_only_detector import INPUT_SIZE  # noqa: E402
from experiments.train_image_detector import train  # noqa: E402


def load_albumentations_builder():
    file = Path(config.path("model_snapshot")) / "src" / "dataset.py"
    spec = importlib.util.spec_from_file_location("image_only_detector_dataset", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_albu_transform


class AlbuAug:
    def __init__(self, transform): self.transform = transform

    def __call__(self, pil):
        return Image.fromarray(self.transform(image=np.asarray(pil))["image"])


if __name__ == "__main__":
    with open(config.path("model_config"), encoding="utf-8") as handle:
        albu_cfg = yaml.safe_load(handle)["train"]["albumentations"]
    augmentation = T.Compose([AlbuAug(load_albumentations_builder()(albu_cfg)), T.Resize((INPUT_SIZE, INPUT_SIZE)),
                              T.RandomHorizontalFlip(), T.ToTensor()])
    train(sys.argv[1], augmentation, config.path("checkpoints.image_detector_aug"))
