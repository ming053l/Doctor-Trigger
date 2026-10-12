"""Train the image-only detector with the deployed distortion-augmentation recipe (App. E.2).

Same training as train_image_detector.py, but each training image first passes through the
albumentations pipeline of the model config's ``train.albumentations`` block (JPEG
recompression, Gaussian blur, Gaussian noise, colour jitter; experiments/_snapshot/src/
augmentation.py), then resize to 224 and a horizontal flip. Without this augmentation the detector
loses its separation on lossy channels (Table 10).
Writes <checkpoints.image_detector_aug>/<ds>/best_val_auc.pt.
Usage: train_image_detector_aug.py <ffpp|dfdc|celebdf>
"""
import sys

import numpy as np
import torchvision.transforms as T
import yaml
from PIL import Image

from doctor_trigger import config
from experiments._snapshot.src.augmentation import build_augmentation
from experiments.shared.image_only_detector import INPUT_SIZE
from experiments.train_image_detector import train


class DistortionAugmentation:
    """PIL -> PIL wrapper around an albumentations pipeline (which works on numpy arrays)."""

    def __init__(self, transform): self.transform = transform

    def __call__(self, pil):
        return Image.fromarray(self.transform(image=np.asarray(pil))["image"])


if __name__ == "__main__":
    with open(config.path("model_config"), encoding="utf-8") as handle:
        recipe = yaml.safe_load(handle)["train"]["albumentations"]
    augmentation = T.Compose([DistortionAugmentation(build_augmentation(recipe)), T.Resize((INPUT_SIZE, INPUT_SIZE)),
                              T.RandomHorizontalFlip(), T.ToTensor()])
    train(sys.argv[1], augmentation, config.path("checkpoints.image_detector_aug"))
