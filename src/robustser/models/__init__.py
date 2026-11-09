"""Compact models used by controlled RobustSER-Tiny experiments."""

from .depthwise_separable_cnn import DepthwiseSeparableCNN
from .paper_inspired_cnn import PaperInspiredSmallCNN


def build_model(name: str, *, num_classes: int):
    models = {
        "paper_inspired_small_cnn": PaperInspiredSmallCNN,
        "depthwise_separable_cnn": DepthwiseSeparableCNN,
    }
    try:
        return models[name](num_classes=num_classes)
    except KeyError as error:
        raise ValueError(f"Unsupported model name: {name}") from error


__all__ = ["DepthwiseSeparableCNN", "PaperInspiredSmallCNN", "build_model"]
