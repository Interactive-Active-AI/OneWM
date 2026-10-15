"""Temporally consistent RGB augmentation for multi-view robot videos."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor


class MultiViewVideoAugmentation:
    """Apply X-WAM-style crop and color jitter to ``[T,3,H,W]`` views.

    Each camera receives an independently sampled crop that is shared by all
    frames in that camera. Color-jitter parameters and operation order are
    shared across every camera and timestep to avoid temporal flicker and
    cross-view color inconsistency. Inputs and outputs use the ``[0, 255]``
    range and may be floating point.
    """

    def __init__(
        self,
        *,
        probability: float = 1.0,
        crop_ratio: float = 0.95,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")
        if not 0.0 < crop_ratio <= 1.0:
            raise ValueError(f"crop_ratio must be in (0, 1], got {crop_ratio}")
        for name, value in (
            ("brightness", brightness),
            ("contrast", contrast),
            ("saturation", saturation),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if not 0.0 <= hue <= 0.5:
            raise ValueError(f"hue must be in [0, 0.5], got {hue}")
        self.probability = float(probability)
        self.crop_ratio = float(crop_ratio)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)

    @staticmethod
    def _validate_view(view: Tensor) -> None:
        if view.ndim != 4 or view.shape[1] != 3:
            raise ValueError(f"video views must be [T,3,H,W], got {tuple(view.shape)}")
        if view.shape[0] <= 0 or view.shape[-2] <= 0 or view.shape[-1] <= 0:
            raise ValueError(
                f"video view dimensions must be positive, got {tuple(view.shape)}"
            )

    def _crop_and_resize(self, view: Tensor) -> Tensor:
        self._validate_view(view)
        height, width = int(view.shape[-2]), int(view.shape[-1])
        crop_height = max(1, int(height * self.crop_ratio))
        crop_width = max(1, int(width * self.crop_ratio))
        if crop_height == height and crop_width == width:
            return view.float()
        top = int(torch.randint(0, height - crop_height + 1, ()).item())
        left = int(torch.randint(0, width - crop_width + 1, ()).item())
        cropped = view[..., top : top + crop_height, left : left + crop_width].float()
        return F.interpolate(
            cropped,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )

    @staticmethod
    def _sample_factor(magnitude: float) -> float | None:
        if magnitude <= 0.0:
            return None
        lower = max(0.0, 1.0 - magnitude)
        return float(torch.empty(()).uniform_(lower, 1.0 + magnitude).item())

    def __call__(self, views: Sequence[Tensor]) -> list[Tensor]:
        if not views:
            raise ValueError("at least one video view is required")
        for view in views:
            self._validate_view(view)
        if self.probability <= 0.0 or (
            self.probability < 1.0 and float(torch.rand(()).item()) >= self.probability
        ):
            return list(views)
        augmented = [self._crop_and_resize(view) for view in views]

        brightness = self._sample_factor(self.brightness)
        contrast = self._sample_factor(self.contrast)
        saturation = self._sample_factor(self.saturation)
        hue = (
            float(torch.empty(()).uniform_(-self.hue, self.hue).item())
            if self.hue > 0.0
            else None
        )
        operation_order = torch.randperm(4).tolist()

        outputs: list[Tensor] = []
        for view in augmented:
            value = (view / 255.0).clamp(0.0, 1.0)
            for operation in operation_order:
                if operation == 0 and brightness is not None:
                    value = TF.adjust_brightness(value, brightness)
                elif operation == 1 and contrast is not None:
                    value = TF.adjust_contrast(value, contrast)
                elif operation == 2 and saturation is not None:
                    value = TF.adjust_saturation(value, saturation)
                elif operation == 3 and hue is not None:
                    value = TF.adjust_hue(value, hue)
            outputs.append(value.clamp(0.0, 1.0).mul(255.0))
        return outputs
