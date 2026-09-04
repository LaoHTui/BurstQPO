"""Backward-compatible mean-model factories."""

from .models import (
    add_mean_models, constant_background, constant_background_model, ercod,
    ercod_model, fred, fred_model, linear_background,
    linear_background_model, polynomial_background,
    polynomial_background_model,
)

__all__ = [
    "add_mean_models", "constant_background",
    "constant_background_model", "ercod", "ercod_model", "fred",
    "fred_model", "linear_background", "linear_background_model",
    "polynomial_background", "polynomial_background_model",
]
