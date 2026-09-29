"""Target transformations that preserve weather quantities at inference time."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def encode_rain_target(values: ArrayLike) -> NDArray[np.float32]:
    """Encode non-negative precipitation amounts with log1p."""
    amounts = np.asarray(values, dtype=np.float32)
    return np.log1p(np.maximum(amounts, 0.0)).astype(np.float32, copy=False)


def decode_rain_target(values: ArrayLike) -> NDArray[np.float32]:
    """Restore precipitation amounts from log1p space, bounded at zero."""
    encoded = np.asarray(values, dtype=np.float32)
    return np.expm1(np.maximum(encoded, 0.0)).astype(np.float32, copy=False)


def decode_model_target(
    target_name: str, values: ArrayLike, target_transform: str | None
) -> ArrayLike:
    """Decode only precipitation predictions tagged as log1p-transformed."""
    if target_name == "precip" and target_transform == "log1p":
        return decode_rain_target(values)
    return values


__all__ = ["decode_model_target", "decode_rain_target", "encode_rain_target"]