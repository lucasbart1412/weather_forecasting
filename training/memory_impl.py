from training._runtime import *
from typing import Any


class MemoryManager:
    """
    Memory manager with aggressive cleanup.
    Capsule gc.collect() and malloc_trim() to force release
    from the underlying C++ memory to the OS.
    """

    @staticmethod
    def trim_memory() -> None:
        """Forces memory release to OS (Linux)."""
        gc.collect()
        try:
            libc = ctypes.CDLL(None)
            libc.malloc_trim(0)
        except (AttributeError, OSError):
            # Not available on Windows
            pass

    @staticmethod
    def free_lgbm_booster(model: Any) -> None:
        """Explicitly frees the C++ dataset from LightGBM."""
        if hasattr(model, "free_dataset"):
            try:
                model.free_dataset()
            except Exception:
                pass
        if hasattr(model, "_Booster") and model._Booster is not None:
            try:
                model._Booster.free_dataset()
            except Exception:
                pass


# ============================================================================
