"""Compatibility entry point for the Streamlit dashboard.

This module intentionally keeps the historical root command working while the
project continues to prefer the canonical app.py entry point.
"""


def main() -> None:
    """Load the Streamlit script through the historical root command."""
    import app  # noqa: F401


if __name__ == "__main__":
    main()
