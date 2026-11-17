"""
Namespace
=========
"""
# src/slamkit/core/namespace.py

__all__ = [
    "Namespace",
]


class Namespace:
    """Base for classes that only group static members (or act as type tags)
    and must never be instantiated."""

    __slots__ = ()

    def __init__(self) -> None:
        raise TypeError(f"{type(self).__name__} is never instantiated.")
