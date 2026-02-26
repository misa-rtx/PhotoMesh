"""View-selection strategies for texture back-projection."""

from photomesh.view_selection.base import ViewSelector
from photomesh.view_selection.closest_z import ClosestZSelector

__all__ = ["ViewSelector", "ClosestZSelector"]
