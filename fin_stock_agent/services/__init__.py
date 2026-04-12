"""Service package.

Avoid eager re-exports here; importing submodules directly keeps startup and
hot-reload paths stable in the single-user app runtime.
"""

__all__ = []
