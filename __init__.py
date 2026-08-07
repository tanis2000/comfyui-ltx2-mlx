"""ComfyUI-LTX2-MLX: LTX-2.3 video generation on Apple Silicon via MLX."""

from . import nodes  # noqa: F401  (imports register nodes via nodes_registry)
from .nodes_registry import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__version__ = "0.1.0"

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
