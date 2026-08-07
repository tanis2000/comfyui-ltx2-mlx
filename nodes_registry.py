from functools import wraps
from typing import Callable, Optional, Type

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

DISPLAY_NAME_PREFIX = "LTX-2 MLX"


def _is_v3_node(node_class: Type) -> bool:
    return hasattr(node_class, "define_schema") and callable(
        getattr(node_class, "define_schema")
    )


def _wrap_define_schema(node_class: Type, display_name: str) -> None:
    original_define_schema = node_class.define_schema

    @classmethod
    @wraps(original_define_schema.__func__)
    def wrapped_define_schema(cls):
        schema = original_define_schema.__func__(cls)
        if schema.display_name is None:
            schema.display_name = display_name
        return schema

    node_class.define_schema = wrapped_define_schema


def comfy_node(
    node_class: Optional[Type] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable:
    def decorator(node_class: Type) -> Type:
        nonlocal name, description

        if name is None:
            name = node_class.__name__

        if description is None:
            description = f"{DISPLAY_NAME_PREFIX} {name}"

        if _is_v3_node(node_class):
            _wrap_define_schema(node_class, description)

        NODE_CLASS_MAPPINGS[name] = node_class
        NODE_DISPLAY_NAME_MAPPINGS[name] = description
        return node_class

    if node_class is None:
        return decorator
    return decorator(node_class)
