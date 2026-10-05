import platform

import folder_paths
from comfy_api.latest import io

from ..nodes_registry import comfy_node

MODEL_CHOICES = [
    "dgrauet/ltx-2.3-mlx-q8",
    "dgrauet/ltx-2.3-mlx-q4",
    "dgrauet/ltx-2.3-mlx",
    "dgrauet/ltx-2.5-mlx-q8",
    "dgrauet/ltx-2.5-mlx-q4",
    "dgrauet/ltx-2.5-mlx",
]

PIPELINE_CHOICES = [
    "two_stage",
    "two_stage_hq",
    "one_stage",
    "distilled",
]

_PIPELINE_CACHE = {}


def _check_apple_silicon():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "comfyui-ltx2-mlx requires Apple Silicon (macOS + arm64). "
            "MLX has no CUDA/Windows/Linux backend."
        )


def _pipeline_class(pipeline_type: str):
    from ltx_pipelines_mlx import (
        DistilledPipeline,
        TI2VidOneStagePipeline,
        TI2VidTwoStagesHQPipeline,
        TI2VidTwoStagesPipeline,
    )

    return {
        "two_stage": TI2VidTwoStagesPipeline,
        "two_stage_hq": TI2VidTwoStagesHQPipeline,
        "one_stage": TI2VidOneStagePipeline,
        "distilled": DistilledPipeline,
    }[pipeline_type]


def _resolve_model_dir(model_dir: str, custom_model_dir: str) -> str:
    return custom_model_dir.strip() or model_dir


@comfy_node(name="LTX2MLXModelLoader", description="LTX-2 MLX Model Loader (T2V/I2V)")
class LTX2MLXModelLoader(io.ComfyNode):
    """Load an LTX-2.3 MLX pipeline for text/image-to-video generation."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTX2MLXModelLoader",
            category="LTX2MLX",
            inputs=[
                io.Combo.Input("model_dir", options=MODEL_CHOICES),
                io.Combo.Input("pipeline_type", options=PIPELINE_CHOICES),
                io.Boolean.Input("low_ram", default=False),
                io.String.Input(
                    "custom_model_dir",
                    default="",
                    optional=True,
                    tooltip="Overrides model_dir if set (local path or HF repo id).",
                ),
                io.Autogrow.Input(
                    "loras",
                    template=io.Autogrow.TemplatePrefix(io.Custom("LTX2MLX_LORA").Input("lora"), prefix="lora_", min=0),
                    optional=True,
                ),
            ],
            outputs=[
                io.Custom("LTX2MLX_PIPELINE").Output(display_name="pipeline"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_dir: str,
        pipeline_type: str,
        low_ram: bool,
        loras: io.Autogrow.Type,
        custom_model_dir: str = "",
    ) -> io.NodeOutput:
        _check_apple_silicon()

        resolved_dir = _resolve_model_dir(model_dir, custom_model_dir)
        lora_paths = tuple(loras.values())
        cache_key = ("t2v", resolved_dir, pipeline_type, low_ram, lora_paths)
        cached = _PIPELINE_CACHE.get(cache_key)
        if cached is not None:
            return io.NodeOutput(cached)

        pipeline_cls = _pipeline_class(pipeline_type)
        kwargs = {"model_dir": resolved_dir}
        if low_ram:
            kwargs["low_ram_streaming"] = True
        pipeline = pipeline_cls(**kwargs)
        # Upstream has no constructor argument for user LoRAs; its CLI sets this before
        # the first generate and the transformer load fuses them in.
        pipeline._pending_loras = list(lora_paths)

        _PIPELINE_CACHE.clear()
        _PIPELINE_CACHE[cache_key] = pipeline
        return io.NodeOutput(pipeline)


@comfy_node(name="LTX2MLXAudioModelLoader", description="LTX-2 MLX Audio Model Loader (A2V)")
class LTX2MLXAudioModelLoader(io.ComfyNode):
    """Load an LTX-2.3 MLX pipeline for audio-to-video generation."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTX2MLXAudioModelLoader",
            category="LTX2MLX",
            inputs=[
                io.Combo.Input("model_dir", options=MODEL_CHOICES),
                io.Boolean.Input("low_ram", default=False),
                io.String.Input(
                    "custom_model_dir",
                    default="",
                    optional=True,
                    tooltip="Overrides model_dir if set (local path or HF repo id).",
                ),
                io.Autogrow.Input(
                    "loras",
                    template=io.Autogrow.TemplatePrefix(io.Custom("LTX2MLX_LORA").Input("lora"), prefix="lora_", min=0),
                    optional=True,
                ),
            ],
            outputs=[
                io.Custom("LTX2MLX_A2V_PIPELINE").Output(display_name="pipeline"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_dir: str,
        low_ram: bool,
        loras: io.Autogrow.Type,
        custom_model_dir: str = "",
    ) -> io.NodeOutput:
        _check_apple_silicon()
        from ltx_pipelines_mlx import A2VidPipelineTwoStage

        resolved_dir = _resolve_model_dir(model_dir, custom_model_dir)
        lora_paths = tuple(loras.values())
        cache_key = ("a2v", resolved_dir, low_ram, lora_paths)
        cached = _PIPELINE_CACHE.get(cache_key)
        if cached is not None:
            return io.NodeOutput(cached)

        kwargs = {"model_dir": resolved_dir}
        if low_ram:
            kwargs["low_ram_streaming"] = True
        pipeline = A2VidPipelineTwoStage(**kwargs)
        pipeline._pending_loras = list(lora_paths)

        _PIPELINE_CACHE.clear()
        _PIPELINE_CACHE[cache_key] = pipeline
        return io.NodeOutput(pipeline)


@comfy_node(name="LTX2MLXLora", description="LTX-2 MLX LoRA")
class LTX2MLXLora(io.ComfyNode):
    """Pick a LoRA (ComfyUI-format LTX-2 safetensors) to fuse into an LTX-2.3 MLX pipeline."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTX2MLXLora",
            category="LTX2MLX",
            inputs=[
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength", default=1.0, min=-100.0, max=100.0, step=0.01),
            ],
            outputs=[
                io.Custom("LTX2MLX_LORA").Output(display_name="lora"),
            ],
        )

    @classmethod
    def execute(cls, lora_name: str, strength: float) -> io.NodeOutput:
        return io.NodeOutput((folder_paths.get_full_path_or_raise("loras", lora_name), strength))
