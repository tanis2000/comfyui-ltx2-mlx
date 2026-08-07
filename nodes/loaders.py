import platform

from comfy_api.latest import io

from ..nodes_registry import comfy_node

MODEL_CHOICES = [
    "dgrauet/ltx-2.3-mlx-q8",
    "dgrauet/ltx-2.3-mlx-q4",
    "dgrauet/ltx-2.3-mlx",
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
        custom_model_dir: str = "",
    ) -> io.NodeOutput:
        _check_apple_silicon()

        resolved_dir = _resolve_model_dir(model_dir, custom_model_dir)
        cache_key = ("t2v", resolved_dir, pipeline_type, low_ram)
        cached = _PIPELINE_CACHE.get(cache_key)
        if cached is not None:
            return io.NodeOutput(cached)

        pipeline_cls = _pipeline_class(pipeline_type)
        kwargs = {"model_dir": resolved_dir}
        if low_ram:
            kwargs["low_ram_streaming"] = True
        pipeline = pipeline_cls(**kwargs)

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
            ],
            outputs=[
                io.Custom("LTX2MLX_A2V_PIPELINE").Output(display_name="pipeline"),
            ],
        )

    @classmethod
    def execute(
        cls, model_dir: str, low_ram: bool, custom_model_dir: str = ""
    ) -> io.NodeOutput:
        _check_apple_silicon()
        from ltx_pipelines_mlx import A2VidPipelineTwoStage

        resolved_dir = _resolve_model_dir(model_dir, custom_model_dir)
        cache_key = ("a2v", resolved_dir, low_ram)
        cached = _PIPELINE_CACHE.get(cache_key)
        if cached is not None:
            return io.NodeOutput(cached)

        kwargs = {"model_dir": resolved_dir}
        if low_ram:
            kwargs["low_ram_streaming"] = True
        pipeline = A2VidPipelineTwoStage(**kwargs)

        _PIPELINE_CACHE.clear()
        _PIPELINE_CACHE[cache_key] = pipeline
        return io.NodeOutput(pipeline)
