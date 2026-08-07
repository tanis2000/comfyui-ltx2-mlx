import os
import tempfile

import folder_paths
import numpy as np
from comfy_api.input_impl import VideoFromFile
from comfy_api.latest import io
from PIL import Image

from ..nodes_registry import comfy_node


def _snap_frame_count(frames: int) -> int:
    """LTX-2.3's VAE has 8x temporal compression: valid frame counts are 8k+1."""
    k = max(0, round((frames - 1) / 8))
    return 8 * k + 1


def _tensor_to_image_path(image_tensor) -> str:
    array = (image_tensor[0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    image = Image.fromarray(array)
    handle, path = tempfile.mkstemp(suffix=".png", prefix="ltx2mlx_cond_")
    os.close(handle)
    image.save(path)
    return path


def _output_path(prefix: str) -> str:
    out_dir = folder_paths.get_output_directory()
    handle, path = tempfile.mkstemp(suffix=".mp4", prefix=prefix, dir=out_dir)
    os.close(handle)
    return path


@comfy_node(name="LTX2MLXGenerate", description="LTX-2 MLX Generate (T2V/I2V)")
class LTX2MLXGenerate(io.ComfyNode):
    """Text-to-video / image-to-video generation with an LTX-2.3 MLX pipeline."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTX2MLXGenerate",
            category="LTX2MLX",
            inputs=[
                io.Custom("LTX2MLX_PIPELINE").Input("pipeline"),
                io.String.Input("prompt", multiline=True, default=""),
                io.Image.Input("image", optional=True),
                io.Int.Input("height", default=480, min=64, max=2160, step=8),
                io.Int.Input("width", default=704, min=64, max=3840, step=8),
                io.Int.Input(
                    "num_frames",
                    default=97,
                    min=9,
                    max=257,
                    step=8,
                    tooltip="Snapped to 8k+1 (VAE 8x temporal compression).",
                ),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF),
                io.Float.Input("cfg_scale", default=3.0, min=0.0, max=20.0, step=0.1),
            ],
            outputs=[
                io.Video.Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        pipeline,
        prompt: str,
        height: int,
        width: int,
        num_frames: int,
        seed: int,
        cfg_scale: float,
        image=None,
    ) -> io.NodeOutput:
        frames = _snap_frame_count(num_frames)
        output_path = _output_path("ltx2mlx_")

        image_path = None
        if image is not None:
            image_path = _tensor_to_image_path(image)

        try:
            pipeline.generate_and_save(
                prompt=prompt,
                output_path=output_path,
                height=height,
                width=width,
                num_frames=frames,
                seed=seed,
                cfg_scale=cfg_scale,
                image=image_path,
            )
        finally:
            if image_path is not None:
                os.remove(image_path)

        return io.NodeOutput(VideoFromFile(output_path))
