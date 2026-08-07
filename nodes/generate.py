import os
import shutil
import tempfile

import folder_paths
import numpy as np
from comfy_api.latest import io, ui
from comfy_api.input_impl import VideoFromFile
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


def _tmp_render_path() -> str:
    handle, path = tempfile.mkstemp(suffix=".mp4", prefix="ltx2mlx_render_")
    os.close(handle)
    return path


def _save_render(tmp_path: str, filename_prefix: str) -> io.NodeOutput:
    """Move a finished render into ComfyUI's output dir and report it to history,
    same convention as the stock SaveVideo node (filename_prefix + counter)."""
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, folder_paths.get_output_directory()
    )
    file = f"{filename}_{counter:05}_.mp4"
    final_path = os.path.join(full_output_folder, file)
    shutil.move(tmp_path, final_path)

    return io.NodeOutput(
        VideoFromFile(final_path),
        ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]),
    )


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
                io.Float.Input("frame_rate", default=24.0, min=1.0, max=60.0),
                io.String.Input("filename_prefix", default="ltx2mlx/video"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
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
        frame_rate: float,
        filename_prefix: str,
        image=None,
    ) -> io.NodeOutput:
        frames = _snap_frame_count(num_frames)
        tmp_path = _tmp_render_path()

        image_path = None
        if image is not None:
            image_path = _tensor_to_image_path(image)

        try:
            pipeline.generate_and_save(
                prompt=prompt,
                output_path=tmp_path,
                height=height,
                width=width,
                num_frames=frames,
                frame_rate=frame_rate,
                seed=seed,
                cfg_scale=cfg_scale,
                image=image_path,
            )
        finally:
            if image_path is not None:
                os.remove(image_path)

        return _save_render(tmp_path, filename_prefix)
