import os
import tempfile

import folder_paths
import soundfile as sf
from comfy_api.input_impl import VideoFromFile
from comfy_api.latest import io

from ..nodes_registry import comfy_node
from .generate import _output_path, _tensor_to_image_path


def _audio_to_wav_path(audio: dict) -> str:
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    # ComfyUI AUDIO tensors are (batch, channels, samples); take the first item.
    samples = waveform[0].transpose(0, 1).cpu().numpy()

    handle, path = tempfile.mkstemp(suffix=".wav", prefix="ltx2mlx_audio_")
    os.close(handle)
    sf.write(path, samples, sample_rate)
    return path


@comfy_node(name="LTX2MLXAudioToVideo", description="LTX-2 MLX Audio to Video (Song Mode)")
class LTX2MLXAudioToVideo(io.ComfyNode):
    """Audio-to-video generation with an LTX-2.3 MLX pipeline (song-driven video)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTX2MLXAudioToVideo",
            category="LTX2MLX",
            inputs=[
                io.Custom("LTX2MLX_A2V_PIPELINE").Input("pipeline"),
                io.String.Input("prompt", multiline=True, default=""),
                io.Audio.Input("audio"),
                io.Image.Input("image", optional=True),
                io.Int.Input("frame_rate", default=24, min=1, max=60),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF),
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
        audio: dict,
        frame_rate: int,
        seed: int,
        image=None,
    ) -> io.NodeOutput:
        output_path = _output_path("ltx2mlx_a2v_")
        audio_path = _audio_to_wav_path(audio)
        image_path = _tensor_to_image_path(image) if image is not None else None

        try:
            pipeline.generate_and_save(
                prompt=prompt,
                output_path=output_path,
                audio_path=audio_path,
                frame_rate=frame_rate,
                seed=seed,
                image=image_path,
            )
        finally:
            os.remove(audio_path)
            if image_path is not None:
                os.remove(image_path)

        return io.NodeOutput(VideoFromFile(output_path))
