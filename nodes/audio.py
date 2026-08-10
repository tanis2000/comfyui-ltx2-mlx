import os
import tempfile

import soundfile as sf
from comfy_api.latest import io

from ..nodes_registry import comfy_node
from .generate import _save_render, _snap_frame_count, _tensor_to_image_path, _tmp_render_path


def _audio_duration(audio: dict) -> float:
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    return waveform[0].shape[-1] / float(sample_rate)


def _audio_to_wav_path(audio: dict, min_duration: float = 0.0) -> tuple[str, float]:
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    # ComfyUI AUDIO tensors are (batch, channels, samples); take the first item.
    samples = waveform[0].transpose(0, 1).cpu().numpy()
    duration = samples.shape[0] / float(sample_rate)

    if min_duration > duration:
        # match_audio_length's video-frame -> duration snapping ("+1 frame"
        # rounding to satisfy the 8k+1 VAE constraint) can ask for slightly
        # more audio than the source clip actually has. ltx_pipelines_mlx
        # sizes its audio-token RoPE tables off the requested video duration
        # regardless of how much real audio content there is, so encoding a
        # too-short clip yields fewer audio-latent tokens than the RoPE
        # tables expect and crashes with a broadcast_shapes mismatch. Pad
        # with silence so the encoded audio always covers the requested
        # duration.
        import numpy as np

        pad_samples = int(round((min_duration - duration) * sample_rate))
        samples = np.concatenate(
            [samples, np.zeros((pad_samples, samples.shape[1]), dtype=samples.dtype)],
            axis=0,
        )
        duration = samples.shape[0] / float(sample_rate)

    handle, path = tempfile.mkstemp(suffix=".wav", prefix="ltx2mlx_audio_")
    os.close(handle)
    sf.write(path, samples, sample_rate)
    return path, duration


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
                io.Int.Input("height", default=480, min=64, max=2160, step=8),
                io.Int.Input("width", default=704, min=64, max=3840, step=8),
                io.Int.Input("frame_rate", default=24, min=1, max=60),
                io.Boolean.Input(
                    "match_audio_length",
                    default=True,
                    tooltip="Derive video length from the audio clip's duration "
                    "(snapped to the 8k+1 VAE constraint). Disable to set num_frames manually.",
                ),
                io.Int.Input(
                    "num_frames",
                    default=97,
                    min=9,
                    max=257,
                    step=8,
                    tooltip="Used only when match_audio_length is off.",
                ),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF),
                io.String.Input("filename_prefix", default="ltx2mlx/a2v"),
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
        audio: dict,
        height: int,
        width: int,
        frame_rate: int,
        match_audio_length: bool,
        num_frames: int,
        seed: int,
        filename_prefix: str,
        image=None,
    ) -> io.NodeOutput:
        tmp_path = _tmp_render_path()
        image_path = _tensor_to_image_path(image) if image is not None else None

        raw_audio_duration = _audio_duration(audio)
        if match_audio_length:
            frames = _snap_frame_count(round(raw_audio_duration * frame_rate) + 1)
        else:
            frames = _snap_frame_count(num_frames)

        audio_path, audio_duration = _audio_to_wav_path(
            audio, min_duration=frames / float(frame_rate)
        )

        try:
            pipeline.generate_and_save(
                prompt=prompt,
                output_path=tmp_path,
                audio_path=audio_path,
                height=height,
                width=width,
                num_frames=frames,
                frame_rate=frame_rate,
                seed=seed,
                image=image_path,
                audio_max_duration=frames / float(frame_rate),
            )
        finally:
            os.remove(audio_path)
            if image_path is not None:
                os.remove(image_path)

        return _save_render(tmp_path, filename_prefix)
