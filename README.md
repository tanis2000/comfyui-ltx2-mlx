# ComfyUI-LTX2-MLX

ComfyUI nodes for running [Lightricks LTX-2.3](https://github.com/Lightricks/LTX-Video) video
generation natively on Apple Silicon, using the [MLX](https://github.com/ml-explore/mlx) port
provided by [dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx).

The official `ComfyUI-LTXVideo` nodes rely on PyTorch, and the fp8 checkpoints Lightricks
ships (e.g. `ltx-2.3-22b-dev-fp8.safetensors`) require CUDA — PyTorch's MPS backend has no
kernel support for the `float8_e4m3fn` dtype, so those checkpoints only run on CUDA or (very
slowly) on CPU. This package sidesteps that entirely by running the model through MLX, which
has its own quantization built for Apple's unified memory, instead of waiting on PyTorch/MPS
to support fp8.

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4). There is no CUDA/Windows/Linux backend — MLX only runs
  on Apple GPUs.
- Python >= 3.11
- Enough unified memory for the model tier you pick:

  | Model dir                     | Size   | RAM needed         |
  |--------------------------------|--------|---------------------|
  | `dgrauet/ltx-2.3-mlx-q4`       | ~12 GB | 16 GB+ (low_ram on) |
  | `dgrauet/ltx-2.3-mlx-q8`       | ~21 GB | 32 GB+              |
  | `dgrauet/ltx-2.3-mlx` (bf16)   | ~42 GB | 64 GB+              |

  Enable `low_ram` on the loader nodes to use block streaming and fit larger tiers on smaller
  machines, at some speed cost.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/vrgamegirl19/comfyui-ltx2-mlx.git
cd comfyui-ltx2-mlx
pip install -r requirements.txt
```

Model weights are pulled automatically from Hugging Face the first time a loader node runs
(`dgrauet/ltx-2.3-mlx-q8`, `-q4`, or the bf16 default), or point `custom_model_dir` at a local
path / different Hugging Face repo.

## Nodes

- **LTX2MLXModelLoader** — loads a text/image-to-video pipeline (`two_stage`, `two_stage_hq`,
  `one_stage`, or `distilled`). Pipelines are cached by `(model_dir, pipeline_type, low_ram)`.
- **LTX2MLXGenerate** — text-to-video, or image-to-video if an `IMAGE` is connected. Frame count
  is snapped to `8k+1` to match the VAE's 8x temporal compression.
- **LTX2MLXAudioModelLoader** — loads the audio-to-video pipeline.
- **LTX2MLXAudioToVideo** — generates video driven by an `AUDIO` input (and an optional
  conditioning `IMAGE`), for song-driven / music-video workflows.

Only one pipeline is kept resident at a time — loading a new one (or switching between the
T2V/I2V and A2V loaders) evicts the previous pipeline from memory, since the 22B model doesn't
comfortably fit twice even at q8/q4.

## Status

Early / alpha, mirroring the maturity level of the upstream `ltx-2-mlx` port. Retake/extend,
keyframe interpolation, and IC-LoRA conditioning from the upstream CLI are not yet wrapped as
nodes here — contributions welcome.

## License

MIT
