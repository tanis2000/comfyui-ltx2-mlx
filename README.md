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

## Benchmark

Measured with the direct Python API (`pipeline.generate_and_save(...)`, no ComfyUI server
overhead), one pipeline at a time so no tier/type shared cached weights:

- **Hardware**: Apple M4 Max, 64 GB unified memory, macOS 26.3
- **Model tier**: `dgrauet/ltx-2.3-mlx-q4`, `low_ram_streaming` off
- **Settings**: 704x480 (two-stage pipelines snap the intermediate/output to a multiple of 64,
  landing at 704x448), 49 frames (`8*6+1`, ~2s @ 24fps), `cfg_scale=3.0`, seed 42, single fixed
  prompt, cold pipeline load per run (no warm cache reuse between pipeline types)

| Pipeline type  | Model/text-encoder load | Generate time | Frames/sec | Notes |
|----------------|-------------------------:|---------------:|-----------:|-------|
| `distilled`      | ~34s  | 88s     | 0.56  | 8-step + 3-step denoise, fastest by a wide margin |
| `two_stage`      | ~41s  | 964s (~16.1 min)  | 0.051 | 30-step guided denoise per stage |
| `two_stage_hq`   | ~40s  | 968s (~16.1 min)  | 0.051 | Same step count as `two_stage`; HQ pass cost didn't show up at this resolution/frame count |
| `one_stage`      | ~24s  | 2465s (~41.1 min) | 0.02  | 30-step guided denoise at full resolution (no half-res first stage) — the slowest option here |

Takeaways:
- `distilled` is roughly 11x faster than `two_stage`/`two_stage_hq` and 28x faster than
  `one_stage` at this resolution/frame count — the right default for iteration and previews.
- `two_stage`/`two_stage_hq` run their expensive guided denoise at half resolution first, then
  a short refinement pass, which is why they land far below `one_stage` despite doing "two
  passes" — `one_stage` pays full 30-step guided denoise cost at full resolution instead.
- These numbers are for the q4 tier only; q8/bf16 will be slower and use proportionally more
  unified memory (see the RAM table above), and a prior session on this same machine saw q8 at
  1920x1080/8s exceed an 8-hour render budget under heavy swap pressure — keep resolution/frame
  count modest, or enable `low_ram`, on tighter-memory machines.
- Load time includes the Gemma text encoder load + prompt encoding + transformer weight load;
  it's a fixed one-time cost per pipeline (re)instantiation, not per-generation, and is small
  relative to generate time for every pipeline type above.

## Status

Early / alpha, mirroring the maturity level of the upstream `ltx-2-mlx` port. Retake/extend,
keyframe interpolation, and IC-LoRA conditioning from the upstream CLI are not yet wrapped as
nodes here — contributions welcome.

## License

MIT
