# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A ComfyUI custom node package that wraps the MLX port of Lightricks LTX-2.3 ([dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx), installed as `ltx-core-mlx` / `ltx-pipelines-mlx`) so video generation runs on Apple Silicon without PyTorch/CUDA fp8 kernels. It is a thin adapter: all model math lives upstream; this repo only converts ComfyUI inputs/outputs and manages pipeline lifetime. Apple Silicon only (macOS + arm64), Python >= 3.11.

The parent `ComfyUI/AGENTS.md` is also loaded. Its node, user-input, and commit conventions apply here. Its model-implementation rules (Comfy Kitchen ops, attention, dtype policy) target core ComfyUI model code, which this package does not contain.

## Commands

No test suite, linter, or build step. In this local StabilityMatrix checkout the Python env with MLX + the upstream packages is `ComfyUI/venv-3.13`. `ComfyUI/venv` is Python 3.10 and lacks them.

```bash
# Install deps (from this directory)
../../venv-3.13/bin/pip install -r requirements.txt

# Import/registration smoke test (run from the ComfyUI root; needs comfy_api + folder_paths on sys.path)
cd ../.. && venv-3.13/bin/python -c "import importlib; m = importlib.import_module('custom_nodes.comfyui-ltx2-mlx'); print(m.NODE_DISPLAY_NAME_MAPPINGS)"

# Run ComfyUI for end-to-end checks (from the ComfyUI root)
venv-3.13/bin/python main.py
```

Changes have been verified end-to-end against a live server: queue via `POST /prompt`, then poll `/history` for zero `node_errors` and a saved mp4. Use the `distilled` pipeline with a q4/q8 tier at tiny settings (e.g. 256×256, 25 frames). Every other pipeline type takes 10–40+ minutes (see README benchmark). `examples/ltx2mlx_text_to_video.json` is a UI-format workflow (for Workflow → Open), not the API-format payload `/prompt` expects.

## Architecture

- **Registration**: `nodes_registry.comfy_node` is a decorator that fills `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` and injects the display name into V3 schemas. A node module only registers if it's imported in `nodes/__init__.py`. Nodes use the V3 API (`comfy_api.latest.io`: `define_schema` + `execute` classmethods), and `node_id` must match the registered `name`.
- **Two parallel pipeline types**: `LTX2MLXModelLoader` → `LTX2MLX_PIPELINE` → `LTX2MLXGenerate` (T2V, or I2V when `image` is connected), and `LTX2MLXAudioModelLoader` → `LTX2MLX_A2V_PIPELINE` → `LTX2MLXAudioToVideo`. The custom socket types keep A2V and T2V pipelines from being cross-wired.
- **Pipeline cache** (`nodes/loaders.py`): module-level `_PIPELINE_CACHE` holds at most one pipeline. It is keyed by `("t2v"|"a2v", resolved_model_dir, [pipeline_type,] low_ram, loras)` and cleared before inserting, so loading any different pipeline evicts the previous one. This is deliberate: the 22B model doesn't fit twice in unified memory.
- **LoRAs**: `LTX2MLXLora` outputs a `(full_path, strength)` tuple. Loaders take any number of them through an `io.Autogrow` `loras` input; prompts that omit it get `{}`. The loader sets upstream's private `_pending_loras` on the pipeline it constructs, the same way the upstream CLI does. Upstream fuses them on every transformer load (non-streaming), or per block at bind time when `low_ram` is on. Because LoRAs are part of the cache key, changing them builds a new pipeline instead of mutating a cached one.
- **Upstream is file-path based**: generate nodes write ComfyUI tensors to temp files (`_tensor_to_image_path` → PNG, `_audio_to_wav_path` → WAV), call upstream `pipeline.generate_and_save(...)` into a temp mp4, delete the inputs in `finally`, then `_save_render` moves the mp4 into the output dir via `folder_paths.get_save_image_path` (same naming as stock `SaveVideo`). It returns `VideoFromFile` + `ui.PreviewVideo`. Both generate nodes are `is_output_node=True`, so no downstream Save/Preview node is needed. `audio.py` reuses these helpers from `generate.py`.
- **Upstream imports are lazy**: `ltx_pipelines_mlx` is imported inside functions so the package still loads on non-Apple machines, and `_check_apple_silicon()` raises a clear error at execute time instead.
- **Model weights** come from Hugging Face on first loader run: the `dgrauet/ltx-2.3-mlx[-q8|-q4]` tiers, plus upstream's default Gemma text encoder `mlx-community/gemma-3-12b-it-4bit`. `custom_model_dir` overrides with a local path or another repo id.

## Constraints and gotchas

- **Frame counts must be 8k+1** (VAE 8x temporal compression). `_snap_frame_count` rounds to the nearest valid count. A2V with `match_audio_length` uses `_snap_frame_count_leq`, which always rounds *down*. Upstream sizes audio RoPE tables from the requested video duration, so requesting more video than there is audio crashes with a `broadcast_shapes` mismatch. Padding the audio with silence was tried and reverted (`e71109b`). Keep video ≤ audio.
- **Upstream API facts**: the constructor kwarg is `low_ram_streaming`, not `low_ram`, and `generate_and_save` requires keyword-only `frame_rate`. When unsure of a signature, read the installed source in `../../venv-3.13/lib/python3.13/site-packages/ltx_pipelines_mlx/`. That directory also has pipelines not yet wrapped as nodes (retake, keyframe interpolation, IC-LoRA, lipdub, etc.).
- **Example workflow coupling**: the UI-format workflows in `examples/` store `widgets_values` positionally, including the frontend's auto-inserted `control_after_generate` widget after `seed`. Adding, removing, or reordering widget inputs on the LTX2MLX nodes requires updating them. Autogrow slots are saved as named inputs (`"name": "loras.lora_0"`, `"label": "lora_0"`), plus one empty trailing slot, and don't appear in `widgets_values`.
- **Versioning/publishing**: the version lives only in `pyproject.toml`. `.github/workflows/publish.yml` publishes to the Comfy registry on any push to `master` that touches `pyproject.toml`, so only push `pyproject.toml` changes when you intend to release.
- Commit style follows history: short imperative subject (`Fix ...`, `Add ...`, `Make ...`, `Bump ...`) with a body explaining the cause and how it was verified.
