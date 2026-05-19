# `benjamin-levin/mlx` — fork of [`ml-explore/mlx`](https://github.com/ml-explore/mlx)

> **Interim fork**: carries two in-flight upstream draft PRs assembled into `main` so users can install the stack as one piece while the PRs land upstream individually. Will be retired once both PRs merge into `ml-explore/mlx`.

## What's different from upstream

| Addition | Effect | Draft PR |
|---|---|---|
| **`mx.fast.fused_qsdpa`** — compiled-in 2-pass FlashAttention on a quantized KV cache. Cooperative tile-shared dequant across all GQA heads inside the Metal kernel; bit-exact vs `mx.fast.scaled_dot_product_attention(bf16)` on dequantized K/V. Decode-shape only (`T_q=1`), `head_dim=256`, `gqa_factor=8`, `bits ∈ {4,8}`, `group_size ∈ {32,64}`. | +18% e2e @ 32k N=1, +27% @ 96k N=1 on Qwen3.6-35B-A3B-4bit (M4 Max 36 GB). `left_padding` extension recovers +22.8% on heterogeneous-prompt N=3 32k batches. | [#1](https://github.com/benjamin-levin/mlx/pull/1) |
| **`mx.fast.fused_swiglu_gather_qmv`** — single Metal kernel fusing `silu(gate) * up` activation inline with the quantized `gather_qmm` matvec. Fast-path: `N % 8 == 0 && K % 512 == 0`, affine quantization. | ~1.05× microbench geomean across 27 shapes vs the explicit three-launch reference. Bit-exact correctness (cos ≥ 0.9999). | [#2](https://github.com/benjamin-levin/mlx/pull/2) |

### Why these exist on Apple Silicon

The M4 series and below have **no native int4 / int8 GPU matmul instructions** — M5's Neural Accelerators added these, but anything M4 or earlier dequantizes every quantized weight to bf16 in registers before the matmul fires, on every use. KV-quantization in particular only pays off when the dequant is fused into the attention kernel; the stock 3-launch quantized-SDPA path is actually *slower* than bf16 SDPA at long context because each launch redoes the dequant. `fused_qsdpa` is the structural fix: cooperative dequant inside a 2-pass FlashAttention kernel, shared across GQA query heads. `fused_swiglu_gather_qmv` collapses three launches into one to amortize the same overhead in the MoE expert path.

## Install

```bash
pip install git+https://github.com/benjamin-levin/mlx.git@main
```

Requires Xcode CLT + CMake; ~5–10 min build on M-series.

For the full Python-level stack on top of this (scheduler-fix, MTP spec decode, persistent prompt cache, SnapKV, etc.), install the matching mlx-lm fork — `setup.py` will pull this fork automatically:

```bash
pip install git+https://github.com/benjamin-levin/mlx-lm.git@main
```

## Context

These changes were extracted from an [optimization study](https://github.com/benjamin-levin/mlx-fast) of 28 strategies attempted on Qwen3.6-35B-A3B-4bit on M4 Max (13 shipped, 15 documented as dead ends, with per-strategy methodology + measurement). See each PR's body for full per-feature methodology, in-PR re-measurement, and any honest corrections vs originally-claimed wins.

---

# MLX

[**Quickstart**](#quickstart) | [**Installation**](#installation) |
[**Documentation**](https://ml-explore.github.io/mlx/build/html/index.html) |
[**Examples**](#examples)

[![CircleCI](https://circleci.com/gh/ml-explore/mlx.svg?style=svg)](https://circleci.com/gh/ml-explore/mlx)

MLX is an array framework for machine learning on Apple silicon,
brought to you by Apple machine learning research.

Some key features of MLX include:

- **Familiar APIs**: MLX has a Python API that closely follows NumPy. MLX
   also has fully featured C++, [C](https://github.com/ml-explore/mlx-c), and
   [Swift](https://github.com/ml-explore/mlx-swift/) APIs, which closely mirror
   the Python API. MLX has higher-level packages like `mlx.nn` and
   `mlx.optimizers` with APIs that closely follow PyTorch to simplify building
   more complex models.

- **Composable function transformations**: MLX supports composable function
  transformations for automatic differentiation, automatic vectorization,
  and computation graph optimization.

- **Lazy computation**: Computations in MLX are lazy. Arrays are only
  materialized when needed.

- **Dynamic graph construction**: Computation graphs in MLX are constructed
  dynamically. Changing the shapes of function arguments does not trigger
  slow compilations, and debugging is simple and intuitive.

- **Multi-device**: Operations can run on any of the supported devices
  (currently the CPU and the GPU).

- **Unified memory**: A notable difference from MLX and other frameworks
  is the *unified memory model*. Arrays in MLX live in shared memory.
  Operations on MLX arrays can be performed on any of the supported
  device types without transferring data.

MLX is designed by machine learning researchers for machine learning
researchers. The framework is intended to be user-friendly, but still efficient
to train and deploy models. The design of the framework itself is also
conceptually simple. We intend to make it easy for researchers to extend and
improve MLX with the goal of quickly exploring new ideas.

The design of MLX is inspired by frameworks like
[NumPy](https://numpy.org/doc/stable/index.html),
[PyTorch](https://pytorch.org/), [Jax](https://github.com/google/jax), and
[ArrayFire](https://arrayfire.org/).

## Examples

The [MLX examples repo](https://github.com/ml-explore/mlx-examples) has a
variety of examples, including:

- [Transformer language model](https://github.com/ml-explore/mlx-examples/tree/main/transformer_lm) training.
- Large-scale text generation with
  [LLaMA](https://github.com/ml-explore/mlx-examples/tree/main/llms/llama) and
  finetuning with [LoRA](https://github.com/ml-explore/mlx-examples/tree/main/lora).
- Generating images with [Stable Diffusion](https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion).
- Speech recognition with [OpenAI's Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper).

## Quickstart

See the [quick start
guide](https://ml-explore.github.io/mlx/build/html/usage/quick_start.html)
in the documentation.

## Installation

MLX is available on [PyPI](https://pypi.org/project/mlx/). To install MLX on
macOS, run:

```bash
pip install mlx
```

To install the CUDA backend on Linux, run:

```bash
pip install mlx[cuda]
```

To install a CPU-only Linux package, run:

```bash
pip install mlx[cpu]
```

Checkout the
[documentation](https://ml-explore.github.io/mlx/build/html/install.html#)
for more information on building the C++ and Python APIs from source.

## Contributing

Check out the [contribution guidelines](https://github.com/ml-explore/mlx/tree/main/CONTRIBUTING.md) for more information
on contributing to MLX. See the
[docs](https://ml-explore.github.io/mlx/build/html/install.html) for more
information on building from source, and running tests.

We are grateful for all of [our
contributors](https://github.com/ml-explore/mlx/tree/main/ACKNOWLEDGMENTS.md#Individual-Contributors). If you contribute
to MLX and wish to be acknowledged, please add your name to the list in your
pull request.

## Citing MLX

The MLX software suite was initially developed with equal contribution by Awni
Hannun, Jagrit Digani, Angelos Katharopoulos, and Ronan Collobert. If you find
MLX useful in your research and wish to cite it, please use the following
BibTex entry:

```text
@software{mlx2023,
  author = {Awni Hannun and Jagrit Digani and Angelos Katharopoulos and Ronan Collobert},
  title = {{MLX}: Efficient and flexible machine learning on Apple silicon},
  url = {https://github.com/ml-explore},
  version = {0.0},
  year = {2023},
}
```
