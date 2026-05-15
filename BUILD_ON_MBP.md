# Build fused-swiglu-gather-qmv MLX on MBP

This branch adds `affine_gather_qmv_swiglu` — a fused SiLU(gate)*up + gather quantized matvec — directly into MLX's Metal kernel library. Lands the V1 kernel from `/Users/ai-studio/code/mlx-fast/mlx_fast/kernels/fused_swiglu_down.py` as a compiled-in MLX op, bypassing `mx.fast.metal_kernel`'s runtime-JIT overhead.

## Prerequisites on MBP

1. **Full Xcode** (NOT just Command Line Tools): `xcodebuild -version` must work.
2. `cmake >= 3.25`
3. Python 3.10+

## Clone + build

```bash
git clone https://github.com/benjamin-levin/mlx.git
cd mlx
git checkout fused-swiglu-gather-qmv

# In a fresh venv:
python3 -m venv .venv
source .venv/bin/activate
pip install setuptools wheel nanobind cmake

# Build (10-30 min, mostly Metal kernel compilation):
CMAKE_BUILD_PARALLEL_LEVEL=8 pip install . --no-build-isolation -v 2>&1 | tee build.log
```

If build fails on the new Metal kernels:
- `affine_gather_qmv_swiglu` template — see `mlx/backend/metal/kernels/quantized.h`
- Instantiation — `mlx/backend/metal/kernels/quantized.metal` line ~92
- Common cause: `metal::exp` not found → may need `#include <metal_math>` at top of quantized.h (already present in original file)

## Verify the kernel compiled in

After successful install:

```bash
python3 -c "
import mlx.core as mx
# Force kernel cache load by invoking gather_qmm normally
# (the new affine_gather_qmv_swiglu will be in the same metallib)
print(mx.__version__)
import os
metallib = os.path.dirname(mx.__file__) + '/lib/mlx.metallib'
print('metallib:', metallib, 'size:', os.path.getsize(metallib))
"
```

A larger metallib than stock 0.31.2 confirms our kernel instantiations were added.

## What this branch adds (current state)

- `mlx/backend/metal/kernels/quantized.h`:
  - `load_vector_swiglu<>` — fused SiLU(gate)*up + load_vector for 2/3/4/5/6/8-bit
  - `qmv_fast_swiglu_impl<>` — fused-input qmv_fast
  - `affine_gather_qmv_swiglu<>` — kernel entry point (parallel to `affine_gather_qmv_fast`)
- `mlx/backend/metal/kernels/quantized.metal`:
  - Template instantiation for the new kernel across {float16, bfloat16, float32} × {group_size} × {bits}
- `mlx/backend/metal/quantized.cpp`:
  - `gather_qmv_swiglu(...)` dispatch function

## What's NOT in this branch yet (todo for full Python access)

1. **Forward declaration** in `mlx/backend/metal/quantized.h` so external callers can use `gather_qmv_swiglu`
2. **Primitive class** `FusedSwiGLUGatherQMV` in `mlx/fast_primitives.h` + `mlx/fast.cpp` modeled on `RMSNorm`
3. **Public C++ API** `mlx::fast::fused_swiglu_gather_qmm(...)` in `mlx/fast.h` + `mlx/fast.cpp`
4. **Python binding** in `python/src/fast.cpp` (search for `scaled_dot_product_attention` for the binding pattern)

Estimated effort once build is verified: 3-5 hours.

## Test plan after full integration

1. Microbench against `mx.gather_qmm(mx.silu(g)*u, ...)` — expect 1.10× per call
2. Replace `fused_swiglu_down` calls in `/Users/ai-studio/code/mlx-fast/mlx_fast/mtp/moe_compile.py` with `mx.fast.fused_swiglu_gather_qmm` — expect 3-5% end-to-end gain over gate+up alone (the savings from removing mx.fast.metal_kernel's runtime-JIT overhead)
