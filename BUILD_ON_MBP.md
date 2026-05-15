# Build fused-swiglu-gather-qmv MLX on MBP

This branch adds `mx.fast.fused_swiglu_gather_qmv` — a Metal kernel that fuses `SiLU(gate) * up` and gather quantized matvec into a single launch. Replaces the V1 runtime-JIT kernel in `mlx-fast` with a compiled-in primitive.

## Prerequisites on MBP

1. **Full Xcode** (NOT just Command Line Tools). Verify:
   ```
   xcrun metal --version  # should NOT say "unable to find utility 'metal'"
   xcode-select -p        # should show /Applications/Xcode.app/...
   ```
   If you only have CLT: install Xcode from the App Store, then `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`.
2. CMake >= 3.25 (`brew install cmake`)
3. Python 3.10+

## Clone + build

```bash
git clone https://github.com/benjamin-levin/mlx.git
cd mlx
git checkout fused-swiglu-gather-qmv

python3 -m venv .venv
source .venv/bin/activate
pip install setuptools wheel nanobind cmake

# Full build (~15-30 min the first time; Metal kernel compilation dominates):
CMAKE_BUILD_PARALLEL_LEVEL=$(sysctl -n hw.ncpu) \
  pip install . --no-build-isolation -v 2>&1 | tee build.log
```

If you hit errors, the most likely culprits are in my added code:
- `mlx/backend/metal/kernels/quantized.h` — the `load_vector_swiglu`, `qmv_fast_swiglu_impl`, and `affine_gather_qmv_swiglu` templates
- `mlx/backend/metal/quantized.cpp` — the `gather_qmv_swiglu` dispatch + `FusedSwiGLUGatherQMV::eval_gpu`
- `mlx/fast_primitives.h` — the `FusedSwiGLUGatherQMV` class
- `mlx/fast.cpp` — the `fused_swiglu_gather_qmv` public API
- `python/src/fast.cpp` — the Python binding

## Verify the kernel works

```python
import mlx.core as mx
import mlx.nn as nn

# Test inputs (decode shapes for Qwen3.6-35B-A3B-4bit MoE)
B, K_EXPERTS, HIDDEN, DIM = 1, 8, 512, 2048
NUM_EXPERTS = 256

gate = mx.random.normal((B, 1, K_EXPERTS, HIDDEN), dtype=mx.bfloat16)
up   = mx.random.normal((B, 1, K_EXPERTS, HIDDEN), dtype=mx.bfloat16)

# Build a fake quantized down_proj (256 experts, dim=2048, hidden=512, 4-bit)
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
down = QuantizedSwitchLinear(HIDDEN, DIM, NUM_EXPERTS, bias=False, group_size=64, bits=4)
down.scales = down.scales.astype(mx.bfloat16)
down.biases = down.biases.astype(mx.bfloat16)
mx.eval(down.parameters())

rhs_indices = mx.random.randint(0, NUM_EXPERTS, (B, 1, K_EXPERTS)).astype(mx.uint32)

# Reference (unfused)
activated = nn.silu(gate) * up
gate_5d = gate.reshape(B, 1, K_EXPERTS, 1, HIDDEN)
up_5d   = up.reshape(B, 1, K_EXPERTS, 1, HIDDEN)
activated_5d = nn.silu(gate_5d) * up_5d
ref = down(activated_5d, rhs_indices).squeeze(-2)
mx.eval(ref)

# Fused — new compiled-in kernel
fused = mx.fast.fused_swiglu_gather_qmv(
    gate.reshape(B, K_EXPERTS, HIDDEN),
    up.reshape(B, K_EXPERTS, HIDDEN),
    down.weight, down.scales, down.biases,
    rhs_indices=rhs_indices.reshape(B, K_EXPERTS),
    group_size=64, bits=4, mode="affine",
)
mx.eval(fused)

diff = mx.abs(ref.astype(mx.float32) - fused.astype(mx.float32))
print(f"max abs diff:  {float(diff.max()):.4e}")  # expect ~7.8e-3 (1 ULP bf16)
print(f"mean abs diff: {float(diff.mean()):.4e}")
print(f"matches MLX precision floor")
```

## Microbench (after install)

```python
import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

HIDDEN, DIM, K, NE = 512, 2048, 8, 256
down = QuantizedSwitchLinear(HIDDEN, DIM, NE, bias=False, group_size=64, bits=4)
down.scales = down.scales.astype(mx.bfloat16)
down.biases = down.biases.astype(mx.bfloat16)
mx.eval(down.parameters())
gate = mx.random.normal((1, 1, K, HIDDEN), dtype=mx.bfloat16)
up   = mx.random.normal((1, 1, K, HIDDEN), dtype=mx.bfloat16)
rhs  = mx.random.randint(0, NE, (1, 1, K)).astype(mx.uint32)
mx.eval(gate, up, rhs)
gate_5d = gate.reshape(1, 1, K, 1, HIDDEN); up_5d = up.reshape(1, 1, K, 1, HIDDEN)

def time_n(fn, w=10, n=500):
    for _ in range(w): mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(n): mx.eval(fn())
    return (time.perf_counter() - t0) / n * 1e6

stock = lambda: down(nn.silu(gate_5d) * up_5d, rhs)
fused = lambda: mx.fast.fused_swiglu_gather_qmv(
    gate.reshape(1, K, HIDDEN), up.reshape(1, K, HIDDEN),
    down.weight, down.scales, down.biases,
    rhs_indices=rhs.reshape(1, K),
)

print(f"stock: {time_n(stock):.1f} us")
print(f"fused: {time_n(fused):.1f} us")
```

Expected: stock ~150μs/call, fused ~135-140μs/call (1.07-1.10× per call). End-to-end model speedup target: ~3-5% over gate+up alone, because this version bypasses `mx.fast.metal_kernel`'s runtime-JIT overhead that ate the gain on the Mac Studio.

## Integration with mlx-fast

After this MLX builds and `mx.fast.fused_swiglu_gather_qmv` is available, update `/Users/ai-studio/code/mlx-fast/mlx_fast/mtp/moe_compile.py`:

```python
# Replace this import:
# from mlx_fast.kernels.fused_swiglu_down import fused_swiglu_down
# with the upstream-MLX version:

def _fused_switch_glu_call_compiled(self, x, indices):
    # ... gate+up fusion as before ...
    dp = self.down_proj
    x = mx.fast.fused_swiglu_gather_qmv(
        x_gate.squeeze(-2), x_up.squeeze(-2),
        dp["weight"], dp["scales"], dp.get("biases"),
        rhs_indices=idx,
        group_size=dp.group_size, bits=dp.bits, mode=dp.mode,
    )
    # ...
```

Then re-run `bench/kernel_fusion_final.py` to confirm the integration speedup translates end-to-end.

## What's in this branch

Commits (newest → oldest):
- `e3506579` — fast: FusedSwiGLUGatherQMV primitive + Python binding
- `18ccbc0a` — docs: MBP build instructions
- `e9ebc3e4` — metal/quantized.cpp: gather_qmv_swiglu dispatch
- `31224ac…` — metal: affine_gather_qmv_swiglu kernel (templates + .metal instantiation)
- `68cf2fd…` — upstream v0.31.2 base
