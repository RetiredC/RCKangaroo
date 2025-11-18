# RCKangaroo v1.5 (local build)

GPU-accelerated **Kangaroo ECDLP** implementation with improvements focused on
**compute time** and **memory / I/O** efficiency.

> Tested with CUDA 12.x and NVIDIA RTX 3060 (SM 8.6).  
> This branch keeps the **original CLI** and adds flags/guides for *tame tuning* and *benchmarks*.

---

---

## 🚀 What’s New in v1.6

### GPU Improvements
- **Warp-aggregated atomics for DP emission**: reduced per-thread atomics to a single warp-level atomic, coalesced writes. **+10–30% throughput** depending on GPU and -dp.
- **Better memory coalescing** for DPs and PCIe transfers.

### New `.dat` Format (v1.6)
- **28B per DP record** (vs 32B in v1.5).
  - X tail: 5 bytes (was 9).
  - Distance: 22 bytes.
  - Type: 1 byte.
- **File tag `TMBM16`** identifies new format.
- **Backward compatible**: reads both v1.5 and v1.6.

### Benchmarks (RTX 3060)
- v1.5: ~750 MKey/s @ -dp 16.
- v1.6: ~870 MKey/s @ -dp 16.
- ~16% faster and ~12.5% smaller `.dat` files.



## Technical Highlights (V1.5)

1) **Jacobian Coordinates on GPU** (opt‑in)
   - Point add/double in Jacobian to avoid modular inversions per step.
   - Convert back to affine **only when needed** (e.g., DPs or output).
   - Mixed add (*Jacobian + Affine precomp*) for jump points.
   - Build switch: `USE_JACOBIAN=1` (enabled by default in this branch).

2) **Batch Inversion (Montgomery trick)**
   - Invert many `Z` values with **a single** field inversion using forward/backward products.
   - Useful for compacting/normalizing states and bulk verifications.

3) **TAMES v1.5 – compact file format**
   - **~30–35% smaller** than the classic layout in our tests (e.g., 84 MB → 57 MB).
   - Contiguous layout + light compression (delta + varint/RLE) with streaming reads.
   - Faster load and lower cache/L2/PCIe pressure.
   - **Compatibility**: the binary still accepts the classic format; if the file is not v1.5, it uses the legacy path.

4) **Less I/O and optimized binary size**
   - Host flags `-ffunction-sections -fdata-sections`; device fatbin compression `-Xfatbin=-compress-all`.
   - L1/tex cache hint via `-Xptxas -dlcm=ca` in `build.sh`.

> Note: *Montgomery Ladder* is available in code but not enforced via CLI; Jacobian + classic/mixed windows showed a better perf/resource balance on Ampere.

---

## Modified / Added Files

- **`RCGpuCore.cu`**  
  Jacobian implementations (double/mixed-add), batch inversion path, and kernel selection via `USE_JACOBIAN`.

- **`RCGpuUtils.h`**  
  Field primitives and helpers for Jacobian (double / mixed add).

- **`utils.h`, `utils.cpp`**  
  - New **TAMES v1.5 reader/writer** (streaming, compact).  
  - Utility cleanups.

- **`GpuKang.cpp`, `GpuKang.h`**  
  - Exposed *tame tuning* parameters (ratio and bits) for controlled benches.
  - Distance generation and stable tame/wild partitioning.

- **`RCKangaroo.cpp`**  
  - CLI parsing + guard rails (consistent error messages).  
  - More verbose bench output.

- **`Makefile`**  
  - Direct `rckangaroo` target (no intermediate archives).  
  - Support for `SM`, `USE_JACOBIAN`, `PROFILE`, and deterministic linking.

- **Helper scripts**  
  - `build.sh` – multi‑SM build wrapper.
  - `bench_grid.sh` – parameter sweep (dp / tame-bits / tame-ratio) with repeats and logs.
  - `bench_rck.sh` – quick A/B benchmark.
  - `summarize_bench.py` – log parser → CSV (speed, wall time, RSS, parameters).

---

## Project Tree (this branch)

```
.
├── logs/                          # output from bench_grid.sh
├── bench_grid.sh
├── bench_rck.sh
├── build.sh
├── Makefile
├── defs.h
├── Ec.cpp
├── Ec.h
├── GpuKang.cpp
├── GpuKang.h
├── RCGpuCore.cu
├── RCGpuUtils.h
├── RCKangaroo.cpp
├── rckangaroo                   # binary after build
├── summarize_bench.py
├── tames71.dat                  # classic format example
├── tames71_v15.dat              # v1.5 compact example
├── utils.cpp
└── utils.h
```

---

## Build

### Option A – `build.sh` (recommended)
```bash
# Syntax: ./build.sh <SM> <USE_JACOBIAN 0|1> <profile: release|debug>
./build.sh 86 1 release     # RTX 3060 (SM 8.6), Jacobian ON
./build.sh 86 0 release     # Jacobian OFF (affine) for A/B
```
Produces `./rckangaroo` in the current directory.

### Option B – `make`
```bash
# Variables: SM, USE_JACOBIAN, PROFILE=(release|debug)
make SM=86 USE_JACOBIAN=1 PROFILE=release -j
```

> Requirements: CUDA 12+, `g++` with C++17, and a driver supporting your target SM.


---

## Usage (CLI)

Minimal example (with TAMES v1.5):
```bash
./rckangaroo \
  -pubkey 0290e6900a58d33393bc1097b5aed31f2e4e7cbd3e5466af958665bc0121248483 \
  -range 71 \
  -dp 16 \
  -start 0 \
  -tames tames71_v15.dat
```

Tame tuning parameters (reflected in bench logs):
```
  -tame-bits <N>      # bits used for tame jumps (e.g., 4–7)
  -tame-ratio <PCT>   # percent of tame kangaroos (e.g., 25–50)
```
Example:
```bash
./rckangaroo ... -tame-bits 4 -tame-ratio 33
```

> Goal: find combinations that **maximize MKeys/s** while minimizing **wall time** and keeping **memory** acceptable.


---

## Automated Benchmarks

### Parameter sweep (grid)
```bash
# Edit the header of the script to set PUBKEY/RANGE/DP/TAMES/etc.
chmod +x bench_grid.sh summarize_bench.py

# Run grid (stores everything under logs/)
./bench_grid.sh

# Summarize to CSV
python3 summarize_bench.py logs > summary.csv
column -s, -t < summary.csv | less -S
```
Jacobian OFF/ON comparison:
```bash
# Jacobian ON
./build.sh 86 1 release && MODE_TAG="j1" ./bench_grid.sh
python3 summarize_bench.py logs > summary_j1.csv

# Jacobian OFF
./build.sh 86 0 release && MODE_TAG="j0" ./bench_grid.sh
python3 summarize_bench.py logs > summary_j0.csv
```

> **TIP**: Use `REPEATS>=5` to reduce jitter; the parser reports **medians** per combination.


---

## Reference Results (indicative)

Quick 71‑bit tests on an RTX 3060:
- **TAMES v1.5**: 84 MB → **57 MB** (~32% smaller).  
- **Wall time**: ~100 s → **~65 s** (Jacobian + v1.5 with same parameters).  
- **RSS**: slight reduction (≈ −20–30 MB depending on run).

> Numbers vary with DP, *tame-bits*, *tame-ratio*, GPU clocks, and driver version.


---

## Troubleshooting

- **`Unknown option -ffunction-sections` from NVCC**: use `build.sh` (passes via `-Xcompiler`).  
- **`No rule to make target 'RCGpuCore.o'`**: use this repo/Makefile or `./build.sh`.  
- **`CUDA error / cap mismatch`**: compile via `./build.sh <your SM> ...` (e.g., 75 for Turing, 86 for Ampere).


---

## License

Inherits the original project’s license (see `LICENSE.TXT` if present).  
Permitted for research and educational use.
