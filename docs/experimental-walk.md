# Experimental SOTA-half selector and large-range jump cap

Based on upstream `main` at `618473acd7f5e696be5ec4f8377c18cc55c1361c`
(checked 2026-10-13; latest release: v4.0). Both changes are **off by
default** and independently selectable. This is a proposal for GPU evaluation,
not a measured speedup of this CUDA implementation.

## Continuation selection

SOTA/SOTA+ and the two-continuation idea are RetiredCoder's work. This port
uses bit 9 of the candidate X coordinate for the half-selector. Bits 0..8
already choose the 512-entry jump-table index; the DP predicate uses the high
word. Using disjoint slices avoids using the same bits directly, but does
not establish statistical independence of the resulting walks.

On J1 steps, first calculate the usual parity-oriented endpoint. If its
rank bit is zero, keep it. Otherwise calculate the opposite-sign endpoint's
X using the same inverse denominator, and choose it only if its rank is
zero. Ties retain the original sign. Calculate Y once for the selected
endpoint. Toggle `INV_FLAG` when selecting the alternate sign so KernelB
replays the actual distance. J2 steps retain the existing rule.

Because the selected sign can now differ from the parity preference,
the J1 two-cycle predictor conservatively requests J2 whenever the next
unsigned jump index equals the current index. It is not a general cycle
detector. Longer-cycle handling remains the upstream KernelB/KernelC logic.

`SelectHalfAlternative` in `ExperimentalWalk.h` is shared by the CUDA
call site and the portable secp256k1 tests. Like upstream KernelA's affine
batch, it assumes distinct input/jump X coordinates. This patch does not
add exceptional-point handling to the existing batch inverter.

## Proposed mitigation for large J2/J3 displacement

Upstream uses exponents `R-10` for J2 and `R-12` for J3. The optional policy is:

```
J1: R/2 + 3                         (unchanged)
J2: min(R-10, R/2 + 16)
J3: min(R-12, R/2 + 14)
```

Integer division is used. Both escape tables remain unchanged through R52
and retain their two-bit exponent difference. At R139, the J2/J3 exponents
change from 129/127 to 85/83. Sampling, even-distance masking, fixed RNG
seed, and point generation stay in the original host code.

The motivation is a possible long-run diffusion problem: under an independent
increment approximation, RMS displacement grows with the square root of
the number of escape jumps. A step of order `2^R/1024` may then be too large
relative to the interval. The actual escape rate, correlations, stationary
distribution and cost of smaller escape jumps need measurement. No R134/R139
solve or NVIDIA experiment currently establishes a slowdown or the benefit
of this cap. The constants above are an experimental policy, not a proved
optimum or a validated production bug fix. No model-derived large-puzzle
speedup is claimed here.

## Build and compare

With a supported CUDA toolkit, make separate build directories:

```sh
# Unchanged default backend and walk.
cmake -S . -B build-default
cmake --build build-default --parallel

# Unchanged walk on CUDA A/B: isolate the selector from the ASM backend.
cmake -S . -B build-control -DRCK_FORCE_CUDA=ON
cmake --build build-control --parallel

# Selector only.
cmake -S . -B build-half -DRCK_EXPERIMENTAL_SOTA_HALF=ON
cmake --build build-half --parallel

# Cap only (may still use the original turbo backend).
cmake -S . -B build-cap -DRCK_EXPERIMENTAL_JUMP_CAP=ON
cmake --build build-cap --parallel

# Both experiments together.
cmake -S . -B build-both -DRCK_EXPERIMENTAL_SOTA_HALF=ON -DRCK_EXPERIMENTAL_JUMP_CAP=ON
cmake --build build-both --parallel
```

Existing toolkit-path configuration still applies. For a cap-only control
using CUDA A/B, also enable `RCK_FORCE_CUDA`. CMake applies the definitions
to both host and CUDA translation units. Other build systems must set the
same definitions consistently for every translation unit; the header
defaults are zero.

SOTA-half explicitly selects compiled CUDA A/B and skips loading the turbo
cubin. The distributed ASM kernels do not implement this selector. A change
in transition count must not be advertised as a wall-time gain over turbo
ASM: the alternate X has extra field work, and the backend also differs.
The unchanged default and forced-CUDA control are both needed for an honest
GPU comparison. Startup prints the selected policy; existing K/DP estimates
remain estimates for the baseline method.

Experimental builds reject `-tames` for both loading and generation. The
current cache header identifies the range, not the walk policy, so it cannot
distinguish these tables from the original ones. `RCK_FORCE_CUDA` alone does
not alter the policy and retains tame support.

## Evidence and remaining validation

Run the portable tests on macOS/Linux with Python 3 and a C++17 compiler:

```sh
python3 tests/run_portable_walk.py
```

These compile the shared helper against upstream `Ec.cpp`, compare selected
points against independent full additions, verify reflection and distance
labels, and check jump-table bounds and points. They do not compile the CUDA
translation unit or execute kernels. CUDA compilation, end-to-end GPU key
recovery, DP behavior, occupancy, wall time and large-range cap behavior
remain to be checked before this proposal is ready to merge.

The motivation came from an earlier **combined CPU prototype**, using this
selector plus chronological DP delivery, Brent cycle detection, salted J3
escapes and a range-first collision filter. On 32 reused synthetic cases
(16 R40/M48/DP14 and 16 R42/M96/DP14), it used 106,519,414 counted group
operations versus 158,847,864 for the original CPU walk rules under the same
fixed restart protocol (up to eight attempts, each capped at `16*sqrt(N)`).
The ratio of normalized work was 1.5808095; the raw sum ratio was 1.4912574.
The selector reused the earlier coalescence holdout, so this was not a fresh
selector holdout. The baseline also used CPU-generated jump tables, not the
official seed-zero GPU run.

Those counts include setup, committed transitions, failed/restarted attempts
and key verification, but do not price the extra alternate-X field work as
an additional group operation. They are **not a benchmark of this patch**:
the coalescence policies are not ported here, the cap was not active in that
prototype, and R40/R42 would not exercise the cap anyway. The attached
`experimental-walk-observations.json` preserves the synthetic fixtures and
recorded counters as historical motivation only. New, separately fixed GPU
cases must compare each option independently through verified key recovery.
