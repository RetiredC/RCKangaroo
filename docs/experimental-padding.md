# Narrower starting margin: conditional model and integration

This is an opt-in experiment, not a measured CUDA speedup. Compare
`RCK_EXPERIMENTAL_JUMP_CAP=ON` against that same build with
`RCK_EXPERIMENTAL_NARROW_PADDING=ON`; keep selector and backend identical.
The margin option is off by default. Normal default sampling is unchanged;
target preparation additionally handles exceptional point additions.

## Starting intervals and key recovery

For interval width `W=2^R`, let `h` be the edge margin. Upstream uses `W/32`;
the experiment uses `W/256`. Let `k` be the secret offset inside the interval
and `q=k+h` the shifted target. Tame labels are sampled below `h`; wild labels
are even and below `W+2h`. Thus a wild point has scalar `w-q`.
The final recovered offset is `q-h` in both main and benchmark modes.
Both initial and restart generation use these same bounds. The new host
sampler excludes zero tame labels to avoid an invalid point in KernelGen.
Target preparation handles doubling at `k=h` and intermediate infinity at
`k=0` after subtracting a nonzero start. Both cases would otherwise enter
the upstream affine helper with a zero denominator. This also fixes those
inputs in default mode. The finite-lattice model correction from excluding
zero tame starts is ignored below.

Simply replacing `Range-5` with `Range-8` while retaining `34*h` would shrink
the wild width to only `0.1328125*W`. Changing the sampling margin without
changing the target shift loses overlap at an edge; changing the shift but
not recovery gives an incorrect key. The shared helpers avoid these mismatches.

No walking-kernel instructions, state buffers, jump-table sizes or backend
selection are added by the margin option. Host setup and starting samples
change. Kernel instruction counts, occupancy and wall time have not been
measured or compiled with NVCC in this validation.

## Ideal overlap calculation

Normalize `W=1`, put `a=h/W`, `L=1+2a`, and fold the uniformly distributed
unknown target by symmetry to `q in [a,a+1/2]`. Assume independent samples,
zero diffusion/drift, uniform positions within the starting supports, equal
positive/negative orientation, one-third tame and two-thirds wild walkers,
and upstream's parity split. Account for useful tame-wild and opposite-sign
wild-wild encounters; same-herd duplicates do not solve the target.

The birthday exponent is `C(q)*s^2` at normalized work `s`, with

```
C_TW = 4/(9*L)
C_WW = 8*q/(9*L*L)
K(q) = sqrt(pi)/(2*sqrt(C_TW + C_WW))
K(a) = 2 * integral(K(q), q=a..a+1/2)
     = (3*sqrt(pi)/2) * L**(3/2)
       * (sqrt(2) - sqrt(1 + 2*a/L))
```

The TW term includes the half of tame labels in the active parity class.
For the WW term, the interval intersection of the wild distribution and
its reflection has length `2*q` on the folded half of the target interval.
These assumptions are an ideal collision model, not properties proved for
the deterministic secp256k1 walk.

| Margin | Exact ideal average K |
| --- | ---: |
| W/32 | 1.1216845636335053 |
| W/256 | 1.1037863848425065 |
| Zero margin (limiting model) | 1.1012616355882270 |

The work ratio is **1.01621525599**, or **1.595652% fewer steps**. At a
fixed step rate this would correspond to 1.621526% higher throughput, but
neither the step-rate premise nor the collision model has been validated
on a GPU. For a fixed target the ratio ranges from 1.00170451 at either
endpoint to 1.02677338 at the center. The latter is also the TW-only
prediction `sqrt(L_old/L_new)`; it is not the average when WW encounters
are counted. Width excess (6.25%) is not density loss (5.882% relative to
zero margin), and neither directly equals the change in K.

Reproduce the closed form and independently integrate the conditional K:

```python
from math import sqrt, pi

def exact(a):
    L = 1 + 2*a
    return 1.5*sqrt(pi)*L**1.5*(sqrt(2)-sqrt(1+2*a/L))

def midpoint(a, n=100000):
    L = 1 + 2*a
    return sum(sqrt(pi)/(2*sqrt(4/(9*L)+8*(a+(i+.5)/(2*n))/(9*L*L)))
               for i in range(n))/n

for a in (1/32, 1/256):
    assert abs(exact(a)-midpoint(a)) < 1e-10
    print(a, exact(a))
print('work ratio:', exact(1/32)/exact(1/256))
```

## Diffusion and validation boundary

At R139, capped J1/J2 exponents are 72/85. Under independent zero-mean
increments and J2 rate `p2`, a walker after `t` transitions would have

```
variance/W^2 = t * (7/3) * ((1-p2)*2**144 + p2*2**170) / 2**278
```

For `t = K(W/256)*2**69.5 / (100*786432)` and `p2=1/1024`, this gives
`sigma/W = 9.0704e-12`; using `p2=1/512` gives `1.2827e-11`.
These are explicit conditional horizons, not observed drift. This formula
omits J3, correlations and any systematic displacement. It does not prove
that the margin remains sufficient in a real long run. The upstream comment
says the margin smooths edges; it does not establish that J2 was its sole cause.

Portable CPU checks validate bounds and synthetic point/label recovery,
including edge neighborhoods and nonzero interval offsets. They do not
simulate a full GPU solve. A real comparison must include host setup, every
transition and escape, DP delay/storage, useless meetings, restarts, memory
and final key verification, using separately fixed identical synthetic
cases and the same backend. Smaller escape jumps and narrower starts may
change cycles, correlations and DP arrival patterns. CUDA compilation,
end-to-end NVIDIA recovery and timed comparisons remain outstanding.
