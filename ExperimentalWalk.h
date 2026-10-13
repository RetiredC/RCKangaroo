// Experimental walk policies. GPLv3, see LICENSE.TXT.
#pragma once

#ifndef RCK_EXPERIMENTAL_SOTA_HALF
#define RCK_EXPERIMENTAL_SOTA_HALF 0
#endif
#ifndef RCK_EXPERIMENTAL_JUMP_CAP
#define RCK_EXPERIMENTAL_JUMP_CAP 0
#endif
#ifndef RCK_EXPERIMENTAL_NARROW_PADDING
#define RCK_EXPERIMENTAL_NARROW_PADDING 0
#endif
#if RCK_EXPERIMENTAL_NARROW_PADDING && !RCK_EXPERIMENTAL_JUMP_CAP
#error "Narrow padding requires RCK_EXPERIMENTAL_JUMP_CAP"
#endif
#ifndef RCK_FORCE_CUDA
#define RCK_FORCE_CUDA 0
#endif
#define RCK_USE_CUDA_WALK (RCK_EXPERIMENTAL_SOTA_HALF || RCK_FORCE_CUDA)

#ifdef __CUDACC__
#define RCK_WALK_INLINE __device__ __forceinline__
#else
#define RCK_WALK_INLINE inline
#endif

namespace ExperimentalWalk {

// The first nine X bits already select one of 512 jumps. Use the next bit
// for continuation selection. Disjoint slices do not prove independence.
constexpr unsigned RankShift = 9;

constexpr int JumpExponent(int range, int table, bool capped) {
    if (table == 0) return range / 2 + 3;
    const int original = range - (table == 1 ? 10 : 12);
    const int bound = range / 2 + (table == 1 ? 16 : 14);
    return capped && bound < original ? bound : original;
}

// Host-only initialization. Keep the target shift and both sampling paths
// on the same margin: h=W/32 normally, h=W/256 in the optional experiment.
// The wild bound is always W+2h, not a fixed multiple of the old margin.
template<class Int>
inline void InitStartMargin(int range, bool narrow, Int& margin) {
    margin.Set(1);
    margin.ShiftLeft(range - (narrow ? 8 : 5));
}

template<class Int>
inline void InitStartBounds(int range, bool narrow, Int& margin, Int& wild) {
    InitStartMargin(range, narrow, margin);
    wild.Set(1);
    wild.ShiftLeft(range);
    wild.Add(margin);
    wild.Add(margin);
}

// Host target preparation can encounter infinity (k=0 after subtracting
// the interval start) or equal points (k=h when adding the margin). The
// upstream affine AddPoints helper handles neither. Infinity is local to
// this preparation and represented as (0,0); the final shifted qG is finite.
template<class Curve, class Point>
inline Point AddStartPoints(Point left, Point right) {
    if (left.x.IsZero() && left.y.IsZero()) return right;
    if (right.x.IsZero() && right.y.IsZero()) return left;
    if (left.x.IsEqual(right.x)) {
        if (left.y.IsEqual(right.y)) return Curve::DoublePoint(left);
        Point infinity;
        infinity.x.SetZero();
        infinity.y.SetZero();
        return infinity;
    }
    return Curve::AddPoints(left, right);
}

template<class Int>
inline void SampleTameDistance(Int& distance, Int& margin, bool reject_zero) {
    // KernelGen cannot represent a zero-scalar tame point. Narrowing raises
    // its probability at small ranges; reject it only in the new experiment
    // so default builds retain upstream's exact sampling/RNG sequence.
    do { distance.RndMax(margin); } while (reject_zero && distance.IsZero());
}

RCK_WALK_INLINE bool SameIndexLoop(unsigned previous, unsigned next) {
    return previous == next;
}

// x/slope initially describe P + oriented_J. inv is 1/(P.x - J.x),
// as in KernelA. Try P - oriented_J only when the preferred rank is one.
// The caller computes Y once and flips the recorded distance sign if true.
// Like KernelA's affine batch, this requires P.x != J.x.
// Ops supplies copy, neg, sub, mul and sqr on four 64-bit field limbs.
template<class Ops>
RCK_WALK_INLINE bool SelectHalfAlternative(
    unsigned long long* x, unsigned long long* slope,
    unsigned long long* px, unsigned long long* py,
    unsigned long long* jx, unsigned long long* oriented_jy,
    unsigned long long* inv) {
    if (((x[0] >> RankShift) & 1) == 0) return false;
    unsigned long long jy[4], other_slope[4], other_x[4];
    Ops::copy(jy, oriented_jy);
    Ops::neg(jy);
    Ops::sub(other_slope, py, jy);
    Ops::mul(other_slope, other_slope, inv);
    Ops::sqr(other_x, other_slope);
    Ops::sub(other_x, other_x, jx);
    Ops::sub(other_x, other_x, px);
    if (((other_x[0] >> RankShift) & 1) != 0) return false;
    Ops::copy(x, other_x);
    Ops::copy(slope, other_slope);
    return true;
}

} // namespace ExperimentalWalk

#undef RCK_WALK_INLINE
