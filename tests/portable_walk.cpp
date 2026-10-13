// GPLv3, see ../LICENSE.TXT. CPU checks of the shared policy helper only.
#include "Ec.h"
#include "ExperimentalWalk.h"
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}
static EcInt scalar(u64 value) { EcInt v; v.Set(value); return v; }
static EcInt power(int bit) { EcInt v = scalar(1); v.ShiftLeft(bit); return v; }
static EcPoint negative(EcPoint p) { p.y.NegModP(); return p; }
static bool equal(EcPoint a, EcPoint b) { return a.IsEqual(b); }
static unsigned rank(EcPoint p) { return (p.x.data[0] >> 9) & 1; }

// Independent field adapter: the production header sees the same four-limb
// interface as CUDA, but uses the repository's unchanged CPU EC arithmetic.
struct FieldOps {
    static unsigned multiplies, squares;
    static EcInt load(const u64* p) {
        EcInt v; v.SetZero(); memcpy(v.data, p, 32); return v;
    }
    static void copy(u64* d, const u64* a) { memmove(d, a, 32); }
    static void neg(u64* a) { auto v = load(a); v.NegModP(); copy(a, v.data); }
    static void sub(u64* d, const u64* a, const u64* b) {
        auto v = load(a), w = load(b); v.SubModP(w); copy(d, v.data);
    }
    static void mul(u64* d, const u64* a, const u64* b) {
        auto v = load(a), w = load(b); v.MulModP(w); copy(d, v.data); ++multiplies;
    }
    static void sqr(u64* d, const u64* a) {
        auto v = load(a), w = v; v.MulModP(w); copy(d, v.data); ++squares;
    }
};
unsigned FieldOps::multiplies = 0, FieldOps::squares = 0;

struct Step { EcPoint point; int sign; bool alternative; unsigned rank_pair; };
static Step step(EcPoint p, EcPoint jump, bool select) {
    require(!p.x.IsEqual(jump.x), "affine precondition violated");
    int sign = (p.y.data[0] & 1) ? -1 : 1;
    EcPoint oriented = sign < 0 ? negative(jump) : jump;
    EcPoint opposite = negative(oriented);
    // Reference calls compute complete endpoints independently of the helper.
    EcPoint a = Ec::AddPoints(p, oriented), c = Ec::AddPoints(p, opposite);
    const bool expected = select && rank(a) == 1 && rank(c) == 0;
    const auto px_before = p.x, py_before = p.y, jx_before = jump.x, jy_before = oriented.y;
    EcInt inv = p.x; inv.SubModP(jump.x); inv.InvModP();
    EcInt slope = p.y; slope.SubModP(oriented.y); slope.MulModP(inv);
    EcInt x = slope; x.MulModP(slope); x.SubModP(jump.x); x.SubModP(p.x);
    FieldOps::multiplies = FieldOps::squares = 0;
    const bool chosen = select && ExperimentalWalk::SelectHalfAlternative<FieldOps>(
        x.data, slope.data, p.x.data, p.y.data, jump.x.data, oriented.y.data, inv.data);
    require(chosen == expected, "wrong rank selection or tie handling");
    require(FieldOps::multiplies == (select && rank(a) ? 1u : 0u), "extra alternative multiplication");
    require(FieldOps::squares == (select && rank(a) ? 1u : 0u), "extra alternative square");
    // One Y multiplication after selection; there is no Y calculation in the helper.
    EcPoint out; out.x = x; out.y = p.x; out.y.SubModP(x);
    out.y.MulModP(slope); out.y.SubModP(p.y);
    require(equal(out, expected ? c : a), "selected point differs from Ec::AddPoints");
    require(Ec::IsValidPoint(out), "selected point is not on secp256k1");
    require(memcmp(p.x.data, px_before.data, 40) == 0 &&
            memcmp(p.y.data, py_before.data, 40) == 0 &&
            memcmp(jump.x.data, jx_before.data, 40) == 0 &&
            memcmp(oriented.y.data, jy_before.data, 40) == 0, "helper changed inputs");
    return {out, chosen ? -sign : sign, chosen, 2 * rank(a) + rank(c)};
}

static void selector_checks() {
    unsigned coverage[2][4] = {}, alternatives = 0;
    for (unsigned i = 0; i < 1024; ++i) {
        EcInt d = power(200), offset = scalar(101 * i + 17); d.Add(offset);
        EcInt jd = scalar(2 * (i + 1));
        EcPoint p = Ec::MultiplyG(d), jump = Ec::MultiplyG(jd);
        for (bool reflect : {false, true}) {
            EcPoint input = reflect ? negative(p) : p;
            const auto result = step(input, jump, true);
            ++coverage[input.y.data[0] & 1][result.rank_pair];
            alternatives += result.alternative;
            auto mirrored = step(negative(input), jump, true);
            require(equal(mirrored.point, negative(result.point)), "reflection equivariance failed");
            require(mirrored.sign == -result.sign, "reflected distance sign failed");
            // Convert reflected input back to a positive scalar before MultiplyG.
            EcInt next = d;
            const int sign = reflect ? -result.sign : result.sign;
            if (sign > 0) next.Add(jd); else require(!next.Sub(jd), "scalar underflow");
            EcPoint labelled = Ec::MultiplyG(next);
            require(equal(result.point, reflect ? negative(labelled) : labelled), "recorded sign/point mismatch");
            (void)step(input, jump, false);
        }
    }
    for (auto& signs : coverage) for (unsigned count : signs)
        require(count > 0, "missing sign/rank-pair coverage");
    std::cout << "selector: 2048 signed cases, all 8 sign/rank pairs, "
              << alternatives << " alternatives; reflection, baseline, labels PASS\n";
}

struct Jump { EcInt distance; EcPoint point; };
using Tables = std::array<std::vector<Jump>, 3>;
static Tables tables(int range, bool capped, bool points) {
    SetRndSeed(0); // Same single RNG stream/order as RCKangaroo.cpp.
    Tables result;
    for (int t = 0; t < 3; ++t) {
        EcInt low = power(ExperimentalWalk::JumpExponent(range, t, capped));
        EcInt high = low; high.ShiftLeft(1);
        for (int i = 0; i < JMP_CNT; ++i) {
            EcInt random; random.RndMax(low);
            Jump j; j.distance = low; j.distance.Add(random); j.distance.data[0] &= ~1ULL;
            require(!j.distance.IsZero() && !(j.distance.data[0] & 1), "zero/odd jump");
            require(!j.distance.IsLessThanU(low) && j.distance.IsLessThanU(high), "jump outside exponent bounds");
            if (points) {
                j.point = Ec::MultiplyG(j.distance);
                require(Ec::IsValidPoint(j.point), "jump point is invalid");
                EcInt half = j.distance; half.ShiftRight(1);
                EcPoint h = Ec::MultiplyG(half), doubled = Ec::DoublePoint(h);
                require(equal(j.point, doubled), "jump scalar/point mismatch");
            }
            result[t].push_back(j);
        }
    }
    return result;
}

static void exponent_checks() {
    unsigned scalar_checks = 0, point_checks = 0;
    for (int range = 32; range <= 170; ++range) {
        for (int t = 0; t < 3; ++t) {
            int original = t == 0 ? range / 2 + 3 : range - (t == 1 ? 10 : 12);
            int cap = range / 2 + (t == 1 ? 16 : 14);
            int expected = t == 0 || original < cap ? original : cap;
            require(ExperimentalWalk::JumpExponent(range, t, false) == original, "baseline exponent changed");
            require(ExperimentalWalk::JumpExponent(range, t, true) == expected, "wrong capped exponent");
            if (range <= 52) require(expected == original, "small-range cap changed jumps");
        }
        require(ExperimentalWalk::JumpExponent(range, 1, true) -
                ExperimentalWalk::JumpExponent(range, 2, true) == 2, "J2/J3 exponent gap changed");
        const bool points = range == 139 || range == 170;
        auto original = tables(range, false, points), capped = tables(range, true, points);
        for (int t = 0; t < 3; ++t) for (int i = 0; i < JMP_CNT; ++i) {
            if (range <= 52 || t == 0)
                require(original[t][i].distance.IsEqual(capped[t][i].distance), "seed-0 compatibility changed");
            scalar_checks += 2;
            if (points) point_checks += 2;
        }
    }
    require(ExperimentalWalk::JumpExponent(52, 1, true) == 42 &&
            ExperimentalWalk::JumpExponent(53, 1, false) == 43 &&
            ExperimentalWalk::JumpExponent(53, 1, true) == 42, "52/53 boundary failed");
    std::cout << "jump policy: R32..170, " << scalar_checks << " seed-0 scalars; "
              << point_checks << " R139/R170 point checks PASS\n";
}

static void trajectory_checks() {
    const auto jump = tables(40, false, true);
    unsigned verified = 0, j2_steps = 0;
    for (bool select : {false, true}) {
        EcInt distance = power(180), start = scalar(1234567); distance.Add(start);
        EcPoint p = Ec::MultiplyG(distance);
        bool next_j2 = false;
        for (int i = 0; i < 8192; ++i) {
            unsigned index = p.x.data[0] & JMP_MASK;
            const bool was_j2 = next_j2;
            Jump j = jump[was_j2 ? 1 : 0][index];
            const auto result = step(p, j.point, select && !was_j2);
            if (result.sign > 0) distance.Add(j.distance);
            else require(!distance.Sub(j.distance), "walk scalar underflow");
            const unsigned next_index = result.point.x.data[0] & JMP_MASK;
            next_j2 = !was_j2 && (select ? ExperimentalWalk::SameIndexLoop(index, next_index) :
                (index == next_index && bool(result.sign < 0) != bool(result.point.y.data[0] & 1)));
            p = result.point;
            require(equal(p, Ec::MultiplyG(distance)), "walk replay scalar/point mismatch");
            j2_steps += was_j2; ++verified;
        }
    }
    require(j2_steps > 0, "trajectory did not exercise J2");
    for (unsigned i = 0; i < JMP_CNT; ++i) for (unsigned j = 0; j < JMP_CNT; ++j)
        require(ExperimentalWalk::SameIndexLoop(i, j) == (i == j), "same-index predictor failed");
    std::cout << "trajectory: " << verified << " full scalar replays, " << j2_steps
              << " J2 steps; all 262144 predictor pairs PASS\n";
}

int main() {
    try {
        InitEc(); selector_checks(); exponent_checks(); trajectory_checks(); DeInitEc();
        std::cout << "PASS (portable CPU helper checks; no CUDA/ASM execution or performance claim)\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n'; return EXIT_FAILURE;
    }
}
