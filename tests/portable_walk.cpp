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

struct ZeroThenSeven {
    unsigned value = 0, calls = 0;
    void RndMax(ZeroThenSeven& limit) {
        require(limit.value == 16, "mock received wrong tame bound");
        value = ++calls < 3 ? 0 : 7;
    }
    bool IsZero() const { return value == 0; }
};

static void padding_sampling_checks() {
    unsigned bounds = 0, samples = 0, defaults = 0;
    for (int range = 32; range <= 170; ++range) {
        for (bool narrow : {false, true}) {
            EcInt margin, wild, target_margin;
            ExperimentalWalk::InitStartBounds(range, narrow, margin, wild);
            ExperimentalWalk::InitStartMargin(range, narrow, target_margin);
            auto expected_margin = power(range - (narrow ? 8 : 5));
            require(margin.IsEqual(expected_margin) && margin.IsEqual(target_margin),
                    "sampling/target margin mismatch");
            // Independent old-style repeated addition catches leaving 34*h
            // after narrowing the margin, which would drop most of the range.
            EcInt repeated = scalar(0);
            for (int i = 0; i < (narrow ? 258 : 34); ++i) repeated.Add(margin);
            require(wild.IsEqual(repeated), "wild support is not W+2h");
            auto width = power(range), excess = wild;
            require(!excess.Sub(width), "wild support narrower than search interval");
            auto twice = margin; twice.ShiftLeft(1);
            require(excess.IsEqual(twice), "wrong padding at wild boundaries");
            require(!(margin.data[0] & 1) && !(wild.data[0] & 1), "odd padding bounds");
            require(wild.data[3] == 0 && wild.data[4] == 0, "bound exceeds 24-byte distance");
            ++bounds;
            SetRndSeed(0x50414400ULL + range);
            for (int i = 0; i < 64; ++i) {
                EcInt tame, distance;
                ExperimentalWalk::SampleTameDistance(tame, margin, narrow);
                distance.RndMax(wild); distance.data[0] &= ~1ULL;
                require(tame.IsLessThanU(margin) && (!narrow || !tame.IsZero()), "invalid tame start");
                require(distance.IsLessThanU(wild) && !(distance.data[0] & 1), "invalid wild start");
                require(tame.data[3] == 0 && tame.data[4] == 0 &&
                        distance.data[3] == 0 && distance.data[4] == 0, "sample exceeds 24-byte distance");
                samples += 2;
            }
        }
        // Preserve the entire old alternating RNG stream, not only bounds.
        EcInt old_margin = power(range - 5), old_wild = scalar(0);
        for (int i = 0; i < 34; ++i) old_wild.Add(old_margin);
        std::array<EcInt, 64> expected;
        SetRndSeed(0x53454544ULL + range);
        for (unsigned i = 0; i < expected.size(); ++i) {
            expected[i].RndMax(i % 3 == 0 ? old_margin : old_wild);
            if (i % 3) expected[i].data[0] &= ~1ULL;
        }
        EcInt margin, wild;
        ExperimentalWalk::InitStartBounds(range, false, margin, wild);
        SetRndSeed(0x53454544ULL + range);
        for (unsigned i = 0; i < expected.size(); ++i) {
            EcInt actual;
            if (i % 3 == 0) ExperimentalWalk::SampleTameDistance(actual, margin, false);
            else { actual.RndMax(wild); actual.data[0] &= ~1ULL; }
            require(actual.IsEqual(expected[i]), "default seeded start sequence changed");
            ++defaults;
        }
    }
    ZeroThenSeven bound; bound.value = 16;
    ZeroThenSeven original, narrow;
    ExperimentalWalk::SampleTameDistance(original, bound, false);
    ExperimentalWalk::SampleTameDistance(narrow, bound, true);
    require(original.IsZero() && original.calls == 1, "default zero/RNG behavior changed");
    require(narrow.value == 7 && narrow.calls == 3, "narrow tame zero rejection failed");
    std::cout << "padding sampling: " << bounds << " R32..170 bounds, " << samples
              << " samples, " << defaults << " default RNG matches; zero rejection PASS\n";
}

static void padding_collision_checks() {
    unsigned cases = 0, collisions = 0, target_shifts = 0;
    for (int range : {32, 40, 52, 53, 78, 139, 170}) {
        for (bool narrow : {false, true}) {
            EcInt h, bound;
            ExperimentalWalk::InitStartBounds(range, narrow, h, bound);
            auto width = power(range), half = power(range - 1);
            auto half_before = half, half_after = half, end_before = width, end = width;
            auto one = scalar(1), two = scalar(2);
            half_before.Sub(one); half_after.Add(one); end_before.Sub(two); end.Sub(one);
            auto h_before = h, h_after = h; h_before.Sub(one); h_after.Add(one);
            std::array<EcInt, 11> targets = {scalar(0), one, two, h_before, h, h_after,
                half_before, half, half_after, end_before, end};
            for (auto k : targets) {
                auto q = k; q.Add(h);
                // Tame parity must match q so both constructed wild labels
                // are even. A small suffix avoids the legacy low64 duplicate
                // filter when testing margins wider than 64 bits.
                auto tame = h; tame.ShiftRight(1);
                auto suffix = scalar(2 + (q.data[0] & 1)); tame.Add(suffix);
                require(!tame.IsZero() && tame.IsLessThanU(h), "collision tame outside support");
                auto lower = q, upper = q;
                require(!lower.Sub(tame), "collision wild label became negative");
                upper.Add(tame);
                require(lower.IsLessThanU(bound) && upper.IsLessThanU(bound) &&
                        !(lower.data[0] & 1) && !(upper.data[0] & 1), "endpoint wild coverage/parity failed");
                require(lower.data[0] != upper.data[0], "wild fixture hits low64 duplicate filter");
                auto q_point = Ec::MultiplyG(q), t_point = Ec::MultiplyG(tame);
                auto lower_base = Ec::MultiplyG(lower), upper_base = Ec::MultiplyG(upper);
                auto negative_q = negative(q_point);
                auto lower_point = Ec::AddPoints(lower_base, negative_q);
                auto upper_point = Ec::AddPoints(upper_base, negative_q);
                require(equal(upper_point, t_point) && equal(lower_point, negative(t_point)),
                        "tame/wild actual point relation failed");
                require(equal(upper_point, negative(lower_point)), "wild/wild reflection failed");
                auto start = power(200), offset = scalar(12345); start.Add(offset);
                auto original_key = start; original_key.Add(k);
                auto original_point = Ec::MultiplyG(original_key);
                // Exercise the actual main-mode order: subtract start first,
                // then add h. k=0 produces intermediate infinity; k=h needs
                // doubling. The zero absolute scalar is a helper sentinel
                // case only, not a valid user-supplied public key.
                for (bool with_start : {false, true}) {
                    auto start_value = with_start ? start : scalar(0);
                    auto absolute = k; absolute.Add(start_value);
                    EcPoint source, minus_start;
                    if (!absolute.IsZero()) source = Ec::MultiplyG(absolute);
                    if (!start_value.IsZero()) minus_start = negative(Ec::MultiplyG(start_value));
                    auto translated = ExperimentalWalk::AddStartPoints<Ec>(source, minus_start);
                    if (k.IsZero())
                        require(translated.x.IsZero() && translated.y.IsZero(), "offset subtraction lost infinity");
                    auto margin_point = Ec::MultiplyG(h);
                    auto shifted = ExperimentalWalk::AddStartPoints<Ec>(translated, margin_point);
                    require(equal(shifted, q_point), "actual host target shift failed at boundary/doubling");
                    ++target_shifts;
                }
                auto verify_recovery = [&](EcInt trial) {
                    require(trial.IsEqual(q) && equal(Ec::MultiplyG(trial), q_point), "shifted collision recovery failed");
                    require(!trial.Sub(h) && trial.IsEqual(k), "padding subtraction recovered wrong offset");
                    trial.Add(start);
                    require(trial.IsEqual(original_key) && equal(Ec::MultiplyG(trial), original_point),
                            "nonzero-start public-key recovery failed");
                    ++collisions;
                };
                auto tw_same = upper; tw_same.Sub(tame); verify_recovery(tw_same);
                auto tw_opposite = lower; tw_opposite.Add(tame); verify_recovery(tw_opposite);
                auto ww = lower; ww.Add(upper);
                require(!(ww.data[0] & 1), "wild/wild recovery is not integral");
                ww.ShiftRight(1); verify_recovery(ww);
                ++cases;
            }
        }
    }
    std::cout << "padding collisions: " << cases << " endpoint/center cases, " << collisions
              << " TW/WW point relations and nonzero-start recoveries; " << target_shifts
              << " actual host target shifts PASS\n";
}

int main() {
    try {
        InitEc(); selector_checks(); exponent_checks(); trajectory_checks();
        padding_sampling_checks(); padding_collision_checks(); DeInitEc();
        std::cout << "PASS (portable CPU helper checks; no CUDA/ASM execution or performance claim)\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n'; return EXIT_FAILURE;
    }
}
