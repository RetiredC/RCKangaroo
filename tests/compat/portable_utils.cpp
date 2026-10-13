// GPL-3.0; CPU-only helpers for unmodified RetiredC/RCKangaroo Ec.cpp.
// Based on interfaces in pinned utils.h:42-49 and utils.cpp:14-54.
// Does not replace the EC algorithms, RNG, GPU code or DP database.
#include "utils.h"
#include <cassert>
#include <chrono>

void _BitScanReverse64(u32* index, u64 mask) {
    assert(mask != 0);  // Existing Ec.cpp callers require a nonzero operand.
    *index = 63 - __builtin_clzll(mask);
}
void _BitScanForward64(u32* index, u64 mask) {
    *index = __builtin_ffsll(mask) - 1;
}
u64 _umul128(u64 first, u64 second, u64* high) {
    const __uint128_t value = (__uint128_t)first * second;
    *high = (u64)(value >> 64);
    return (u64)value;
}
u64 __shiftright128(u64 low, u64 high, u8 shift) {
    // x86 SHRD masks the count to six bits, including the zero case.
    return (u64)((((__uint128_t)high << 64) | low) >> (shift & 63));
}
u64 __shiftleft128(u64 low, u64 high, u8 shift) {
    // Return the high limb of SHLD's conceptual 128-bit left shift.
    return (u64)(((((__uint128_t)high << 64) | low) << (shift & 63)) >> 64);
}
u64 GetTickCount64() {
    return (u64)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
