// GPL-3.0; compatibility glue for unmodified RetiredC/RCKangaroo sources.
// Only the two x86 intrinsics used by pinned Ec.cpp are provided.
#pragma once
#include <cstdint>

inline unsigned char _addcarry_u64(unsigned char carry, unsigned long long a,
                                   unsigned long long b, unsigned long long* out) {
    const __uint128_t sum = (__uint128_t)a + b + carry;
    *out = (unsigned long long)sum;
    return (unsigned char)(sum >> 64);
}

inline unsigned char _subborrow_u64(unsigned char borrow, unsigned long long a,
                                    unsigned long long b, unsigned long long* out) {
    const __uint128_t subtrahend = (__uint128_t)b + borrow;
    *out = a - (unsigned long long)subtrahend;
    return (unsigned char)((__uint128_t)a < subtrahend);
}
