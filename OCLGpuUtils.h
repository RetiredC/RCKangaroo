// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC

// Helper functions for OpenCL C to replace PTX assembly with carry
// u32, u64 are defined in defs.h

// Adds a, b, and carry_in. Returns the sum. Sets carry_out to 1 if there was a carry, 0 otherwise.
static inline u32 ocl_add_carry_u32(u32 a, u32 b, u32 carry_in, u32* carry_out) {
    u64 sum_long = (u64)a + (u64)b + (u64)carry_in;
    *carry_out = (sum_long > 0xFFFFFFFFU) ? 1 : 0;
    return (u32)sum_long;
}

// Adds a, b, and carry_in. Returns the sum. Sets carry_out to 1 if there was a carry, 0 otherwise.
static inline u64 ocl_add_carry_u64(u64 a, u64 b, u64 carry_in, u64* carry_out) {
    u64 res_ab = a + b;
    u64 c_ab = (res_ab < a); // Carry from a + b
    u64 final_res = res_ab + carry_in;
    u64 c_res_cin = (final_res < res_ab); // Carry from (a+b) + carry_in
    *carry_out = c_ab | c_res_cin;
    return final_res;
}

// Subtracts b and borrow_in from a. Returns the difference. Sets borrow_out to 1 if there was a borrow, 0 otherwise.
static inline u32 ocl_sub_borrow_u32(u32 a, u32 b, u32 borrow_in, u32* borrow_out) {
    u64 val_a = (u64)a;
    u64 val_b = (u64)b;
    u64 val_bin = (u64)borrow_in;
    u64 diff_long = val_a - val_b - val_bin;
    *borrow_out = (diff_long > 0xFFFFFFFFU) ? 1 : 0; // If it wrapped around (became very large positive)
    return (u32)diff_long;
}

// Subtracts b and borrow_in from a. Returns the difference. Sets borrow_out to 1 if there was a borrow, 0 otherwise.
static inline u64 ocl_sub_borrow_u64(u64 a, u64 b, u64 borrow_in, u64* borrow_out) {
    u64 res_ab = a - b;
    u64 b_ab = (res_ab > a); // Borrow from a - b
    u64 final_res = res_ab - borrow_in;
    u64 b_res_bin = (final_res > res_ab); // Borrow from (a-b) - borrow_in
    *borrow_out = b_ab | b_res_bin;
    return final_res;
}

// These helpers will be used to replace the asm volatile macros.
// The global carry flag management used in CUDA PTX (where one instruction sets CC and next uses it)
// needs to be converted to explicit carry variables in OpenCL C.
// This means functions like AddModP, SubModP, etc., will need local variables to store and pass these carries.

// Helper functions for 192-bit addition/subtraction
static inline void ocl_add192(u64* res, const u64* val) {
    u64 carry64 = 0;
    res[0] = ocl_add_carry_u64(res[0], val[0], 0, &carry64);
    res[1] = ocl_add_carry_u64(res[1], val[1], carry64, &carry64);
    u64 dummy_carry;
    res[2] = ocl_add_carry_u64(res[2], val[2], carry64, &dummy_carry);
}

static inline void ocl_sub192(u64* res, const u64* val) {
    u64 borrow64 = 0;
    res[0] = ocl_sub_borrow_u64(res[0], val[0], 0, &borrow64);
    res[1] = ocl_sub_borrow_u64(res[1], val[1], borrow64, &borrow64);
    u64 dummy_borrow;
    res[2] = ocl_sub_borrow_u64(res[2], val[2], borrow64, &dummy_borrow);
}


//PTX asm
//"volatile" is important
// Note: Macros for operations with carry flags (add_cc, addc, sub_cc, subc, etc.)
// are being removed and replaced by direct calls to ocl_add_carry_u32/u64
// and ocl_sub_borrow_u32/u64 helper functions within the main code logic.
// This requires refactoring of functions like AddModP, SubModP, etc.,
// to manage carry/borrow variables explicitly.

#define add_64(res, a, b)				res = a + b;
#define add_32(res, a, b)				res = a + b;
#define sub_64(res, a, b)				res = a - b;
#define sub_32(res, a, b)				res = a - b;

#define mul_lo_64(res, a, b)			res = a * b;
#define mul_hi_64(res, a, b)			res = mul_hi(a, b); // OpenCL C built-in
#define mad_lo_64(res, a, b, c)			res = a * b + c;
#define mad_hi_64(res, a, b, c)			res = mad_hi(a, b, c); // OpenCL C built-in

#define mul_lo_32(res, a, b)			res = a * b;
#define mul_hi_32(res, a, b)			res = mul_hi(a, b); // OpenCL C built-in
#define mad_lo_32(res, a, b, c)			res = a * b + c;
#define mad_hi_32(res, a, b, c)			res = mad_hi(a, b, c); // OpenCL C built-in

#define mul_wide_32(res, a, b)			res = (u64)a * b;
#define mad_wide_32(res,a,b,c)			res = (u64)a * b + c;

#define st_cs_v4_b32(addr,val)			*(addr) = val; // OpenCL doesn't have a direct equivalent for cache streaming store. Standard store.


//P-related constants
#define P_0			0xFFFFFFFEFFFFFC2Full
#define P_123		0xFFFFFFFFFFFFFFFFull
#define P_INV32		0x000003D1

#define Add192to192(r, v) ocl_add192(r, v)
#define Sub192from192(r, v) ocl_sub192(r, v)

#define Copy_int4_x2(dst, src) {\
  ((int4*)(dst))[0] = ((int4*)(src))[0]; \
  ((int4*)(dst))[1] = ((int4*)(src))[1]; }

#define Copy_u64_x4(dst, src) {\
  ((u64*)(dst))[0] = ((u64*)(src))[0]; \
  ((u64*)(dst))[1] = ((u64*)(src))[1]; \
  ((u64*)(dst))[2] = ((u64*)(src))[2]; \
  ((u64*)(dst))[3] = ((u64*)(src))[3]; }

inline void NegModP(u64* input_res)
{
    u64 P_const[4] = {P_0, P_123, P_123, P_123};
    u64 borrow64 = 0;
    input_res[0] = ocl_sub_borrow_u64(P_const[0], input_res[0], 0, &borrow64);
    input_res[1] = ocl_sub_borrow_u64(P_const[1], input_res[1], borrow64, &borrow64);
    input_res[2] = ocl_sub_borrow_u64(P_const[2], input_res[2], borrow64, &borrow64);
    u64 dummy_borrow;
    input_res[3] = ocl_sub_borrow_u64(P_const[3], input_res[3], borrow64, &dummy_borrow);
}

inline void SubModP(u64* res, u64* val1, u64* val2)
{
    u64 borrow64 = 0;
    res[0] = ocl_sub_borrow_u64(val1[0], val2[0], 0, &borrow64);
    res[1] = ocl_sub_borrow_u64(val1[1], val2[1], borrow64, &borrow64);
    res[2] = ocl_sub_borrow_u64(val1[2], val2[2], borrow64, &borrow64);
    res[3] = ocl_sub_borrow_u64(val1[3], val2[3], borrow64, &borrow64);

    if (borrow64)
    {
        u64 P_const[4] = {P_0, P_123, P_123, P_123};
        u64 carry64_addP = 0;
        res[0] = ocl_add_carry_u64(res[0], P_const[0], 0, &carry64_addP);
        res[1] = ocl_add_carry_u64(res[1], P_const[1], carry64_addP, &carry64_addP);
        res[2] = ocl_add_carry_u64(res[2], P_const[2], carry64_addP, &carry64_addP);
        u64 dummy_carry;
        res[3] = ocl_add_carry_u64(res[3], P_const[3], carry64_addP, &dummy_carry);
    }
}

inline void AddModP(u64* res, u64* val1, u64* val2)
{
	u64 tmp[4];
	u64 op_carry = 0;
	tmp[0] = ocl_add_carry_u64(val1[0], val2[0], 0, &op_carry);
	tmp[1] = ocl_add_carry_u64(val1[1], val2[1], op_carry, &op_carry);
	tmp[2] = ocl_add_carry_u64(val1[2], val2[2], op_carry, &op_carry);
	tmp[3] = ocl_add_carry_u64(val1[3], val2[3], op_carry, &op_carry);
	Copy_u64_x4(res, tmp);

    u64 P_const[4] = {P_0, P_123, P_123, P_123};
    u64 temp_sub_res[4];
    u64 borrow64_final = 0;

    temp_sub_res[0] = ocl_sub_borrow_u64(tmp[0], P_const[0], 0, &borrow64_final);
    temp_sub_res[1] = ocl_sub_borrow_u64(tmp[1], P_const[1], borrow64_final, &borrow64_final);
    temp_sub_res[2] = ocl_sub_borrow_u64(tmp[2], P_const[2], borrow64_final, &borrow64_final);
    temp_sub_res[3] = ocl_sub_borrow_u64(tmp[3], P_const[3], borrow64_final, &borrow64_final);

    if (borrow64_final == 0) {
        Copy_u64_x4(res, temp_sub_res);
    } else {
        Copy_u64_x4(res, tmp);
    }
}

inline void add_320_to_256(u64* res, u64* val)
{
    u64 carry64 = 0;
    res[0] = ocl_add_carry_u64(res[0], val[0], 0, &carry64);
    res[1] = ocl_add_carry_u64(res[1], val[1], carry64, &carry64);
    res[2] = ocl_add_carry_u64(res[2], val[2], carry64, &carry64);
    res[3] = ocl_add_carry_u64(res[3], val[3], carry64, &carry64);
    res[4] = ocl_add_carry_u64(val[4], 0ull, carry64, &carry64);
}

inline void mul_256_by_P0inv(u32* res, u32* val)
{
	u64 tmp64[7];
	u32* tmp = (u32*)tmp64;
	mul_wide_32(*(u64*)res, val[0], P_INV32);
	mul_wide_32(tmp64[0], val[1], P_INV32);
	mul_wide_32(tmp64[1], val[2], P_INV32);
	mul_wide_32(tmp64[2], val[3], P_INV32);
	mul_wide_32(tmp64[3], val[4], P_INV32);
	mul_wide_32(tmp64[4], val[5], P_INV32);
	mul_wide_32(tmp64[5], val[6], P_INV32);
	mul_wide_32(tmp64[6], val[7], P_INV32);

    u32 carry32_chain1 = 0;
    res[1] = ocl_add_carry_u32(res[1], tmp[0], 0, &carry32_chain1);
    res[2] = ocl_add_carry_u32(tmp[1], tmp[2], carry32_chain1, &carry32_chain1);
    res[3] = ocl_add_carry_u32(tmp[3], tmp[4], carry32_chain1, &carry32_chain1);
    res[4] = ocl_add_carry_u32(tmp[5], tmp[6], carry32_chain1, &carry32_chain1);
    res[5] = ocl_add_carry_u32(tmp[7], tmp[8], carry32_chain1, &carry32_chain1);
    res[6] = ocl_add_carry_u32(tmp[9], tmp[10], carry32_chain1, &carry32_chain1);
    res[7] = ocl_add_carry_u32(tmp[11], tmp[12], carry32_chain1, &carry32_chain1);
    u32 dummy_carry1;
    res[8] = ocl_add_carry_u32(tmp[13], 0, carry32_chain1, &dummy_carry1);

    u32 carry32_chain2 = 0;
    res[1] = ocl_add_carry_u32(res[1], val[0], 0, &carry32_chain2);
    res[2] = ocl_add_carry_u32(res[2], val[1], carry32_chain2, &carry32_chain2);
    res[3] = ocl_add_carry_u32(res[3], val[2], carry32_chain2, &carry32_chain2);
    res[4] = ocl_add_carry_u32(res[4], val[3], carry32_chain2, &carry32_chain2);
    res[5] = ocl_add_carry_u32(res[5], val[4], carry32_chain2, &carry32_chain2);
    res[6] = ocl_add_carry_u32(res[6], val[5], carry32_chain2, &carry32_chain2);
    res[7] = ocl_add_carry_u32(res[7], val[6], carry32_chain2, &carry32_chain2);
    res[8] = ocl_add_carry_u32(res[8], val[7], carry32_chain2, &carry32_chain2);
    u32 dummy_carry2;
    res[9] = ocl_add_carry_u32(0, 0, carry32_chain2, &dummy_carry2);
}

inline void mul_256_by_64(u64* res, u64* val256, u64 val64)
{
	u64 tmp64[7];
	u32* tmp = (u32*)tmp64;
	u32* rs = (u32*)res;
	u32* a = (u32*)val256;
	u32* b = (u32*)&val64;

	mul_wide_32(res[0], a[0], b[0]);
	mul_wide_32(tmp64[0], a[1], b[0]);
	mul_wide_32(tmp64[1], a[2], b[0]);
	mul_wide_32(tmp64[2], a[3], b[0]);
	mul_wide_32(tmp64[3], a[4], b[0]);
	mul_wide_32(tmp64[4], a[5], b[0]);
	mul_wide_32(tmp64[5], a[6], b[0]);
	mul_wide_32(tmp64[6], a[7], b[0]);

    u32 carry_chain1 = 0;
    rs[1] = ocl_add_carry_u32(rs[1], tmp[0], 0, &carry_chain1);
    rs[2] = ocl_add_carry_u32(tmp[1], tmp[2], carry_chain1, &carry_chain1);
    rs[3] = ocl_add_carry_u32(tmp[3], tmp[4], carry_chain1, &carry_chain1);
    rs[4] = ocl_add_carry_u32(tmp[5], tmp[6], carry_chain1, &carry_chain1);
    rs[5] = ocl_add_carry_u32(tmp[7], tmp[8], carry_chain1, &carry_chain1);
    rs[6] = ocl_add_carry_u32(tmp[9], tmp[10], carry_chain1, &carry_chain1);
    rs[7] = ocl_add_carry_u32(tmp[11], tmp[12], carry_chain1, &carry_chain1);
    u32 dummy_c1;
    rs[8] = ocl_add_carry_u32(tmp[13], 0, carry_chain1, &dummy_c1);

	u64 kk[7];
	u32* k = (u32*)kk;
	mul_wide_32(kk[0], a[0], b[1]);
	mul_wide_32(tmp64[0], a[1], b[1]);
	mul_wide_32(tmp64[1], a[2], b[1]);
	mul_wide_32(tmp64[2], a[3], b[1]);
	mul_wide_32(tmp64[3], a[4], b[1]);
	mul_wide_32(tmp64[4], a[5], b[1]);
	mul_wide_32(tmp64[5], a[6], b[1]);
	mul_wide_32(tmp64[6], a[7], b[1]);

    u32 carry_chain2 = 0;
    k[1] = ocl_add_carry_u32(k[1], tmp[0], 0, &carry_chain2);
    k[2] = ocl_add_carry_u32(tmp[1], tmp[2], carry_chain2, &carry_chain2);
    k[3] = ocl_add_carry_u32(tmp[3], tmp[4], carry_chain2, &carry_chain2);
    k[4] = ocl_add_carry_u32(tmp[5], tmp[6], carry_chain2, &carry_chain2);
    k[5] = ocl_add_carry_u32(tmp[7], tmp[8], carry_chain2, &carry_chain2);
    k[6] = ocl_add_carry_u32(tmp[9], tmp[10], carry_chain2, &carry_chain2);
    k[7] = ocl_add_carry_u32(tmp[11], tmp[12], carry_chain2, &carry_chain2);
    u32 dummy_c2;
    k[8] = ocl_add_carry_u32(tmp[13], 0, carry_chain2, &dummy_c2);

    u32 carry_chain3 = 0;
    rs[1] = ocl_add_carry_u32(rs[1], k[0], 0, &carry_chain3);
    rs[2] = ocl_add_carry_u32(rs[2], k[1], carry_chain3, &carry_chain3);
    rs[3] = ocl_add_carry_u32(rs[3], k[2], carry_chain3, &carry_chain3);
    rs[4] = ocl_add_carry_u32(rs[4], k[3], carry_chain3, &carry_chain3);
    rs[5] = ocl_add_carry_u32(rs[5], k[4], carry_chain3, &carry_chain3);
    rs[6] = ocl_add_carry_u32(rs[6], k[5], carry_chain3, &carry_chain3);
    rs[7] = ocl_add_carry_u32(rs[7], k[6], carry_chain3, &carry_chain3);
    rs[8] = ocl_add_carry_u32(rs[8], k[7], carry_chain3, &carry_chain3);
    u32 dummy_c3;
    rs[9] = ocl_add_carry_u32(k[8], 0, carry_chain3, &dummy_c3);
}

inline void MulModP(u64 *res, u64 *val1, u64 *val2)
{
	u64 buff[8], tmp[5], tmp2[2], tmp3;
	mul_256_by_64(tmp, val1, val2[1]);
	mul_256_by_64(buff, val1, val2[0]);
	add_320_to_256(buff + 1, tmp);
	mul_256_by_64(tmp, val1, val2[2]);
	add_320_to_256(buff + 2, tmp);
	mul_256_by_64(tmp, val1, val2[3]);
	add_320_to_256(buff + 3, tmp);

	mul_256_by_P0inv((u32*)tmp, (u32*)(buff + 4));

    u64 carry64_b1 = 0;
    buff[0] = ocl_add_carry_u64(buff[0], tmp[0], 0, &carry64_b1);
    buff[1] = ocl_add_carry_u64(buff[1], tmp[1], carry64_b1, &carry64_b1);
    buff[2] = ocl_add_carry_u64(buff[2], tmp[2], carry64_b1, &carry64_b1);
    buff[3] = ocl_add_carry_u64(buff[3], tmp[3], carry64_b1, &carry64_b1);
    u64 dummy_b1_t4;
    tmp[4] = ocl_add_carry_u64(tmp[4], 0ull, carry64_b1, &dummy_b1_t4);

	u32* t32 = (u32*)tmp;
	u32* a32 = (u32*)tmp2;
	u32* k = (u32*)&tmp3;

	mul_wide_32(tmp2[0], t32[8], P_INV32);
	mul_wide_32(tmp3, t32[9], P_INV32);

    u32 carry32_b2_s1 = 0;
    a32[1] = ocl_add_carry_u32(a32[1], k[0], 0, &carry32_b2_s1);
    u32 dummy_b2_a2;
    a32[2] = ocl_add_carry_u32(k[1], (u32)0, carry32_b2_s1, &dummy_b2_a2);

    u32 carry32_b2_s2 = 0;
    a32[1] = ocl_add_carry_u32(a32[1], t32[8], 0, &carry32_b2_s2);
    a32[2] = ocl_add_carry_u32(a32[2], t32[9], carry32_b2_s2, &carry32_b2_s2);
    u32 dummy_b2_a3;
    a32[3] = ocl_add_carry_u32(0, 0, carry32_b2_s2, &dummy_b2_a3);

    u64 carry64_b3 = 0;
    res[0] = ocl_add_carry_u64(buff[0], tmp2[0], 0, &carry64_b3);
    res[1] = ocl_add_carry_u64(buff[1], tmp2[1], carry64_b3, &carry64_b3);
    res[2] = ocl_add_carry_u64(buff[2], 0ull, carry64_b3, &carry64_b3);
    u64 dummy_b3_r3;
    res[3] = ocl_add_carry_u64(buff[3], 0ull, carry64_b3, &dummy_b3_r3);
}

inline void add_320_to_256s(u32* res, u64 _v1, u64 _v2, u64 _v3, u64 _v4, u64 _v5, u64 _v6, u64 _v7, u64 _v8)
{
	u32* v1 = (u32*)&_v1;
	u32* v2 = (u32*)&_v2;
	u32* v3 = (u32*)&_v3;
	u32* v4 = (u32*)&_v4;
	u32* v5 = (u32*)&_v5;
	u32* v6 = (u32*)&_v6;
	u32* v7 = (u32*)&_v7;
	u32* v8 = (u32*)&_v8;

    u32 carry32_s1 = 0;
    res[0] = ocl_add_carry_u32(res[0], v1[0], 0, &carry32_s1);
    res[1] = ocl_add_carry_u32(res[1], v1[1], carry32_s1, &carry32_s1);
    res[2] = ocl_add_carry_u32(res[2], v3[0], carry32_s1, &carry32_s1);
    res[3] = ocl_add_carry_u32(res[3], v3[1], carry32_s1, &carry32_s1);
    res[4] = ocl_add_carry_u32(res[4], v5[0], carry32_s1, &carry32_s1);
    res[5] = ocl_add_carry_u32(res[5], v5[1], carry32_s1, &carry32_s1);
    res[6] = ocl_add_carry_u32(res[6], v7[0], carry32_s1, &carry32_s1);
    res[7] = ocl_add_carry_u32(res[7], v7[1], carry32_s1, &carry32_s1);
    u32 dummy_carry_s_res8;
    res[8] = ocl_add_carry_u32(res[8], 0, carry32_s1, &dummy_carry_s_res8);

    u32 carry32_s2 = 0;
    res[1] = ocl_add_carry_u32(res[1], v2[0], 0, &carry32_s2);
    res[2] = ocl_add_carry_u32(res[2], v2[1], carry32_s2, &carry32_s2);
    res[3] = ocl_add_carry_u32(res[3], v4[0], carry32_s2, &carry32_s2);
    res[4] = ocl_add_carry_u32(res[4], v4[1], carry32_s2, &carry32_s2);
    res[5] = ocl_add_carry_u32(res[5], v6[0], carry32_s2, &carry32_s2);
    res[6] = ocl_add_carry_u32(res[6], v6[1], carry32_s2, &carry32_s2);
    res[7] = ocl_add_carry_u32(res[7], v8[0], carry32_s2, &carry32_s2);
    res[8] = ocl_add_carry_u32(res[8], v8[1], carry32_s2, &carry32_s2);
    u32 dummy_carry_s_res9;
    res[9] = ocl_add_carry_u32(0,0, carry32_s2, &dummy_carry_s_res9);
}

inline void SqrModP(u64* res, u64* val)
{
	u64 buff[8], tmp[5], tmp2[2], tmp3, mm;
	u32* a = (u32*)val;
	u64 mar[28];
	u32* b32 = (u32*)buff;
	u32* m32 = (u32*)mar;
//calc 512 bits
	mul_wide_32(mar[0], a[1], a[0]); //ab
	mul_wide_32(mar[1], a[2], a[0]); //ac
	mul_wide_32(mar[2], a[3], a[0]); //ad
	mul_wide_32(mar[3], a[4], a[0]); //ae
	mul_wide_32(mar[4], a[5], a[0]); //af
	mul_wide_32(mar[5], a[6], a[0]); //ag
	mul_wide_32(mar[6], a[7], a[0]); //ah
	mul_wide_32(mar[7], a[2], a[1]); //bc
	mul_wide_32(mar[8], a[3], a[1]); //bd
	mul_wide_32(mar[9], a[4], a[1]); //be
	mul_wide_32(mar[10], a[5], a[1]); //bf
	mul_wide_32(mar[11], a[6], a[1]); //bg
	mul_wide_32(mar[12], a[7], a[1]); //bh
	mul_wide_32(mar[13], a[3], a[2]); //cd
	mul_wide_32(mar[14], a[4], a[2]); //ce
	mul_wide_32(mar[15], a[5], a[2]); //cf
	mul_wide_32(mar[16], a[6], a[2]); //cg
	mul_wide_32(mar[17], a[7], a[2]); //ch
	mul_wide_32(mar[18], a[4], a[3]); //de
	mul_wide_32(mar[19], a[5], a[3]); //df
	mul_wide_32(mar[20], a[6], a[3]); //dg
	mul_wide_32(mar[21], a[7], a[3]); //dh
	mul_wide_32(mar[22], a[5], a[4]); //ef
	mul_wide_32(mar[23], a[6], a[4]); //eg
	mul_wide_32(mar[24], a[7], a[4]); //eh
	mul_wide_32(mar[25], a[6], a[5]); //fg
	mul_wide_32(mar[26], a[7], a[5]); //fh
	mul_wide_32(mar[27], a[7], a[6]); //gh
//a
	mul_wide_32(buff[0], a[0], a[0]); //aa
    u32 carry_sqa = 0;
    b32[1] = ocl_add_carry_u32(b32[1], m32[0], 0, &carry_sqa);
    b32[2] = ocl_add_carry_u32(m32[1], m32[2], carry_sqa, &carry_sqa);
    b32[3] = ocl_add_carry_u32(m32[3], m32[4], carry_sqa, &carry_sqa);
    b32[4] = ocl_add_carry_u32(m32[5], m32[6], carry_sqa, &carry_sqa);
    b32[5] = ocl_add_carry_u32(m32[7], m32[8], carry_sqa, &carry_sqa);
    b32[6] = ocl_add_carry_u32(m32[9], m32[10], carry_sqa, &carry_sqa);
    b32[7] = ocl_add_carry_u32(m32[11], m32[12], carry_sqa, &carry_sqa);
    b32[8] = ocl_add_carry_u32(m32[13], 0, carry_sqa, &carry_sqa);
	b32[9] = carry_sqa;
//b+
	mul_wide_32(mm, a[1], a[1]); //bb
	add_320_to_256s(b32 + 1, mar[0], mm, mar[7], mar[8], mar[9], mar[10], mar[11], mar[12]);
	mul_wide_32(mm, a[2], a[2]); //cc
	add_320_to_256s(b32 + 2, mar[1], mar[7], mm, mar[13], mar[14], mar[15], mar[16], mar[17]);
	mul_wide_32(mm, a[3], a[3]); //dd
	add_320_to_256s(b32 + 3, mar[2], mar[8], mar[13], mm, mar[18], mar[19], mar[20], mar[21]);
	mul_wide_32(mm, a[4], a[4]); //ee
	add_320_to_256s(b32 + 4, mar[3], mar[9], mar[14], mar[18], mm, mar[22], mar[23], mar[24]);
	mul_wide_32(mm, a[5], a[5]); //ff
	add_320_to_256s(b32 + 5, mar[4], mar[10], mar[15], mar[19], mar[22], mm, mar[25], mar[26]);
	mul_wide_32(mm, a[6], a[6]); //gg
	add_320_to_256s(b32 + 6, mar[5], mar[11], mar[16], mar[20], mar[23], mar[25], mm, mar[27]);
	mul_wide_32(mm, a[7], a[7]); //hh
	add_320_to_256s(b32 + 7, mar[6], mar[12], mar[17], mar[21], mar[24], mar[26], mar[27], mm);
//fast mod P
	mul_256_by_P0inv((u32*)tmp, (u32*)(buff + 4));
    u64 carry64_b1_sq = 0; // Renamed to avoid conflict with MulModP if they were inlined in same scope by chance
    buff[0] = ocl_add_carry_u64(buff[0], tmp[0], 0, &carry64_b1_sq);
    buff[1] = ocl_add_carry_u64(buff[1], tmp[1], carry64_b1_sq, &carry64_b1_sq);
    buff[2] = ocl_add_carry_u64(buff[2], tmp[2], carry64_b1_sq, &carry64_b1_sq);
    buff[3] = ocl_add_carry_u64(buff[3], tmp[3], carry64_b1_sq, &carry64_b1_sq);
    u64 dummy_b1_t4_sq;
    tmp[4] = ocl_add_carry_u64(tmp[4], 0ull, carry64_b1_sq, &dummy_b1_t4_sq);
//see mul_256_by_P0inv for details
	u32* t32 = (u32*)tmp;
	u32* a32 = (u32*)tmp2;
	u32* k = (u32*)&tmp3;
	mul_wide_32(tmp2[0], t32[8], P_INV32);
	mul_wide_32(tmp3, t32[9], P_INV32);
    u32 carry32_b2_s1_sq = 0;
    a32[1] = ocl_add_carry_u32(a32[1], k[0], 0, &carry32_b2_s1_sq);
    u32 dummy_b2_a2_sq;
    a32[2] = ocl_add_carry_u32(k[1], (u32)0, carry32_b2_s1_sq, &dummy_b2_a2_sq);
    u32 carry32_b2_s2_sq = 0;
    a32[1] = ocl_add_carry_u32(a32[1], t32[8], 0, &carry32_b2_s2_sq);
    a32[2] = ocl_add_carry_u32(a32[2], t32[9], carry32_b2_s2_sq, &carry32_b2_s2_sq);
    u32 dummy_b2_a3_sq;
    a32[3] = ocl_add_carry_u32(0, 0, carry32_b2_s2_sq, &dummy_b2_a3_sq);

    u64 carry64_b3_sq = 0;
    res[0] = ocl_add_carry_u64(buff[0], tmp2[0], 0, &carry64_b3_sq);
    res[1] = ocl_add_carry_u64(buff[1], tmp2[1], carry64_b3_sq, &carry64_b3_sq);
    res[2] = ocl_add_carry_u64(buff[2], 0ull, carry64_b3_sq, &carry64_b3_sq);
    u64 dummy_b3_r3_sq;
    res[3] = ocl_add_carry_u64(buff[3], 0ull, carry64_b3_sq, &dummy_b3_r3_sq);
}

inline void add_288(u32* res, u32* val1, u32* val2)
{
    u32 carry32 = 0;
    res[0] = ocl_add_carry_u32(val1[0], val2[0], 0, &carry32);
    res[1] = ocl_add_carry_u32(val1[1], val2[1], carry32, &carry32);
    res[2] = ocl_add_carry_u32(val1[2], val2[2], carry32, &carry32);
    res[3] = ocl_add_carry_u32(val1[3], val2[3], carry32, &carry32);
    res[4] = ocl_add_carry_u32(val1[4], val2[4], carry32, &carry32);
    res[5] = ocl_add_carry_u32(val1[5], val2[5], carry32, &carry32);
    res[6] = ocl_add_carry_u32(val1[6], val2[6], carry32, &carry32);
    res[7] = ocl_add_carry_u32(val1[7], val2[7], carry32, &carry32);
    u32 dummy_carry;
    res[8] = ocl_add_carry_u32(val1[8], val2[8], carry32, &dummy_carry);
}

inline void neg_288(u32* res)
{
    u32 borrow32 = 0;
    res[0] = ocl_sub_borrow_u32(0, res[0], 0, &borrow32);
    res[1] = ocl_sub_borrow_u32(0, res[1], borrow32, &borrow32);
    res[2] = ocl_sub_borrow_u32(0, res[2], borrow32, &borrow32);
    res[3] = ocl_sub_borrow_u32(0, res[3], borrow32, &borrow32);
    res[4] = ocl_sub_borrow_u32(0, res[4], borrow32, &borrow32);
    res[5] = ocl_sub_borrow_u32(0, res[5], borrow32, &borrow32);
    res[6] = ocl_sub_borrow_u32(0, res[6], borrow32, &borrow32);
    res[7] = ocl_sub_borrow_u32(0, res[7], borrow32, &borrow32);
    u32 dummy_borrow;
    res[8] = ocl_sub_borrow_u32(0, res[8], borrow32, &dummy_borrow);
}

inline void mul_288_by_i32(u32* res, u32* val288, int ival32)
{
	u32 val32 = abs(ival32);
	u64 tmp64_mul[4];
	u32* tmp = (u32*)tmp64_mul;
	u64* r64 = (u64*)res;
	r64[0] = (u64)val288[0] * val32;
	r64[1] = (u64)val288[2] * val32;
	r64[2] = (u64)val288[4] * val32;
	r64[3] = (u64)val288[6] * val32;
	tmp64_mul[0] = (u64)val288[1] * val32;
	tmp64_mul[1] = (u64)val288[3] * val32;
	tmp64_mul[2] = (u64)val288[5] * val32;
	tmp64_mul[3] = (u64)val288[7] * val32;

    u32 carry32 = 0;
    res[1] = ocl_add_carry_u32(res[1], tmp[0], 0, &carry32);
    res[2] = ocl_add_carry_u32(res[2], tmp[1], carry32, &carry32);
    res[3] = ocl_add_carry_u32(res[3], tmp[2], carry32, &carry32);
    res[4] = ocl_add_carry_u32(res[4], tmp[3], carry32, &carry32);
    res[5] = ocl_add_carry_u32(res[5], tmp[4], carry32, &carry32);
    res[6] = ocl_add_carry_u32(res[6], tmp[5], carry32, &carry32);
    res[7] = ocl_add_carry_u32(res[7], tmp[6], carry32, &carry32);

    u64 temp_mad = (u64)val288[8] * val32 + tmp[7] + carry32;
    res[8] = (u32)temp_mad;

	if (ival32 < 0)
		neg_288(res);
}

inline void set_288_i32(u32* res, int val)
{
	res[0] = val;
	res[1] = (val < 0) ? 0xFFFFFFFF : 0;
	res[2] = res[1];
	res[3] = res[1];
	res[4] = res[1];
	res[5] = res[1];
	res[6] = res[1];
	res[7] = res[1];
	res[8] = res[1];
}

inline void mul_P_by_32(u32* res, u32 val)
{
	/*__align__(8)*/ u32 tmp[3];
	mul_wide_32(*(u64*)tmp, val, P_INV32);

    u32 carry32_add = 0;
    tmp[1] = ocl_add_carry_u32(tmp[1], val, 0, &carry32_add);
    u32 dummy_carry_add;
    tmp[2] = ocl_add_carry_u32(0, 0, carry32_add, &dummy_carry_add);

    u32 borrow32_sub = 0;
    res[0] = ocl_sub_borrow_u32(0, tmp[0], 0, &borrow32_sub);
    res[1] = ocl_sub_borrow_u32(0, tmp[1], borrow32_sub, &borrow32_sub);
    res[2] = ocl_sub_borrow_u32(0, tmp[2], borrow32_sub, &borrow32_sub);
    res[3] = ocl_sub_borrow_u32(0, 0, borrow32_sub, &borrow32_sub);
    res[4] = ocl_sub_borrow_u32(0, 0, borrow32_sub, &borrow32_sub);
    res[5] = ocl_sub_borrow_u32(0, 0, borrow32_sub, &borrow32_sub);
    res[6] = ocl_sub_borrow_u32(0, 0, borrow32_sub, &borrow32_sub);
    res[7] = ocl_sub_borrow_u32(0, 0, borrow32_sub, &borrow32_sub);
    u32 dummy_borrow_sub;
    res[8] = ocl_sub_borrow_u32(val, 0, borrow32_sub, &dummy_borrow_sub);
}

inline void shiftR_288_by_30(u32* res)
{
    for (int i = 0; i < 8; ++i) {
        res[i] = (res[i] >> 30) | (res[i+1] << 2);
    }
    res[8] = ((i32)res[8]) >> 30;
}

inline void add_288_P(u32* res)
{
    u32 P_val[9] = {0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0};
    u32 carry32 = 0;
    res[0] = ocl_add_carry_u32(res[0], P_val[0], 0, &carry32);
    res[1] = ocl_add_carry_u32(res[1], P_val[1], carry32, &carry32);
    res[2] = ocl_add_carry_u32(res[2], P_val[2], carry32, &carry32);
    res[3] = ocl_add_carry_u32(res[3], P_val[3], carry32, &carry32);
    res[4] = ocl_add_carry_u32(res[4], P_val[4], carry32, &carry32);
    res[5] = ocl_add_carry_u32(res[5], P_val[5], carry32, &carry32);
    res[6] = ocl_add_carry_u32(res[6], P_val[6], carry32, &carry32);
    res[7] = ocl_add_carry_u32(res[7], P_val[7], carry32, &carry32);
    u32 dummy_carry;
    res[8] = ocl_add_carry_u32(res[8], P_val[8], carry32, &dummy_carry);
}

inline void sub_288_P(u32* res)
{
    u32 P_val[9] = {0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0};
    u32 borrow32 = 0;
    res[0] = ocl_sub_borrow_u32(res[0], P_val[0], 0, &borrow32);
    res[1] = ocl_sub_borrow_u32(res[1], P_val[1], borrow32, &borrow32);
    res[2] = ocl_sub_borrow_u32(res[2], P_val[2], borrow32, &borrow32);
    res[3] = ocl_sub_borrow_u32(res[3], P_val[3], borrow32, &borrow32);
    res[4] = ocl_sub_borrow_u32(res[4], P_val[4], borrow32, &borrow32);
    res[5] = ocl_sub_borrow_u32(res[5], P_val[5], borrow32, &borrow32);
    res[6] = ocl_sub_borrow_u32(res[6], P_val[6], borrow32, &borrow32);
    res[7] = ocl_sub_borrow_u32(res[7], P_val[7], borrow32, &borrow32);
    u32 dummy_borrow;
    res[8] = ocl_sub_borrow_u32(res[8], P_val[8], borrow32, &dummy_borrow);
}

#define APPLY_DIV_SHIFT()	matrix[0] <<= index; matrix[1] <<= index; kbnt -= index; _val >>= index;
#define DO_INV_STEP()		{kbnt = -kbnt; int tmp_step = -_modp; _modp = _val; _val = tmp_step; tmp_step = -matrix[0]; \
							matrix[0] = matrix[2]; matrix[2] = tmp_step; tmp_step = -matrix[1]; matrix[1] = matrix[3]; matrix[3] = tmp_step;} // Renamed tmp to tmp_step

inline void InvModP(u32* res)
{
	int matrix[4], _val, _modp, index, cnt, mx, kbnt;
	/*__align__(8)*/ u32 modp[9];
	/*__align__(8)*/ u32 val[9];
	/*__align__(8)*/ u32 a[9];
	/*__align__(8)*/ u32 inv_tmp[4][9]; // Renamed tmp to inv_tmp to avoid conflict with DO_INV_STEP macro

	((u64*)modp)[0] = P_0;
	((u64*)modp)[1] = P_123;
	((u64*)modp)[2] = P_123;
	((u64*)modp)[3] = P_123;
	modp[8] = 0;
	res[8] = 0; // Assuming res is at least 288 bits (u32[9])
	val[0] = res[0]; val[1] = res[1]; val[2] = res[2]; val[3] = res[3];
	val[4] = res[4]; val[5] = res[5]; val[6] = res[6]; val[7] = res[7];
	val[8] = 0;
	matrix[0] = matrix[3] = 1;
	matrix[1] = matrix[2] = 0;
	kbnt = -1;
	_val = (int)res[0];
	_modp = (int)P_0;
    index = ctz(_val | 0x40000000);
	APPLY_DIV_SHIFT();
	cnt = 30 - index;
	while (cnt > 0)
	{
		if (kbnt < 0)
			DO_INV_STEP();
		mx = (kbnt + 1 < cnt) ? 31 - kbnt : 32 - cnt;
		i32 mul = (-_modp * _val) & 7;
		mul &= 0xFFFFFFFF >> mx;
		_val += _modp * mul;
		matrix[2] += matrix[0] * mul;
		matrix[3] += matrix[1] * mul;
        index = ctz(_val | (1 << cnt));
		APPLY_DIV_SHIFT();
		cnt -= index;
	}
	mul_288_by_i32(inv_tmp[0], modp, matrix[0]);
	mul_288_by_i32(inv_tmp[1], val, matrix[1]);
	mul_288_by_i32(inv_tmp[2], modp, matrix[2]);
	mul_288_by_i32(inv_tmp[3], val, matrix[3]);
	add_288(modp, inv_tmp[0], inv_tmp[1]);
	shiftR_288_by_30(modp);
	add_288(val, inv_tmp[2], inv_tmp[3]);
	shiftR_288_by_30(val);
	set_288_i32(inv_tmp[1], matrix[1]);
	set_288_i32(inv_tmp[3], matrix[3]);
	mul_P_by_32(res, (inv_tmp[1][0] * 0xD2253531) & 0x3FFFFFFF);
	add_288(res, res, inv_tmp[1]);
	shiftR_288_by_30(res);
	mul_P_by_32(a, (inv_tmp[3][0] * 0xD2253531) & 0x3FFFFFFF);
	add_288(a, a, inv_tmp[3]);
	shiftR_288_by_30(a);
	while (1)
	{
		matrix[0] = matrix[3] = 1;
		matrix[1] = matrix[2] = 0;
		_val = val[0];
		_modp = modp[0];
        index = ctz(_val | 0x40000000);
		APPLY_DIV_SHIFT();
		cnt = 30 - index;
		while (cnt > 0)
		{
			if (kbnt < 0)
				DO_INV_STEP();
			mx = (kbnt + 1 < cnt) ? 31 - kbnt : 32 - cnt;
			i32 mul = (-_modp * _val) & 7;
			mul &= 0xFFFFFFFF >> mx;
			_val += _modp * mul;
			matrix[2] += matrix[0] * mul;
			matrix[3] += matrix[1] * mul;
            index = ctz(_val | (1 << cnt));
			APPLY_DIV_SHIFT();
			cnt -= index;
		}
		mul_288_by_i32(inv_tmp[0], modp, matrix[0]);
		mul_288_by_i32(inv_tmp[1], val, matrix[1]);
		mul_288_by_i32(inv_tmp[2], modp, matrix[2]);
		mul_288_by_i32(inv_tmp[3], val, matrix[3]);
		add_288(modp, inv_tmp[0], inv_tmp[1]);
		shiftR_288_by_30(modp);
		add_288(val, inv_tmp[2], inv_tmp[3]);
		shiftR_288_by_30(val);
		mul_288_by_i32(inv_tmp[0], res, matrix[0]);
		mul_288_by_i32(inv_tmp[1], a, matrix[1]);

		if ((val[0] | val[1] | val[2] | val[3] | val[4] | val[5] | val[6] | val[7]) == 0)
			break;

		mul_288_by_i32(inv_tmp[2], res, matrix[2]);
		mul_288_by_i32(inv_tmp[3], a, matrix[3]);
		mul_P_by_32(res, ((inv_tmp[0][0] + inv_tmp[1][0]) * 0xD2253531) & 0x3FFFFFFF);
		add_288(res, res, inv_tmp[0]);
		add_288(res, res, inv_tmp[1]);
		shiftR_288_by_30(res);
		mul_P_by_32(a, ((inv_tmp[2][0] + inv_tmp[3][0]) * 0xD2253531) & 0x3FFFFFFF);
		add_288(a, a, inv_tmp[2]);
		add_288(a, a, inv_tmp[3]);
		shiftR_288_by_30(a);
	}
	mul_P_by_32(res, ((inv_tmp[0][0] + inv_tmp[1][0]) * 0xD2253531) & 0x3FFFFFFF);
	add_288(res, res, inv_tmp[0]);
	add_288(res, res, inv_tmp[1]);
	shiftR_288_by_30(res);
	if ((int)modp[8] < 0)
		neg_288(res);
	while ((int)res[8] < 0)
		add_288_P(res);
	while ((int)res[8] > 0)
		sub_288_P(res);
}
