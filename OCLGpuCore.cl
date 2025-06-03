// OpenCL Port of RCGpuCore.cu
// (c) 2024, RetiredCoder (RC) & AI Assistant
// License: GPLv3, see "LICENSE.TXT" file

// Basic OpenCL type compatibility (assuming defs.h is not directly included or needs override)
// These should match the types used in OCLGpuUtils.h and TKparams
typedef ulong u64;
typedef long ilong; // Assuming i64 maps to long
typedef uint u32;
typedef int i32;
typedef ushort u16;
typedef short i16;
typedef uchar u8;
typedef char i8;

#include "OCLGpuUtils.h"

// Definitions from defs.h that might be needed by kernels
#define STEP_CNT			1000
#define JMP_CNT				512
// PNT_GROUP_CNT, BLOCK_SIZE, OLD_GPU are defined per kernel version or build
#define JMP_MASK			(JMP_CNT-1)
#define DPTABLE_MAX_CNT		16
#define MAX_DP_CNT			(256 * 1024)
#define GPU_DP_SIZE			48

#define DP_FLAG				0x8000
#define INV_FLAG			0x4000
#define JMP2_FLAG			0x2000
#define MD_LEN				10

typedef struct {
	__global u64* Kangs;
	u32 KangCnt;
	u32 BlockCnt;
	u32 BlockSize;
	u32 GroupCnt;    // PNT_GROUP_CNT for the current kernel launch
	__global u64* L2;
	u64 DP;
	__global u32* DPs_out; // First element is count, then data.
	__global u64* Jumps1;
	__global u64* Jumps2;
	__global u64* Jumps3;
	__global u64* JumpsList;
	__global u32* DPTable;   // First KangCnt elements are counters, then data.
	__global u64* L1S2;      // Using ulong to accommodate OLD_GPU; non-OLD will cast to uint* internally.
	__global u64* LastPnts;
	__global u64* LoopTable;
	__global u32* dbg_buf;
	__global u32* LoopedKangs; // First element is total count, LoopedKangs[1] is current index counter. Data from LoopedKangs[2].
	bool IsGenMode;
} TKparams_ocl;

#define get_num_groups_x()       get_num_groups(0)
#define get_group_id_x()         get_group_id(0)
#define get_local_id_x()         get_local_id(0)
#define get_local_size_x()       get_local_size(0)
#define barrier_local()          barrier(CLK_LOCAL_MEM_FENCE)
#define atomic_add_global_uint(ptr, val) atom_add(ptr, val)
#define atomic_and_global_uint(ptr, val) atom_and(ptr, val)
#define atomic_and_global_ulong(ptr, val) atom_and(ptr, val)


#define KERNEL_A_LDS_JMP1_SIZE_U64 (8 * JMP_CNT)
#define KERNEL_A_LDS_JLIST_ELEMENTS (2048)

#define KERNEL_B_LDS_JMP_D_SIZE_U64 (3 * JMP_CNT)
#define KERNEL_C_LDS_JMP_TABLE_SIZE_U64 (12 * JMP_CNT)


#define ALIGN16 __attribute__((aligned(16)))

#define Copy_4_ulongs_local_to_private(dst_private_ulong_arr, src_local_ulong_ptr) \
    for(int _i=0; _i<4; ++_i) (dst_private_ulong_arr)[_i] = (src_local_ulong_ptr)[_i];

#define Copy_4_ulongs_const_to_private(dst_private_ulong_arr, src_const_ulong_ptr) \
    for(int _i=0; _i<4; ++_i) (dst_private_ulong_arr)[_i] = (src_const_ulong_ptr)[_i];

#define Copy_4_ulongs_global_to_private(dst, src_global) for(int _k=0;_k<4;++_k) (dst)[_k]=(src_global)[_k];
#define Copy_4_ulongs_private_to_global(dst_global, src) for(int _k=0;_k<4;++_k) (dst_global)[_k]=(src)[_k];
#define Copy_4_ulongs_private_to_local(dst_local, src)   for(int _k=0;_k<4;++_k) (dst_local)[_k]=(src)[_k];

// Default to non-OLD_GPU settings if OLD_GPU is not defined by compiler
#ifndef PNT_GROUP_CNT_KERNEL
    #ifdef OLD_GPU
        #define PNT_GROUP_CNT_KERNEL 64
        #define BLOCK_SIZE_KERNEL 512
    #else
        #define PNT_GROUP_CNT_KERNEL 24
        #define BLOCK_SIZE_KERNEL 256
    #endif
#endif

// Content from previous overwrite (KernelA versions)
#ifndef OLD_GPU

__kernel void KernelA_main(
    __global const TKparams_ocl* Kparams_ptr,
    __local ulong* local_lds,
    __constant ulong* jmp2_table_arg)
{
    const ulong dp_mask64 = ~((1UL << (64 - Kparams_ptr->DP)) - 1);
    const uint kang_base_idx = PNT_GROUP_CNT_KERNEL * (get_local_id_x() + get_group_id_x() * get_local_size_x());

    ulong l2_area_stride = 4UL * PNT_GROUP_CNT_KERNEL * get_num_groups_x() * get_local_size_x();
    __global ulong* L2_base = Kparams_ptr->L2;
    __global ulong* L2x_thread_base = L2_base + (2 * get_local_id_x() + 4 * get_local_size_x() * get_group_id_x());
    __global ulong* L2y_thread_base = L2x_thread_base + l2_area_stride;
    __global ulong* L2s_thread_base = L2y_thread_base + l2_area_stride;

    ulong l2_group_stride_offset = get_local_size_x() * 4 * get_num_groups_x();


	__global ushort* jlist_ushort_base = (__global ushort*)(Kparams_ptr->JumpsList);
    uint jlist_base_offset = (get_group_id_x() * STEP_CNT * PNT_GROUP_CNT_KERNEL * get_local_size_x() / (sizeof(ulong)/sizeof(ushort)));
    __global ushort* jlist_current_step_base = jlist_ushort_base + jlist_base_offset;
    uint jlist_warp_output_offset = (get_local_id_x() / 32) * (32 * PNT_GROUP_CNT_KERNEL / 8);
    __global ushort* jlist_ushort_for_write = jlist_current_step_base + jlist_warp_output_offset;


    ulong lastpnts_area_stride = 4UL * PNT_GROUP_CNT_KERNEL * get_num_groups_x() * get_local_size_x();
    __global ulong* x_last0_base = Kparams_ptr->LastPnts + (2 * get_local_id_x() + 4 * get_local_size_x() * get_group_id_x());
    __global ulong* y_last0_base = x_last0_base + lastpnts_area_stride;

	__local ulong* jmp1_table = local_lds;
	__local ushort* lds_jlist = (__local ushort*)&local_lds[KERNEL_A_LDS_JMP1_SIZE_U64];

	uint i = get_local_id_x();
	while (i < JMP_CNT) {
        for(int k=0; k<8; ++k) {
            jmp1_table[8 * i + k] = Kparams_ptr->Jumps1[12 * i + k];
        }
		i += get_local_size_x();
    }
    barrier_local();

	ALIGN16 ulong x[4], y[4], tmp[4], tmp2[4];
	ushort jmp_ind_ushort;

	for (uint group = 0; group < PNT_GROUP_CNT_KERNEL; group++) {
        uint current_kang_absolute_idx = kang_base_idx + group;
        __global ulong* L2x_g = L2x_thread_base + l2_group_stride_offset * group;
        __global ulong* L2y_g = L2y_thread_base + l2_group_stride_offset * group;

		tmp[0] = Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 0]; tmp[1] = Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 1];
		tmp[2] = Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 2]; tmp[3] = Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 3];
        L2x_g[0] = tmp[0]; L2x_g[1] = tmp[1]; L2x_g[2] = tmp[2]; L2x_g[3] = tmp[3];

		tmp[0] = Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 4]; tmp[1] = Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 5];
		tmp[2] = Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 6]; tmp[3] = Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 7];
        L2y_g[0] = tmp[0]; L2y_g[1] = tmp[1]; L2y_g[2] = tmp[2]; L2y_g[3] = tmp[3];
	}

	uint L1S2_val = ((__global uint*)Kparams_ptr->L1S2)[get_group_id_x() * get_local_size_x() + get_local_id_x()];

    for (int step_ind = 0; step_ind < STEP_CNT; step_ind++) {
        ALIGN16 ulong inverse[4];
		__local ulong* jmp_table_local_ref;
        __constant ulong* jmp_table_const_ref;
		ALIGN16 ulong jmp_x[4];
		ALIGN16 ulong jmp_y[4];

        __global ulong* L2x_g0 = L2x_thread_base + l2_group_stride_offset * 0;
        __global ulong* L2s_g0 = L2s_thread_base + l2_group_stride_offset * 0;
        Copy_4_ulongs_global_to_private(x, L2x_g0);

		jmp_ind_ushort = (ushort)(x[0] % JMP_CNT);
        bool use_jmp2_0 = (L1S2_val >> 0) & 1;
        if(use_jmp2_0) jmp_table_const_ref = jmp2_table_arg; else jmp_table_local_ref = jmp1_table;
        if(use_jmp2_0) Copy_4_ulongs_const_to_private(jmp_x, jmp_table_const_ref + 8 * jmp_ind_ushort);
		else Copy_4_ulongs_local_to_private(jmp_x, jmp_table_local_ref + 8 * jmp_ind_ushort);
		SubModP(inverse, x, jmp_x);
        Copy_4_ulongs_private_to_global(L2s_g0, inverse);

		for (int group = 1; group < PNT_GROUP_CNT_KERNEL; group++) {
            __global ulong* L2x_g = L2x_thread_base + l2_group_stride_offset * group;
            __global ulong* L2s_g = L2s_thread_base + l2_group_stride_offset * group;
            Copy_4_ulongs_global_to_private(x, L2x_g);

            jmp_ind_ushort = (ushort)(x[0] % JMP_CNT);
            bool use_jmp2_g = (L1S2_val >> group) & 1;
            if(use_jmp2_g) jmp_table_const_ref = jmp2_table_arg; else jmp_table_local_ref = jmp1_table;
            if(use_jmp2_g) Copy_4_ulongs_const_to_private(jmp_x, jmp_table_const_ref + 8 * jmp_ind_ushort);
			else Copy_4_ulongs_local_to_private(jmp_x, jmp_table_local_ref + 8 * jmp_ind_ushort);
			SubModP(tmp, x, jmp_x);
			MulModP(inverse, inverse, tmp);
            Copy_4_ulongs_private_to_global(L2s_g, inverse);
		}

		InvModP((u32*)inverse);

        for (int group = PNT_GROUP_CNT_KERNEL - 1; group >= 0; group--) {
            ALIGN16 ulong x0[4], y0[4], dxs[4];
            __global ulong* L2x_g = L2x_thread_base + l2_group_stride_offset * group;
            __global ulong* L2y_g = L2y_thread_base + l2_group_stride_offset * group;
            __global ulong* L2s_gm1 = L2s_thread_base + l2_group_stride_offset * (group -1);

            Copy_4_ulongs_global_to_private(x0, L2x_g);
            Copy_4_ulongs_global_to_private(y0, L2y_g);

            jmp_ind_ushort = (ushort)(x0[0] % JMP_CNT);
            bool use_jmp2_grp = (L1S2_val >> group) & 1;
            if(use_jmp2_grp) jmp_table_const_ref = jmp2_table_arg; else jmp_table_local_ref = jmp1_table;

            if(use_jmp2_grp) {
                Copy_4_ulongs_const_to_private(jmp_x, jmp_table_const_ref + 8 * jmp_ind_ushort);
                Copy_4_ulongs_const_to_private(jmp_y, jmp_table_const_ref + 8 * jmp_ind_ushort + 4);
            } else {
                Copy_4_ulongs_local_to_private(jmp_x, jmp_table_local_ref + 8 * jmp_ind_ushort);
			    Copy_4_ulongs_local_to_private(jmp_y, jmp_table_local_ref + 8 * jmp_ind_ushort + 4);
            }

            uint inv_flag_bit = (uint)y0[0] & 1;
			if (inv_flag_bit) { jmp_ind_ushort |= INV_FLAG; NegModP(jmp_y); }

            if (group) {
                Copy_4_ulongs_global_to_private(tmp, L2s_gm1);
				SubModP(tmp2, x0, jmp_x);
				MulModP(dxs, tmp, inverse);
				MulModP(inverse, inverse, tmp2);
            } else { Copy_u64_x4(dxs, inverse); }
			SubModP(tmp2, y0, jmp_y); MulModP(tmp, tmp2, dxs); SqrModP(tmp2, tmp);
			SubModP(x, tmp2, jmp_x); SubModP(x, x, x0);
            Copy_4_ulongs_private_to_global(L2x_g, x);
			SubModP(y, x0, x); MulModP(y, y, tmp); SubModP(y, y, y0);
            Copy_4_ulongs_private_to_global(L2y_g, y);

			if (((L1S2_val >> group) & 1) == 0) {
				uint jmp_next = (uint)(x[0] % JMP_CNT);
				jmp_next |= ((uint)y[0] & 1) ? 0 : INV_FLAG;
				L1S2_val |= (jmp_ind_ushort == jmp_next) ? (1u << group) : 0;
			} else {
				L1S2_val &= ~(1u << group); jmp_ind_ushort |= JMP2_FLAG;
			}

			if ((x[3] & dp_mask64) == 0) {
				uint current_kang_abs_idx = kang_base_idx + group;
				uint ind_dp = atomic_add_global_uint((__global uint*)&Kparams_ptr->DPTable[current_kang_abs_idx], 1U);
				ind_dp = min(ind_dp, (uint)DPTABLE_MAX_CNT - 1);
				__global ulong* dst_dp_data = (__global ulong*)((__global uint*)Kparams_ptr->DPTable + Kparams_ptr->KangCnt + (current_kang_abs_idx * DPTABLE_MAX_CNT + ind_dp) * (GPU_DP_SIZE / sizeof(uint)));
                dst_dp_data[0] = x[0]; dst_dp_data[1] = x[1]; dst_dp_data[2] = x[2]; dst_dp_data[3] = x[3];
				jmp_ind_ushort |= DP_FLAG;
			}

			lds_jlist[8 * get_local_id_x() + (group % 8)] = jmp_ind_ushort;
			if ((group % 8) == 0) {
                uint jlist_offset = (group / 8) * 32 + (get_local_id_x() % 32);
                __global ushort* jlist_target = jlist_current_step_base + jlist_warp_output_offset + jlist_offset;
                for(int k_jl=0; k_jl < 8; ++k_jl) {
                    jlist_target[k_jl] = lds_jlist[8 * get_local_id_x() + k_jl];
                }
            }
			if (step_ind + MD_LEN >= STEP_CNT) {
				int n = step_ind + MD_LEN - STEP_CNT;
				__global ulong* x_last = x_last0_base + n * (2 * lastpnts_area_stride) + l2_group_stride_offset * group;
				__global ulong* y_last = y_last0_base + n * (2 * lastpnts_area_stride) + l2_group_stride_offset * group;
                Copy_4_ulongs_private_to_global(x_last, x);
                Copy_4_ulongs_private_to_global(y_last, y);
			}
        }
		jlist_current_step_base += PNT_GROUP_CNT_KERNEL * get_local_size_x() / (sizeof(ulong)/sizeof(ushort));
    }
	((__global uint*)Kparams_ptr->L1S2)[get_group_id_x() * get_local_size_x() + get_local_id_x()] = L1S2_val;

	for (uint group = 0; group < PNT_GROUP_CNT_KERNEL; group++) {
        uint current_kang_absolute_idx = kang_base_idx + group;
        __global ulong* L2x_g = L2x_thread_base + l2_group_stride_offset * group;
        __global ulong* L2y_g = L2y_thread_base + l2_group_stride_offset * group;
        Copy_4_ulongs_global_to_private(tmp, L2x_g);
		Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 0] = tmp[0]; Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 1] = tmp[1];
		Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 2] = tmp[2]; Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 3] = tmp[3];
		Copy_4_ulongs_global_to_private(tmp, L2y_g);
		Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 4] = tmp[0]; Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 5] = tmp[1];
		Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 6] = tmp[2]; Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 7] = tmp[3];
	}
}
#endif // OLD_GPU (KernelA_main)

#ifdef OLD_GPU
__kernel void KernelA_oldgpu(
    __global const TKparams_ocl* Kparams_ptr,
    __local ulong* local_lds,
    __constant ulong* jmp2_table_arg)
{
	__local ulong* jmp1_table = local_lds;
	__local ushort* lds_jlist = (__local ushort*)&local_lds[KERNEL_A_LDS_JMP1_SIZE_U64];
    uint lx_base_offset_ulongs = KERNEL_A_LDS_JMP1_SIZE_U64 + (KERNEL_A_LDS_JLIST_ELEMENTS * sizeof(ushort) + sizeof(ulong) -1) / sizeof(ulong);
    __local ulong* Lx_lds = &local_lds[lx_base_offset_ulongs];
    __local ulong* Ly_lds = Lx_lds + (4 * PNT_GROUP_CNT_KERNEL);
    __local ulong* Ls_lds = Ly_lds + (4 * PNT_GROUP_CNT_KERNEL);

	uint i = get_local_id_x();
	while (i < JMP_CNT) {
        for(int k=0; k<8; ++k) {
            jmp1_table[8 * i + k] = Kparams_ptr->Jumps1[12 * i + k];
        }
		i += get_local_size_x();
	}
    barrier_local();

	ALIGN16 ulong inverse[4];
	ALIGN16 ulong x[4], y[4], tmp[4], tmp2[4];
	const ulong dp_mask64 = ~((1UL << (64 - Kparams_ptr->DP)) - 1);
	ushort jmp_ind_ushort;

    __global ushort* jlist_ushort_base = (__global ushort*)(Kparams_ptr->JumpsList);
    uint jlist_base_offset = (get_group_id_x() * STEP_CNT * PNT_GROUP_CNT_KERNEL * get_local_size_x() / (sizeof(ulong)/sizeof(ushort)));
    __global ushort* jlist_current_step_base = jlist_ushort_base + jlist_base_offset;
    uint jlist_warp_output_offset = (get_local_id_x() / 32) * (32 * PNT_GROUP_CNT_KERNEL / 8);
    __global ushort* jlist_ushort_for_write = jlist_current_step_base + jlist_warp_output_offset;

    ulong lastpnts_area_stride = 4UL * PNT_GROUP_CNT_KERNEL * get_num_groups_x() * get_local_size_x();
    __global ulong* x_last0_base = Kparams_ptr->LastPnts + (2 * get_local_id_x() + 4 * get_local_size_x() * get_group_id_x());
    __global ulong* y_last0_base = x_last0_base + lastpnts_area_stride;

	uint kang_base_idx = PNT_GROUP_CNT_KERNEL * (get_local_id_x() + get_group_id_x() * get_local_size_x());
	for (uint group = 0; group < PNT_GROUP_CNT_KERNEL; group++) {
        uint current_kang_absolute_idx = kang_base_idx + group;
		tmp[0] = Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 0]; tmp[1] = Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 1];
		tmp[2] = Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 2]; tmp[3] = Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 3];
		Copy_4_ulongs_private_to_local(Lx_lds + group*4, tmp);
		tmp[0] = Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 4]; tmp[1] = Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 5];
		tmp[2] = Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 6]; tmp[3] = Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 7];
		Copy_4_ulongs_private_to_local(Ly_lds + group*4, tmp);
	}

	ulong L1S2_val_ul = Kparams_ptr->L1S2[get_group_id_x() * get_local_size_x() + get_local_id_x()];

    __local ulong* jmp_table_local_ref;
    __constant ulong* jmp_table_const_ref;
	ALIGN16 ulong jmp_x[4];
	ALIGN16 ulong jmp_y[4];

	for (int group = 0; group < PNT_GROUP_CNT_KERNEL; group++) {
		Copy_4_ulongs_local_to_private(x, Lx_lds + group*4);
		jmp_ind_ushort = (ushort)(x[0] % JMP_CNT);
        bool use_jmp2 = (L1S2_val_ul >> group) & 1;
        if(use_jmp2) jmp_table_const_ref = jmp2_table_arg; else jmp_table_local_ref = jmp1_table;

        if(use_jmp2) Copy_4_ulongs_const_to_private(jmp_x, jmp_table_const_ref + 8 * jmp_ind_ushort);
        else Copy_4_ulongs_local_to_private(jmp_x, jmp_table_local_ref + 8 * jmp_ind_ushort);

		SubModP(tmp, x, jmp_x);
		if (group == 0) {
			Copy_u64_x4(inverse, tmp);
			Copy_4_ulongs_private_to_local(Ls_lds, tmp);
		} else {
			MulModP(inverse, inverse, tmp);
			if ((group & 1) == 0)
				Copy_4_ulongs_private_to_local(Ls_lds + (group/2)*4, inverse);
		}
	}

	int g_beg = PNT_GROUP_CNT_KERNEL - 1;
	int g_end = -1;
	int g_inc = -1;
	int s_mask = 1;
	int jlast_add = 0;
	ALIGN16 ulong t_cache[4], x0_cache[4], jmpx_cached[4];
	t_cache[0]=0; t_cache[1]=0; t_cache[2]=0; t_cache[3]=0;
	x0_cache[0]=0; x0_cache[1]=0; x0_cache[2]=0; x0_cache[3]=0;
    jmpx_cached[0]=0; jmpx_cached[1]=0; jmpx_cached[2]=0; jmpx_cached[3]=0;

	for (int step_ind = 0; step_ind < STEP_CNT; step_ind++) {
		ALIGN16 ulong next_inv[4];
		InvModP((u32*)inverse);

		int group = g_beg;
		bool cached = false;
		while (group != g_end) {
			ALIGN16 ulong dx[4], x0[4], y0[4], dx0[4];
			if (cached) { Copy_u64_x4(x0, x0_cache); }
			else { Copy_4_ulongs_local_to_private(x0, Lx_lds + group*4); }
			Copy_4_ulongs_local_to_private(y0, Ly_lds + group*4);

			jmp_ind_ushort = (ushort)(x0[0] % JMP_CNT);
            bool use_jmp2_g = (L1S2_val_ul >> group) & 1;
            if(use_jmp2_g) jmp_table_const_ref = jmp2_table_arg; else jmp_table_local_ref = jmp1_table;

            if (cached) { Copy_u64_x4(jmp_x, jmpx_cached); }
			else {
                if(use_jmp2_g) Copy_4_ulongs_const_to_private(jmp_x, jmp_table_const_ref + 8 * jmp_ind_ushort);
				else Copy_4_ulongs_local_to_private(jmp_x, jmp_table_local_ref + 8 * jmp_ind_ushort);
			}
            if(use_jmp2_g) Copy_4_ulongs_const_to_private(jmp_y, jmp_table_const_ref + 8 * jmp_ind_ushort + 4);
            else Copy_4_ulongs_local_to_private(jmp_y, jmp_table_local_ref + 8 * jmp_ind_ushort + 4);

            uint inv_flag_bit = (uint)y0[0] & 1;
			if (inv_flag_bit) { jmp_ind_ushort |= INV_FLAG; NegModP(jmp_y); }

			if (group == g_end - g_inc) { Copy_u64_x4(dx0, inverse); }
			else {
				if ((group & 1) == s_mask) {
					if (cached) { Copy_u64_x4(tmp, t_cache); cached = false; }
					else { Copy_4_ulongs_local_to_private(tmp, Ls_lds + ((group + g_inc) / 2)*4 ); }
				} else {
					Copy_4_ulongs_local_to_private(t_cache, Ls_lds + ((group + g_inc + g_inc) / 2)*4 );
					cached = true;
					Copy_4_ulongs_local_to_private(x0_cache, Lx_lds + (group + g_inc)*4);

                    ushort jmp_tmp_ushort = (ushort)(x0_cache[0] % JMP_CNT);
					ALIGN16 ulong dx2[4];
                    bool use_jmp2_tmp = (L1S2_val_ul >> (group + g_inc)) & 1;
                    __constant ulong* jmp_table_c_tmp; __local ulong* jmp_table_l_tmp;
                    if(use_jmp2_tmp) jmp_table_c_tmp = jmp2_table_arg; else jmp_table_l_tmp = jmp1_table;

					if(use_jmp2_tmp) Copy_4_ulongs_const_to_private(jmpx_cached, jmp_table_c_tmp + 8 * jmp_tmp_ushort);
                    else Copy_4_ulongs_local_to_private(jmpx_cached, jmp_table_l_tmp + 8 * jmp_tmp_ushort);
					SubModP(dx2, x0_cache, jmpx_cached);
					MulModP(tmp, t_cache, dx2);
				}
				SubModP(dx, x0, jmp_x); MulModP(dx0, tmp, inverse); MulModP(inverse, inverse, dx);
			}
			SubModP(tmp2, y0, jmp_y); MulModP(tmp, tmp2, dx0); SqrModP(tmp2, tmp);
			SubModP(x, tmp2, jmp_x); SubModP(x, x, x0);
			Copy_4_ulongs_private_to_local(Lx_lds + group*4, x);
			SubModP(y, x0, x); MulModP(y, y, tmp);	SubModP(y, y, y0);
			Copy_4_ulongs_private_to_local(Ly_lds + group*4, y);

			if (((L1S2_val_ul >> group) & 1) == 0) {
				uint jmp_next = (uint)(x[0] % JMP_CNT);
				jmp_next |= ((uint)y[0] & 1) ? 0 : INV_FLAG;
				L1S2_val_ul |= (jmp_ind_ushort == jmp_next) ? (1UL << group) : 0;
			} else {
				L1S2_val_ul &= ~(1UL << group); jmp_ind_ushort |= JMP2_FLAG;
			}

			if ((x[3] & dp_mask64) == 0) {
				uint current_kang_abs_idx = kang_base_idx + group;
				uint ind_dp = atomic_add_global_uint((__global uint*)&Kparams_ptr->DPTable[current_kang_abs_idx], 1U);
				ind_dp = min(ind_dp, (uint)DPTABLE_MAX_CNT - 1);
                __global ulong* dst_dp_data = (__global ulong*)((__global uint*)Kparams_ptr->DPTable + Kparams_ptr->KangCnt + (current_kang_abs_idx * DPTABLE_MAX_CNT + ind_dp) * (GPU_DP_SIZE / sizeof(uint)));
                dst_dp_data[0] = x[0]; dst_dp_data[1] = x[1]; dst_dp_data[2] = x[2]; dst_dp_data[3] = x[3];
				jmp_ind_ushort |= DP_FLAG;
			}
			lds_jlist[8 * get_local_id_x() + (group % 8)] = jmp_ind_ushort;
            if (((group + jlast_add) % 8) == 0) {
                uint jlist_offset = (group / 8) * 32 + (get_local_id_x() % 32);
                __global ushort* jlist_target = jlist_current_step_base + jlist_warp_output_offset + jlist_offset;
                 for(int k_jl=0; k_jl < 8; ++k_jl) {
                    jlist_target[k_jl] = lds_jlist[8 * get_local_id_x() + k_jl];
                }
            }
			if (step_ind + MD_LEN >= STEP_CNT) {
				int n = step_ind + MD_LEN - STEP_CNT;
				__global ulong* x_last = x_last0_base + n * (2 * lastpnts_area_stride) + (get_local_size_x() * 4 * get_num_groups_x()) * group;
				__global ulong* y_last = y_last0_base + n * (2 * lastpnts_area_stride) + (get_local_size_x() * 4 * get_num_groups_x()) * group;
                Copy_4_ulongs_private_to_global(x_last, x);
                Copy_4_ulongs_private_to_global(y_last, y);
			}
			jmp_ind_ushort = (ushort)(x[0] % JMP_CNT);
            bool use_jmp2_next = (L1S2_val_ul >> group) & 1;
            if(use_jmp2_next) jmp_table_const_ref = jmp2_table_arg; else jmp_table_local_ref = jmp1_table;
            if(use_jmp2_next) Copy_4_ulongs_const_to_private(jmp_x, jmp_table_const_ref + 8 * jmp_ind_ushort);
			else Copy_4_ulongs_local_to_private(jmp_x, jmp_table_local_ref + 8 * jmp_ind_ushort);
			SubModP(dx, x, jmp_x);
			if (group == g_beg) {
				Copy_u64_x4(next_inv, dx);
				Copy_4_ulongs_private_to_local(Ls_lds + (g_beg/2)*4, dx);
			} else {
				MulModP(next_inv, next_inv, dx);
				if ((group & 1) == s_mask) { Copy_4_ulongs_private_to_local(Ls_lds + (group/2)*4, next_inv); }
			}
			group += g_inc;
		}
		jlist_current_step_base += PNT_GROUP_CNT_KERNEL * get_local_size_x() / (sizeof(ulong)/sizeof(ushort));
		Copy_u64_x4(inverse, next_inv);
		if (g_inc < 0) { g_beg = 0; g_end = PNT_GROUP_CNT_KERNEL; g_inc = 1; s_mask = 0; jlast_add = 1; }
		else { g_beg = PNT_GROUP_CNT_KERNEL - 1; g_end = -1; g_inc = -1; s_mask = 1; jlast_add = 0; }
	}
	Kparams_ptr->L1S2[get_group_id_x() * get_local_size_x() + get_local_id_x()] = L1S2_val_ul;
	kang_base_idx = PNT_GROUP_CNT_KERNEL * (get_local_id_x() + get_group_id_x() * get_local_size_x());
	for (uint group = 0; group < PNT_GROUP_CNT_KERNEL; group++) {
        uint current_kang_absolute_idx = kang_base_idx + group;
		Copy_4_ulongs_local_to_private(tmp, Lx_lds + group*4);
		Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 0] = tmp[0]; Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 1] = tmp[1];
		Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 2] = tmp[2]; Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 3] = tmp[3];
		Copy_4_ulongs_local_to_private(tmp, Ly_lds + group*4);
		Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 4] = tmp[0]; Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 5] = tmp[1];
		Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 6] = tmp[2]; Kparams_ptr->Kangs[current_kang_absolute_idx * 12 + 7] = tmp[3];
	}
}
#endif // OLD_GPU

// ------------- KERNEL B HELPERS AND KERNEL B -------------------

static inline void BuildDP(
    __global const TKparams_ocl* Kparams_ptr,
    int kang_ind,
    ulong* d) // d is private ulong[3]
{
	uint ind = atom_add((__global uint*)&Kparams_ptr->DPTable[kang_ind], 1U);

	if (ind >= DPTABLE_MAX_CNT) return;

    __global uint* dp_data_section_base = Kparams_ptr->DPTable + Kparams_ptr->KangCnt;
    __global ulong* rx_src_ptr = (__global ulong*)(dp_data_section_base + (kang_ind * DPTABLE_MAX_CNT + ind) * (16 / sizeof(uint)));

    ALIGN16 ulong rx_val[2];
    rx_val[0] = rx_src_ptr[0];
    rx_val[1] = rx_src_ptr[1];

	uint pos = atom_add((__global uint*)Kparams_ptr->DPs_out, 1U);
	pos = min(pos, (uint)MAX_DP_CNT - 1);

	__global ulong* DPs_entry = (__global ulong*)((__global uint*)Kparams_ptr->DPs_out + 4 + pos * (GPU_DP_SIZE / sizeof(uint)));

    DPs_entry[0] = rx_val[0];
    DPs_entry[1] = rx_val[1];
    DPs_entry[2] = d[0];
    DPs_entry[3] = d[1];
    DPs_entry[4] = d[2];
    ((__global u32*)DPs_entry)[10] = 3 * kang_ind / Kparams_ptr->KangCnt;
}

static inline bool ProcessJumpDistance(
    u32 step_ind,
    u32 d_cur,
    ulong* d,
    u32 kang_ind,
    __local ulong* jmp1_d_local,
    __constant ulong* jmp2_d_const,
    __global const TKparams_ocl* Kparams_ptr,
    ulong* table,
    u32* cur_ind,
    u8 iter
) {
	__local ulong* jmp_d_local_sel = jmp1_d_local;
    __constant ulong* jmp_d_const_sel = 0;

	if (d_cur & JMP2_FLAG) {
        jmp_d_const_sel = jmp2_d_const;
    }

	ALIGN16 ulong jmp_dist_val[3];
    uint jmp_mask_val = d_cur & JMP_MASK;

    if (jmp_d_const_sel) {
        jmp_dist_val[0] = jmp_d_const_sel[3 * jmp_mask_val + 0];
        jmp_dist_val[1] = jmp_d_const_sel[3 * jmp_mask_val + 1];
        jmp_dist_val[2] = jmp_d_const_sel[3 * jmp_mask_val + 2];
    } else {
        jmp_dist_val[0] = jmp_d_local_sel[3 * jmp_mask_val + 0];
        jmp_dist_val[1] = jmp_d_local_sel[3 * jmp_mask_val + 1];
        jmp_dist_val[2] = jmp_d_local_sel[3 * jmp_mask_val + 2];
    }

	if (d_cur & INV_FLAG) {	ocl_sub192(d, jmp_dist_val); }
    else { ocl_add192(d, jmp_dist_val); }

    int found_ind = -1;
    for(uint check_idx = 0; check_idx < MD_LEN; ++check_idx) {
        if (table[check_idx] == d[0]) {
            found_ind = check_idx;
            break;
        }
    }

	table[iter] = d[0];
	*cur_ind = (iter + 1) % MD_LEN;

	if (found_ind < 0) {
		if (d_cur & DP_FLAG)
			BuildDP(Kparams_ptr, kang_ind, d);
		return false;
	}

	uint LoopSize = (iter + MD_LEN - found_ind) % MD_LEN;
	if (!LoopSize) LoopSize = MD_LEN;
	atom_add((__global uint*)&Kparams_ptr->dbg_buf[LoopSize], 1U);

	uint ind_LastPnts = MD_LEN - 1 - ((STEP_CNT - 1 - step_ind) % LoopSize);
	uint loop_table_idx = atom_add((__global uint*)&Kparams_ptr->LoopedKangs[1], 1U);
    if (loop_table_idx < (Kparams_ptr->KangCnt * 2)) {
	    Kparams_ptr->LoopedKangs[2 + loop_table_idx] = kang_ind | (ind_LastPnts << 28);
    }
	return true;
}

#define DO_ITER_KERNEL_B(iter_val) {\
	u32 cur_dAB = jlist_for_thread[get_local_id_x()]; \
	u16 cur_dA_ushort = (u16)(cur_dAB & 0xFFFFU); \
	u16 cur_dB_ushort = (u16)(cur_dAB >> 16); \
	if (!LoopedA) \
		LoopedA = ProcessJumpDistance(step_ind, cur_dA_ushort, dA, kang_ind, jmp1_d_local, (__constant ulong*)(Kparams_ptr->Jumps2 + 8), Kparams_ptr, RegsA, &cur_indA, iter_val); \
	if (!LoopedB) \
		LoopedB = ProcessJumpDistance(step_ind, cur_dB_ushort, dB, kang_ind + 1, jmp1_d_local, (__constant ulong*)(Kparams_ptr->Jumps2 + 8), Kparams_ptr, RegsB, &cur_indB, iter_val); \
	jlist_for_thread += get_local_size_x(); \
	step_ind++; \
}

__kernel void KernelB_main(
    __global const TKparams_ocl* Kparams_ptr,
    __local ulong* local_lds)
{
	__local ulong* jmp1_d_local = local_lds;
    __constant ulong* jmp2_d_distances = Kparams_ptr->Jumps2 + 8;


	uint i = get_local_id_x();
	while (i < JMP_CNT) {
		jmp1_d_local[3 * i + 0] = Kparams_ptr->Jumps1[12 * i + 8];
		jmp1_d_local[3 * i + 1] = Kparams_ptr->Jumps1[12 * i + 9];
		jmp1_d_local[3 * i + 2] = Kparams_ptr->Jumps1[12 * i + 10];
		i += get_local_size_x();
	}

    __global u32* jlist0_base = (__global u32*)(Kparams_ptr->JumpsList +
                                (ulong)get_group_id_x() * STEP_CNT * PNT_GROUP_CNT_KERNEL * get_local_size_x() / 2);

	barrier_local();

	ALIGN16 ulong RegsA[MD_LEN], RegsB[MD_LEN];

	for (uint gr_ind2 = 0; gr_ind2 < (PNT_GROUP_CNT_KERNEL/2); gr_ind2++) {
		#pragma unroll
		for (int k_loop = 0; k_loop < MD_LEN; k_loop++) {
            ulong loop_table_base_offset = MD_LEN * get_local_size_x() * PNT_GROUP_CNT_KERNEL * get_group_id_x();
            loop_table_base_offset += 2 * MD_LEN * get_local_size_x() * gr_ind2;
            loop_table_base_offset += get_local_id_x();
			RegsA[k_loop] = Kparams_ptr->LoopTable[loop_table_base_offset + k_loop * get_local_size_x()];
			RegsB[k_loop] = Kparams_ptr->LoopTable[loop_table_base_offset + (k_loop + MD_LEN) * get_local_size_x()];
		}
		u32 cur_indA = 0;
		u32 cur_indB = 0;

		__global u32* jlist_for_thread = jlist0_base + gr_ind2 * get_local_size_x();

		uint kang_base_idx_kernel = PNT_GROUP_CNT_KERNEL * (get_local_id_x() + get_group_id_x() * get_local_size_x());
        uint kang_ind = kang_base_idx_kernel + gr_ind2 * 2;

		ALIGN16 ulong dA[3], dB[3];
		dA[0] = Kparams_ptr->Kangs[kang_ind * 12 + 8]; dA[1] = Kparams_ptr->Kangs[kang_ind * 12 + 9]; dA[2] = Kparams_ptr->Kangs[kang_ind * 12 + 10];
		dB[0] = Kparams_ptr->Kangs[(kang_ind + 1) * 12 + 8]; dB[1] = Kparams_ptr->Kangs[(kang_ind + 1) * 12 + 9]; dB[2] = Kparams_ptr->Kangs[(kang_ind + 1) * 12 + 10];

		bool LoopedA = false;
		bool LoopedB = false;
		u32 step_ind = 0;
		while (step_ind < STEP_CNT) {
			DO_ITER_KERNEL_B(0); DO_ITER_KERNEL_B(1); DO_ITER_KERNEL_B(2); DO_ITER_KERNEL_B(3); DO_ITER_KERNEL_B(4);
			DO_ITER_KERNEL_B(5); DO_ITER_KERNEL_B(6); DO_ITER_KERNEL_B(7); DO_ITER_KERNEL_B(8); DO_ITER_KERNEL_B(9);
		}

		Kparams_ptr->Kangs[kang_ind * 12 + 8] = dA[0]; Kparams_ptr->Kangs[kang_ind * 12 + 9] = dA[1]; Kparams_ptr->Kangs[kang_ind * 12 + 10] = dA[2];
		Kparams_ptr->Kangs[(kang_ind + 1) * 12 + 8] = dB[0]; Kparams_ptr->Kangs[(kang_ind + 1) * 12 + 9] = dB[1]; Kparams_ptr->Kangs[(kang_ind + 1) * 12 + 10] = dB[2];

		#pragma unroll
		for (int k_loop = 0; k_loop < MD_LEN; k_loop++) {
            ulong loop_table_base_offset = MD_LEN * get_local_size_x() * PNT_GROUP_CNT_KERNEL * get_group_id_x();
            loop_table_base_offset += 2 * MD_LEN * get_local_size_x() * gr_ind2;
            loop_table_base_offset += get_local_id_x();

            uint final_idx_A = (k_loop + MD_LEN - cur_indA) % MD_LEN;
			Kparams_ptr->LoopTable[loop_table_base_offset + final_idx_A * get_local_size_x()] = RegsA[k_loop];
			uint final_idx_B = (k_loop + MD_LEN - cur_indB) % MD_LEN;
			Kparams_ptr->LoopTable[loop_table_base_offset + (final_idx_B + MD_LEN) * get_local_size_x()] = RegsB[k_loop];
		}
	}
}

// ------------- KERNEL C -------------------
__kernel void KernelC_main(
    __global const TKparams_ocl* Kparams_ptr,
    __local ulong* local_lds) // For jmp3_table
{
	__local ulong* jmp3_table = local_lds;

	uint loc_id = get_local_id_x();
    uint loc_size = get_local_size_x();

	for (uint i = loc_id; i < JMP_CNT; i += loc_size) {
		for(int k=0; k<12; ++k) { // Each Jumps3 entry is 12 ulongs
            jmp3_table[12 * i + k] = Kparams_ptr->Jumps3[12 * i + k];
        }
	}
	barrier_local();

	while (1) {
		uint ind = atom_add(&Kparams_ptr->LoopedKangs[1], 1U);
		if (ind >= Kparams_ptr->LoopedKangs[0]) break;

		u32 kang_data = Kparams_ptr->LoopedKangs[2 + ind];
		u32 kang_ind = kang_data & 0x0FFFFFFF;
		u32 last_ind = kang_data >> 28;

		ALIGN16 ulong x0[4], x[4];
		ALIGN16 ulong y0[4], y[4];
		ALIGN16 ulong jmp_x[4];
		ALIGN16 ulong jmp_y[4];
		ALIGN16 ulong inverse[4];
		ulong tmp_arith[4], tmp2_arith[4];

        uint orig_block_idx = kang_ind / (BLOCK_SIZE_KERNEL * PNT_GROUP_CNT_KERNEL);
        uint temp_idx = kang_ind % (BLOCK_SIZE_KERNEL * PNT_GROUP_CNT_KERNEL);
        uint orig_thread_idx = temp_idx / PNT_GROUP_CNT_KERNEL;
        uint orig_group_idx = temp_idx % PNT_GROUP_CNT_KERNEL;

        ulong lastpnts_total_area_stride_per_step = 2UL * (4UL * PNT_GROUP_CNT_KERNEL * get_num_groups_x() * get_local_size_x());
        ulong x_base_offset_in_lastpnts = (2 * orig_thread_idx + 4 * get_local_size_x() * orig_block_idx);
        ulong group_offset_in_lastpnts = (get_local_size_x() * 4 * get_num_groups_x()) * orig_group_idx;


        __global ulong* x_last_ptr = Kparams_ptr->LastPnts +
                                   last_ind * lastpnts_total_area_stride_per_step +
                                   x_base_offset_in_lastpnts +
                                   group_offset_in_lastpnts;
        __global ulong* y_last_ptr = x_last_ptr + (lastpnts_total_area_stride_per_step / 2);


        Copy_4_ulongs_global_to_private(x0, x_last_ptr);
		Copy_4_ulongs_global_to_private(y0, y_last_ptr);

		uint jmp_idx_masked = (uint)(x0[0] % JMP_CNT);
		Copy_4_ulongs_local_to_private(jmp_x, jmp3_table + 12 * jmp_idx_masked);
		Copy_4_ulongs_local_to_private(jmp_y, jmp3_table + 12 * jmp_idx_masked + 4);

        SubModP(inverse, x0, jmp_x);
		InvModP((u32*)inverse);

		uint inv_flag_kerc = (uint)y0[0] & 1;
		if (inv_flag_kerc) NegModP(jmp_y);

		SubModP(tmp_arith, y0, jmp_y);
		MulModP(tmp2_arith, tmp_arith, inverse);
		SqrModP(tmp_arith, tmp2_arith);

		SubModP(x, tmp_arith, jmp_x); SubModP(x, x, x0);
		SubModP(y, x0, x); MulModP(y, y, tmp2_arith); SubModP(y, y, y0);

        __global ulong* kang_dst_ptr = Kparams_ptr->Kangs + kang_ind * 12;
		kang_dst_ptr[0] = x[0]; kang_dst_ptr[1] = x[1]; kang_dst_ptr[2] = x[2]; kang_dst_ptr[3] = x[3];
		kang_dst_ptr[4] = y[0]; kang_dst_ptr[5] = y[1]; kang_dst_ptr[6] = y[2]; kang_dst_ptr[7] = y[3];

		ALIGN16 ulong d_kc[3];
		d_kc[0] = kang_dst_ptr[8]; d_kc[1] = kang_dst_ptr[9]; d_kc[2] = kang_dst_ptr[10];

        ALIGN16 ulong jmp3_dist[3];
        jmp3_dist[0] = jmp3_table[12 * jmp_idx_masked + 8];
        jmp3_dist[1] = jmp3_table[12 * jmp_idx_masked + 9];
        jmp3_dist[2] = jmp3_table[12 * jmp_idx_masked + 10];

		if (inv_flag_kerc) ocl_sub192(d_kc, jmp3_dist);
		else ocl_add192(d_kc, jmp3_dist);
		kang_dst_ptr[8] = d_kc[0]; kang_dst_ptr[9] = d_kc[1]; kang_dst_ptr[10] = d_kc[2];

        uint l1s2_flat_idx = orig_block_idx * BLOCK_SIZE_KERNEL + orig_thread_idx;
        #ifdef OLD_GPU
            atomic_and_global_ulong(&Kparams_ptr->L1S2[l1s2_flat_idx], ~(1UL << orig_group_idx));
        #else
            atomic_and_global_uint(&((__global uint*)Kparams_ptr->L1S2)[l1s2_flat_idx], ~(1U << orig_group_idx));
        #endif
	}
}

// ------------- Helper EC Functions & KernelGen -------------------

static const ulong GX_0 = 0x59F2815B16F81798UL;
static const ulong GX_1 = 0x029BFCDB2DCE28D9UL;
static const ulong GX_2 = 0x55A06295CE870B07UL;
static const ulong GX_3 = 0x79BE667EF9DCBBACUL;
static const ulong GY_0 = 0x9C47D08FFB10D4B8UL;
static const ulong GY_1 = 0xFD17B448A6855419UL;
static const ulong GY_2 = 0x5DA4FBFC0E1108A8UL;
static const ulong GY_3 = 0x483ADA7726A3C465UL;


static inline void AddPoints_ocl(
    ulong* res_x, ulong* res_y,
    ulong* pnt1x, ulong* pnt1y,
    ulong* pnt2x, ulong* pnt2y)
{
	ALIGN16 ulong tmp[4], tmp2[4], lambda[4], lambda2[4];
	ALIGN16 ulong inverse[4];
	SubModP(inverse, pnt2x, pnt1x);
	InvModP((u32*)inverse);
	SubModP(tmp, pnt2y, pnt1y);
	MulModP(lambda, tmp, inverse);
	MulModP(lambda2, lambda, lambda);
	SubModP(tmp, lambda2, pnt1x);
	SubModP(res_x, tmp, pnt2x);
    // Reverted to original logic based on RCKangaroo.cpp / RCGpuCore.cu
	SubModP(tmp, pnt2x, res_x);
	MulModP(tmp2, tmp, lambda);
	SubModP(res_y, tmp2, pnt2y);
}

static inline void DoublePoint_ocl(
    ulong* res_x, ulong* res_y,
    ulong* pntx, ulong* pnty)
{
	ALIGN16 ulong tmp[4], tmp2[4], lambda[4], lambda2[4];
	ALIGN16 ulong inverse[4];
	AddModP(inverse, pnty, pnty);
	InvModP((u32*)inverse);
	MulModP(tmp2, pntx, pntx);
	AddModP(tmp, tmp2, tmp2);
	AddModP(tmp, tmp, tmp2);
	MulModP(lambda, tmp, inverse);
	MulModP(lambda2, lambda, lambda);
	SubModP(tmp, lambda2, pntx);
	SubModP(res_x, tmp, pntx);
	SubModP(tmp, pntx, res_x);
	MulModP(tmp2, tmp, lambda);
	SubModP(res_y, tmp2, pnty);
}

__kernel void KernelGen_main(__global const TKparams_ocl* Kparams_ptr)
{
	uint kang_base_idx = PNT_GROUP_CNT_KERNEL * (get_local_id_x() + get_group_id_x() * get_local_size_x());

	for (uint group = 0; group < PNT_GROUP_CNT_KERNEL; group++)
	{
		ALIGN16 ulong x0[4], y0[4], d[3];
		ALIGN16 ulong x[4], y[4];
		ALIGN16 ulong tx[4], ty[4];
		ALIGN16 ulong t2x[4], t2y[4];

		uint current_kang_absolute_idx = kang_base_idx + group;
        __global ulong* kang_data = Kparams_ptr->Kangs + current_kang_absolute_idx * 12;

		x0[0] = kang_data[0]; x0[1] = kang_data[1]; x0[2] = kang_data[2]; x0[3] = kang_data[3];
		y0[0] = kang_data[4]; y0[1] = kang_data[5]; y0[2] = kang_data[6]; y0[3] = kang_data[7];
		d[0]  = kang_data[8]; d[1]  = kang_data[9]; d[2]  = kang_data[10];

		tx[0] = GX_0; tx[1] = GX_1; tx[2] = GX_2; tx[3] = GX_3;
		ty[0] = GY_0; ty[1] = GY_1; ty[2] = GY_2; ty[3] = GY_3;

		bool first = true;
		int n = 2;
		while ((n >= 0) && !d[n])
			n--;
		if (n < 0) continue;

        int top_bit_in_d_n = (63 - clz(d[n]));

		for (int i = 0; i <= (n * 64 + top_bit_in_d_n); i++)
		{
			uchar v = (d[i / 64] >> (i % 64)) & 1;
			if (v) {
				if (first) {
					first = false;
					Copy_u64_x4(x, tx);
					Copy_u64_x4(y, ty);
				} else {
					AddPoints_ocl(t2x, t2y, x, y, tx, ty);
					Copy_u64_x4(x, t2x);
					Copy_u64_x4(y, t2y);
				}
			}
			DoublePoint_ocl(t2x, t2y, tx, ty);
			Copy_u64_x4(tx, t2x);
			Copy_u64_x4(ty, t2y);
		}

		if (!Kparams_ptr->IsGenMode) {
			if (current_kang_absolute_idx >= Kparams_ptr->KangCnt / 3) {
				AddPoints_ocl(t2x, t2y, x, y, x0, y0);
				Copy_u64_x4(x, t2x);
				Copy_u64_x4(y, t2y);
			}
        }

		kang_data[0] = x[0]; kang_data[1] = x[1]; kang_data[2] = x[2]; kang_data[3] = x[3];
		kang_data[4] = y[0]; kang_data[5] = y[1]; kang_data[6] = y[2]; kang_data[7] = y[3];
	}
}
