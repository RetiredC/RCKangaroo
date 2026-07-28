// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include "defs.h"
#include "RCGpuUtils.h"

//imp2 table points for KernelA
__device__ __constant__ u64 jmp2_table[8 * JMP_CNT];


#define BLOCK_CNT	gridDim.x
#define BLOCK_X		blockIdx.x
#define THREAD_X	threadIdx.x

//coalescing
#define LOAD_VAL_256(dst, ptr, group) { *((int4*)&(dst)[0]) = *((int4*)&(ptr)[BLOCK_SIZE * 4 * Kparams.BlockCnt * (group)]); *((int4*)&(dst)[2]) = *((int4*)&(ptr)[BLOCK_SIZE * 4 * Kparams.BlockCnt * (group) + 2]); }
#define SAVE_VAL_256(ptr, src, group) { *((int4*)&(ptr)[BLOCK_SIZE * 4 * Kparams.BlockCnt * (group)]) = *((int4*)&(src)[0]); *((int4*)&(ptr)[BLOCK_SIZE * 4 * Kparams.BlockCnt * (group) + 2]) = *((int4*)&(src)[2]); }


extern __shared__ u64 LDS[]; 

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//this kernel performs main jumps
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelA(const TKparams Kparams)
{
	u64 PartStride = PNT_GROUP_CNT * (Kparams.BlockCnt * 256 * 32); //in bytes
	u64* L2x = Kparams.L2 + 4 * (THREAD_X + BLOCK_SIZE * BLOCK_X);
	u64* L2y = L2x + PartStride / 8;
	u64* L2s = L2y + PartStride / 8;
	//list of distances of performed jumps for KernelB
	u16* jlist = (u16*)(Kparams.JumpsList + (u64)BLOCK_X * (STEP_CNT + MD_LEN) * PNT_GROUP_CNT * BLOCK_SIZE / 4); //one rec is 2b so u64(8)/4=2
	jlist += THREAD_X;
	//list of last visited points for KernelC
	u64* x_last0 = Kparams.LastPnts + 4 * (THREAD_X + BLOCK_SIZE * BLOCK_X);
	u64* y_last0 = x_last0 + PartStride / 8;
      
	u64* jmp1_table = LDS; //32KB


	int i = THREAD_X;
	while (i < JMP_CNT)
    {	
		*(int4*)&jmp1_table[8 * i + 0] = *(int4*)&Kparams.Jumps1[12 * i + 0];
		*(int4*)&jmp1_table[8 * i + 2] = *(int4*)&Kparams.Jumps1[12 * i + 2];
		*(int4*)&jmp1_table[8 * i + 4] = *(int4*)&Kparams.Jumps1[12 * i + 4];
		*(int4*)&jmp1_table[8 * i + 6] = *(int4*)&Kparams.Jumps1[12 * i + 6];
		i += BLOCK_SIZE;
    }

    __syncthreads(); 

	__align__(16) u64 x[4], y[4], tmp[4], tmp2[4];
	u32 jmp_ind;

	u32 L1S2 = Kparams.L1S2[BLOCK_X * BLOCK_SIZE + THREAD_X];

    for (int step_ind = 0; step_ind < STEP_CNT; step_ind++)
    {
        __align__(16) u64 inverse[5];
		u64* jmp_table;
		__align__(16) u64 jmp_x[4];
		__align__(16) u64 jmp_y[4];
		
		//first group
		LOAD_VAL_256(x, L2x, 0);
		jmp_ind = x[0] % JMP_CNT;
		jmp_table = ((L1S2 >> 0) & 1) ? jmp2_table : jmp1_table;
		Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
		SubModP(inverse, x, jmp_x);
		SAVE_VAL_256(L2s, inverse, 0);
		//the rest
		for (int group = 1; group < PNT_GROUP_CNT; group++)
		{
			LOAD_VAL_256(x, L2x, group);
			jmp_ind = x[0] % JMP_CNT;
			jmp_table = ((L1S2 >> group) & 1) ? jmp2_table : jmp1_table;
			Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
			SubModP(tmp, x, jmp_x);
			MulModP(inverse, inverse, tmp);
			SAVE_VAL_256(L2s, inverse, group);
		}

		InvModP((u32*)inverse);
        for (int group = PNT_GROUP_CNT - 1; group >= 0; group--)
        {
            __align__(16) u64 x0[4];
            __align__(16) u64 y0[4];
            __align__(16) u64 dxs[4];

			LOAD_VAL_256(x0, L2x, group);
            LOAD_VAL_256(y0, L2y, group);
			jmp_ind = x0[0] % JMP_CNT;
			jmp_table = ((L1S2 >> group) & 1) ? jmp2_table : jmp1_table;
			Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
			Copy_int4_x2(jmp_y, jmp_table + 8 * jmp_ind + 4);
			u32 inv_flag = (u32)y0[0] & 1;
			if (inv_flag)
			{
				jmp_ind |= INV_FLAG;
				NegModP(jmp_y);
			}
            if (group)
            {
				LOAD_VAL_256(tmp, L2s, group - 1);
				SubModP(tmp2, x0, jmp_x);
				MulModP(dxs, tmp, inverse);
				MulModP(inverse, inverse, tmp2);
            }
			else
				Copy_u64_x4(dxs, inverse);

			SubModP(tmp2, y0, jmp_y);
			MulModP(tmp, tmp2, dxs);
			SqrModP(tmp2, tmp);

			SubModP(x, tmp2, jmp_x);
			SubModP(x, x, x0); 
			SAVE_VAL_256(L2x, x, group); 

			SubModP(y, x0, x);
			MulModP(y, y, tmp);
			SubModP(y, y, y0);
			SAVE_VAL_256(L2y, y, group);

			if (((L1S2 >> group) & 1) == 0) //normal mode, check L1S2 loop
			{
				u32 jmp_next = x[0] % JMP_CNT;
				jmp_next |= ((u32)y[0] & 1) ? 0 : INV_FLAG; //inverted
				L1S2 |= (jmp_ind == jmp_next) ? (1u << group) : 0; //loop L1S2 detected
			}
			else
			{
				L1S2 &= ~(1u << group);
				jmp_ind |= JMP2_FLAG;
			}
	
			if ( (((u32*)x)[7] & Kparams.dp_mask) == 0)
			{
				u32 kang_ind = (THREAD_X + BLOCK_X * BLOCK_SIZE) + group * (Kparams.BlockCnt * BLOCK_SIZE);
				u32 ind = atomicAdd(Kparams.DPTable + kang_ind, 1);
				ind = min(ind, DPTABLE_MAX_CNT - 1);
				int4* dst = (int4*)(Kparams.DPTable + Kparams.KangCnt + (kang_ind * DPTABLE_MAX_CNT + ind) * 4);
				dst[0] = ((int4*)x)[0];
				jmp_ind |= DP_FLAG;
			}

			st_cs_b16(&jlist[group * 256], jmp_ind);

			if (step_ind + MD_LEN >= STEP_CNT) //store last kangs to be able to find loop exit point
			{
				int n = step_ind + MD_LEN - STEP_CNT;
				u64* x_last = x_last0 + n * 2 * PartStride / 8;
				u64* y_last = y_last0 + n * 2 * PartStride / 8;
				SAVE_VAL_256(x_last, x, group);
				SAVE_VAL_256(y_last, y, group);
			}
			
        }
		jlist += PNT_GROUP_CNT * BLOCK_SIZE;
    } 

	Kparams.L1S2[BLOCK_X * BLOCK_SIZE + THREAD_X] = L1S2;
} 

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void BuildDP(const TKparams& Kparams, int kang_ind, u64* d)
{
	int ind = atomicAdd(Kparams.DPTable + kang_ind, 0x10000);
	ind >>= 16;
	if (ind >= DPTABLE_MAX_CNT)
		return;
	int4 rx = *(int4*)(Kparams.DPTable + Kparams.KangCnt + (kang_ind * DPTABLE_MAX_CNT + ind) * 4);
	u32 pos = atomicAdd(Kparams.DPs_out, 1);
	pos = min(pos, MAX_DP_CNT - 1);
	u32* DPs = Kparams.DPs_out + 4 + pos * GPU_DP_SIZE / 4;
	*(int4*)&DPs[0] = rx;
	*(int4*)&DPs[4] = ((int4*)d)[0];
	*(u64*)&DPs[8] = d[2];
	DPs[10] = kang_ind;
}

//we compare 64bit, so chance to detect false loop is about 1/2^60 which is quite ok
__device__ __forceinline__ bool ProcessJumpDistance(u32 step_ind, u32 d_cur, u64* d, u32 kang_ind, u64* jmp_d, const TKparams& Kparams, u64* table, u32* cur_ind, u8 iter)
{
	__align__(16) u64 jmp[3];
	((int4*)(jmp))[0] = ((int4*)(jmp_d + 2 * (d_cur & JMP_MASK_ADV)))[0];
	jmp[2] = *(jmp_d + (d_cur & JMP_MASK_ADV) + 8 * JMP_CNT);

	Add192to192(d, jmp);

	//check in table
	int found_ind = iter + MD_LEN - 4;

	while (1)
	{
		if (table[found_ind % MD_LEN] == d[0])
			break;
		found_ind -= 2;
		if (table[found_ind % MD_LEN] == d[0])
			break;
		found_ind -= 2;
		if (table[found_ind % MD_LEN] == d[0])
			break;
		found_ind = iter;
		if (table[found_ind] == d[0])
			break;
		found_ind = -1;
		break;
	}

	table[iter] = d[0];
	*cur_ind = (iter + 1) % MD_LEN;

	if (d_cur & DP_FLAG)
		BuildDP(Kparams, kang_ind, d);

	if (found_ind < 0)	
		return false;

	u32 LoopSize = (iter + MD_LEN - found_ind) % MD_LEN;
	if (!LoopSize)
		LoopSize = MD_LEN;
	atomicAdd(Kparams.dbg_buf + LoopSize, 1); //dbg

	//calc index in LastPnts
	u32 ind_LastPnts = MD_LEN - 1 - ((STEP_CNT - 1 - step_ind) % LoopSize);
	u32 ind = atomicAdd(Kparams.LoopedKangs, 1);
	Kparams.LoopedKangs[2 + ind] = kang_ind | (ind_LastPnts << 28);
	return true;
}

#define DO_ITER(iter) {\
	u32 cur_dAB = jlist[THREAD_X]; \
	u16 cur_dA = cur_dAB & 0xFFFF; \
	u16 cur_dB = cur_dAB >> 16; \
	if (!LoopedA) \
		LoopedA = ProcessJumpDistance(step_ind, cur_dA, dA, kang_ind, jmp_d, Kparams, RegsA, &cur_indA, iter); \
	if (!LoopedB) \
		LoopedB = ProcessJumpDistance(step_ind, cur_dB, dB, kang_ind2, jmp_d, Kparams, RegsB, &cur_indB, iter); \
	jlist += BLOCK_SIZE * PNT_GROUP_CNT / 2; \
	step_ind++; \
}

//this kernel counts distances and detects loops Size>2
//Loops Level1 statistics for JMP_CNT=512: L1S2 = 1/1024 (so one loop every 1024 jumps), L1S4 = L1S2/1024, L1S6 = L1S4/256, L1S8 = L1S6/158, L1S10 = L1S8/82. L1S12 = L1S10/50. 
// For RTX4090 at 8HG/s for 24 hours and JMP_CNT=512: jumps = 691200bln, L1S2 = 682bln, L1S4 = 666mln, L1S6 = 2.6mln, L1S8 = 16.5k, L1S10 = 201. L1S12 = 4.
// I don't see any reasons to catch L1S12 because we have 786432 kangs, if we lose 4 kangs every day, we lose 1460 kangs a year which is about 0.19%.
// This degradation depends only on speed of a single kangaroo, so it's about the same for all 40xx/50xx GPUs.
// Since we lose kangs gradually, for a year we lose 0.19/2 = 0.1% of speed, so you should catch L1S12 only if you are going to solve same point for decades.
// Or you can check all kangs for L1S12 on CPU once a day and restart looped kangs.
// Level2 loops are very rare (about every 2^30 jumps) and they have even size too so they will be handled by the same code. We don't know what loop level we catch so we use JmpTable3 for escaping.
// Minimal loop that we cannot handle is L1S4 + Jump3 + L1S4 = 2^50 (we catch it but cannot escape because it includes JmpTable3), 
// so if we try to solve 140bits (2^70 jumps) with single card (2^20 kangs, every kang will do 2^50 jumps) we will have a problem. But if we use 100cards it will be ok.
// note that kang speed is about 20K/sec so even 2^40 for every kang will take 1.5 years (0.1% chance to loop), 2^50 would take 1500 years.
// Also, if you are in doubts, it's a good idea to check all kangs on CPU once a day and restart looped kangs (and check how many kangs are looped because large number is bad).
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelB(const TKparams Kparams)
{
	u64* jmp_d = LDS; //48KB, 192bit(24bytes) dists (pos and neg, table1 and table2)

	int i = THREAD_X;
	while (i < JMP_CNT)
	{
		//copy 48KB
		jmp_d[12 * i + 0] = Kparams.JmpDists12[12 * i + 0];
		jmp_d[12 * i + 1] = Kparams.JmpDists12[12 * i + 1];
		jmp_d[12 * i + 2] = Kparams.JmpDists12[12 * i + 2];
		jmp_d[12 * i + 3] = Kparams.JmpDists12[12 * i + 3];
		jmp_d[12 * i + 4] = Kparams.JmpDists12[12 * i + 4];
		jmp_d[12 * i + 5] = Kparams.JmpDists12[12 * i + 5];
		jmp_d[12 * i + 6] = Kparams.JmpDists12[12 * i + 6];
		jmp_d[12 * i + 7] = Kparams.JmpDists12[12 * i + 7];
		jmp_d[12 * i + 8] = Kparams.JmpDists12[12 * i + 8];
		jmp_d[12 * i + 9] = Kparams.JmpDists12[12 * i + 9];
		jmp_d[12 * i + 10] = Kparams.JmpDists12[12 * i + 10];
		jmp_d[12 * i + 11] = Kparams.JmpDists12[12 * i + 11];
		i += BLOCK_SIZE;
	}

	u32* jlist0 = (u32*)(Kparams.JumpsList + (u64)BLOCK_X * (STEP_CNT + MD_LEN) * PNT_GROUP_CNT * BLOCK_SIZE / 4);

	__syncthreads();

	u64 RegsA[MD_LEN], RegsB[MD_LEN];

	//we process two kangs at once
	for (u32 gr_ind2 = 0; gr_ind2 < PNT_GROUP_CNT / 2; gr_ind2++)
	{	
		#pragma unroll
		for (int i = 0; i < MD_LEN; i++)
		{
			RegsA[i] = Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X + 2 * MD_LEN * BLOCK_SIZE * gr_ind2 + i * BLOCK_SIZE + THREAD_X];
			RegsB[i] = Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X + 2 * MD_LEN * BLOCK_SIZE * gr_ind2 + (i + MD_LEN) * BLOCK_SIZE + THREAD_X];
		}
		u32 cur_indA = 0;
		u32 cur_indB = 0;

		u32* jlist = jlist0 + gr_ind2 * BLOCK_SIZE;

//calc original kang_ind
		u32 tind = (THREAD_X + gr_ind2 * BLOCK_SIZE); //0..3071
		u32 thr_ind = (2 * tind) % 256; // 0..255	 1tind will handle 2src_thr

		u32 gr_ind = tind / 128; // 0..23    1tind will handle 2src_thr

		u32 kang_ind = (BLOCK_X * BLOCK_SIZE + thr_ind) + gr_ind * (Kparams.BlockCnt * BLOCK_SIZE);
		u32 kang_ind2 = kang_ind + 1;

		__align__(8) u64 dA[3], dB[3];

		dA[0] = Kparams.dists[kang_ind * 4 + 0];
		dA[1] = Kparams.dists[kang_ind * 4 + 1];
		dA[2] = Kparams.dists[kang_ind * 4 + 2];
		dB[0] = Kparams.dists[kang_ind2 * 4 + 0];
		dB[1] = Kparams.dists[kang_ind2 * 4 + 1];
		dB[2] = Kparams.dists[kang_ind2 * 4 + 2];

		bool LoopedA = false;
		bool LoopedB = false;
		u32 step_ind = 0;
		while (step_ind < STEP_CNT)
		{
			DO_ITER(0);
			DO_ITER(1);
			DO_ITER(2);
			DO_ITER(3);
			DO_ITER(4);
			DO_ITER(5);
			DO_ITER(6);
			DO_ITER(7);
			DO_ITER(8);
			DO_ITER(9);
		}

		Kparams.dists[kang_ind * 4 + 0] = dA[0];
		Kparams.dists[kang_ind * 4 + 1] = dA[1];
		Kparams.dists[kang_ind * 4 + 2] = dA[2];
		Kparams.dists[kang_ind2 * 4 + 0] = dB[0];
		Kparams.dists[kang_ind2 * 4 + 1] = dB[1];
		Kparams.dists[kang_ind2 * 4 + 2] = dB[2];

		//store so cur_ind is 0 at next loading
		#pragma unroll
		for (int i = 0; i < MD_LEN; i++)
		{
			int ind = (i + MD_LEN - cur_indA) % MD_LEN;
			Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X + 2 * MD_LEN * BLOCK_SIZE * gr_ind2 + ind * BLOCK_SIZE + THREAD_X] = RegsA[i];
			ind = (i + MD_LEN - cur_indB) % MD_LEN;
			Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X + 2 * MD_LEN * BLOCK_SIZE * gr_ind2 + (ind + MD_LEN) * BLOCK_SIZE + THREAD_X] = RegsB[i];
		}
	}
}

//this kernel performs single jump3 for looped kangs
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelC(const TKparams Kparams)
{
	u64 PartStride = PNT_GROUP_CNT * (Kparams.BlockCnt * 256 * 32);
	u64* jmp3_table = LDS; //48KB

	int i = THREAD_X;
	while (i < JMP_CNT)
	{
		*(int4*)&jmp3_table[12 * i + 0] = *(int4*)&Kparams.Jumps3[12 * i + 0];
		*(int4*)&jmp3_table[12 * i + 2] = *(int4*)&Kparams.Jumps3[12 * i + 2];
		*(int4*)&jmp3_table[12 * i + 4] = *(int4*)&Kparams.Jumps3[12 * i + 4];
		*(int4*)&jmp3_table[12 * i + 6] = *(int4*)&Kparams.Jumps3[12 * i + 6];
		*(int4*)&jmp3_table[12 * i + 8] = *(int4*)&Kparams.Jumps3[12 * i + 8];
		*(int4*)&jmp3_table[12 * i + 10] = *(int4*)&Kparams.Jumps3[12 * i + 10];
		i += BLOCK_SIZE;
	}

	__syncthreads();

	while (1)
	{
		u32 ind = atomicAdd(Kparams.LoopedKangs + 1, 1);
		if (ind >= Kparams.LoopedKangs[0])
			break;
		u32 kang_ind = Kparams.LoopedKangs[2 + ind] & 0x0FFFFFFF;
		u32 last_ind = Kparams.LoopedKangs[2 + ind] >> 28;

		__align__(16) u64 x0[4], x[4];
		__align__(16) u64 y0[4], y[4];
		__align__(16) u64 jmp_x[4];
		__align__(16) u64 jmp_y[4];
		__align__(16) u64 inverse[5];
		u64 tmp[4], tmp2[4];

		u64* x_last0 = Kparams.LastPnts + 4 * kang_ind;
		u64* y_last0 = x_last0 + PartStride / 8;

		u64* x_last = x_last0 + last_ind * 2 * PartStride / 8;
		u64* y_last = y_last0 + last_ind * 2 * PartStride / 8;
		LOAD_VAL_256(x0, x_last, 0);
		LOAD_VAL_256(y0, y_last, 0);

		u32 jmp_ind = x0[0] % JMP_CNT;
		Copy_int4_x2(jmp_x, jmp3_table + 12 * jmp_ind);
		Copy_int4_x2(jmp_y, jmp3_table + 12 * jmp_ind + 4);
		SubModP(inverse, x0, jmp_x);
		InvModP((u32*)inverse);

		u32 inv_flag = y0[0] & 1;
		if (inv_flag)
			NegModP(jmp_y);

		SubModP(tmp, y0, jmp_y);
		MulModP(tmp2, tmp, inverse);
		SqrModP(tmp, tmp2);

		SubModP(x, tmp, jmp_x);
		SubModP(x, x, x0);
		SubModP(y, x0, x);
		MulModP(y, y, tmp2);
		SubModP(y, y, y0);

		//save kang
		Kparams.L2[4 * kang_ind + 0] = x[0];
		Kparams.L2[4 * kang_ind + 1] = x[1];
		Kparams.L2[4 * kang_ind + 2] = x[2];
		Kparams.L2[4 * kang_ind + 3] = x[3];
		Kparams.L2[4 * kang_ind + PartStride / 8 + 0] = y[0];
		Kparams.L2[4 * kang_ind + PartStride / 8 + 1] = y[1];
		Kparams.L2[4 * kang_ind + PartStride / 8 + 2] = y[2];
		Kparams.L2[4 * kang_ind + PartStride / 8 + 3] = y[3];

		//add distance
		u64 d[3];
		d[0] = Kparams.dists[kang_ind * 4 + 0];
		d[1] = Kparams.dists[kang_ind * 4 + 1];
		d[2] = Kparams.dists[kang_ind * 4 + 2];
		if (inv_flag)
			Sub192from192(d, jmp3_table + 12 * jmp_ind + 8)
		else
			Add192to192(d, jmp3_table + 12 * jmp_ind + 8);
		Kparams.dists[kang_ind * 4 + 0] = d[0];
		Kparams.dists[kang_ind * 4 + 1] = d[1];
		Kparams.dists[kang_ind * 4 + 2] = d[2];
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define GX_0	0x59F2815B16F81798ull
#define GX_1	0x029BFCDB2DCE28D9ull
#define GX_2	0x55A06295CE870B07ull
#define GX_3	0x79BE667EF9DCBBACull
#define GY_0	0x9C47D08FFB10D4B8ull
#define GY_1	0xFD17B448A6855419ull
#define GY_2	0x5DA4FBFC0E1108A8ull
#define GY_3	0x483ADA7726A3C465ull

__device__ __forceinline__ void AddPoints(u64* res_x, u64* res_y, u64* pnt1x, u64* pnt1y, u64* pnt2x, u64* pnt2y)
{
	__align__(16) u64 tmp[4], tmp2[4], lambda[4], lambda2[4];
	__align__(16) u64 inverse[5];
	SubModP(inverse, pnt2x, pnt1x);
	InvModP((u32*)inverse);
	SubModP(tmp, pnt2y, pnt1y);
	MulModP(lambda, tmp, inverse);
	MulModP(lambda2, lambda, lambda);
	SubModP(tmp, lambda2, pnt1x);
	SubModP(res_x, tmp, pnt2x);
	SubModP(tmp, pnt2x, res_x);
	MulModP(tmp2, tmp, lambda);
	SubModP(res_y, tmp2, pnt2y);
}

__device__ __forceinline__ void DoublePoint(u64* res_x, u64* res_y, u64* pntx, u64* pnty)
{
	__align__(16) u64 tmp[4], tmp2[4], lambda[4], lambda2[4];
	__align__(16) u64 inverse[5];
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

//this kernel calculates start points of kangs
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelGen(const TKparams Kparams)
{
	u32 PartStride = PNT_GROUP_CNT * (Kparams.BlockCnt * 256 * 32);
	u64* L2x = Kparams.L2 + 4 * (THREAD_X + BLOCK_SIZE * BLOCK_X);
	u64* L2y = L2x + PartStride / 8;
	u64* L2d = L2y + PartStride / 8;

	for (u32 group = 0; group < PNT_GROUP_CNT; group++)
	{
		__align__(16) u64 x0[4], y0[4], d[4];
		__align__(16) u64 x[4], y[4];
		__align__(16) u64 tx[4], ty[4];
		__align__(16) u64 t2x[4], t2y[4];
		
		u32 kang_ind = THREAD_X + BLOCK_X * BLOCK_SIZE + group * (BLOCK_SIZE * Kparams.BlockCnt);

		LOAD_VAL_256(x0, L2x, group)
		LOAD_VAL_256(y0, L2y, group)
		LOAD_VAL_256(d, L2d, group)
		
		tx[0] = GX_0; tx[1] = GX_1; tx[2] = GX_2; tx[3] = GX_3;
		ty[0] = GY_0; ty[1] = GY_1; ty[2] = GY_2; ty[3] = GY_3;

		bool first = true;
		int n = 2;
		while ((n >= 0) && !d[n]) 
			n--;
		if (n < 0)
			continue; //error
		int index = __clzll(d[n]);
		for (int i = 0; i <= 64 * n + (63 - index); i++)
		{
			u8 v = (d[i / 64] >> (i % 64)) & 1;
			if (v)
			{
				if (first)
				{
					first = false;
					Copy_u64_x4(x, tx);
					Copy_u64_x4(y, ty);
				}
				else
				{
					AddPoints(t2x, t2y, x, y, tx, ty);
					Copy_u64_x4(x, t2x);
					Copy_u64_x4(y, t2y);
				}
			}
			DoublePoint(t2x, t2y, tx, ty);
			Copy_u64_x4(tx, t2x);
			Copy_u64_x4(ty, t2y);
		}

		if (!Kparams.IsGenMode)
			if (kang_ind >= Kparams.KangCnt / 3)
			{
				AddPoints(t2x, t2y, x, y, x0, y0);
				Copy_u64_x4(x, t2x);
				Copy_u64_x4(y, t2y);
			}

		SAVE_VAL_256(L2x, x, group)
		SAVE_VAL_256(L2y, y, group)
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CallGpuKernelA(TKparams Kparams)
{
	KernelA << < Kparams.BlockCnt, Kparams.BlockSize, Kparams.KernelA_LDS_Size >> > (Kparams);
}
void CallGpuKernelB(TKparams Kparams)
{
	KernelB << < Kparams.BlockCnt, Kparams.BlockSize, Kparams.KernelB_LDS_Size >> > (Kparams);
}
void CallGpuKernelC(TKparams Kparams)
{
	KernelC << < Kparams.BlockCnt, Kparams.BlockSize, Kparams.KernelC_LDS_Size >> > (Kparams);
}

void CallGpuKernelGen(TKparams Kparams)
{
	KernelGen << < Kparams.BlockCnt, Kparams.BlockSize, 0 >> > (Kparams);
}

cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table)
{
	cudaError_t err = cudaFuncSetAttribute(KernelA, cudaFuncAttributeMaxDynamicSharedMemorySize, Kparams.KernelA_LDS_Size);
	if (err != cudaSuccess)
		return err;
	err = cudaFuncSetAttribute(KernelB, cudaFuncAttributeMaxDynamicSharedMemorySize, Kparams.KernelB_LDS_Size);
	if (err != cudaSuccess)
		return err;
	err = cudaFuncSetAttribute(KernelC, cudaFuncAttributeMaxDynamicSharedMemorySize, Kparams.KernelC_LDS_Size);
	if (err != cudaSuccess)
		return err;
	err = cudaMemcpyToSymbol(jmp2_table, _jmp2_table, JMP_CNT * 64);
	if (err != cudaSuccess)
		return err;
	return cudaSuccess;
}
