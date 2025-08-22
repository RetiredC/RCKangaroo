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
#define LOAD_VAL_256(dst, ptr, group) { *((int4*)&(dst)[0]) = *((int4*)&(ptr)[BLOCK_SIZE * 4 * BLOCK_CNT * (group)]); *((int4*)&(dst)[2]) = *((int4*)&(ptr)[2 * BLOCK_SIZE + BLOCK_SIZE * 4 * BLOCK_CNT * (group)]); }
#define SAVE_VAL_256(ptr, src, group) { *((int4*)&(ptr)[BLOCK_SIZE * 4 * BLOCK_CNT * (group)]) = *((int4*)&(src)[0]); *((int4*)&(ptr)[2 * BLOCK_SIZE + BLOCK_SIZE * 4 * BLOCK_CNT * (group)]) = *((int4*)&(src)[2]); }


extern __shared__ u64 LDS[]; 

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef OLD_GPU

//this kernel performs main jumps
extern "C" __launch_bounds__(BLOCK_SIZE, 1)

// === Jacobian option (experimental) =========================================
// Para habilitar: compilar con -DUSE_JACOBIAN=1. Mantiene el formato de salida.
// Implementa sumas J+A y conversión por lotes sólo para la verificación DP.
// ============================================================================
__global__ void KernelA(const TKparams Kparams)
{
	u64* L2x = Kparams.L2 + 2 * THREAD_X + 4 * BLOCK_SIZE * BLOCK_X;
	u64* L2y = L2x + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;
	u64* L2s = L2y + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;
	//list of distances of performed jumps for KernelB
	int4* jlist = (int4*)(Kparams.JumpsList + (u64)BLOCK_X * STEP_CNT * PNT_GROUP_CNT * BLOCK_SIZE / 4);
	jlist += (THREAD_X / 32) * 32 * PNT_GROUP_CNT / 8;
	//list of last visited points for KernelC
	u64* x_last0 = Kparams.LastPnts + 2 * THREAD_X + 4 * BLOCK_SIZE * BLOCK_X;
	u64* y_last0 = x_last0 + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;
      
	u64* jmp1_table = LDS; //32KB
	u16* lds_jlist = (u16*)&LDS[8 * JMP_CNT]; //4KB, must be aligned 16bytes

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
	u64 dp_mask64 = ~((1ull << (64 - Kparams.DP)) - 1);
	u16 jmp_ind;

	//copy kangs from global to L2
	u32 kang_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE);
	for (u32 group = 0; group < PNT_GROUP_CNT; group++)
	{	
		tmp[0] = Kparams.Kangs[(kang_ind + group) * 12 + 0];
		tmp[1] = Kparams.Kangs[(kang_ind + group) * 12 + 1];
		tmp[2] = Kparams.Kangs[(kang_ind + group) * 12 + 2];
		tmp[3] = Kparams.Kangs[(kang_ind + group) * 12 + 3];
		SAVE_VAL_256(L2x, tmp, group);
		tmp[0] = Kparams.Kangs[(kang_ind + group) * 12 + 4];
		tmp[1] = Kparams.Kangs[(kang_ind + group) * 12 + 5];
		tmp[2] = Kparams.Kangs[(kang_ind + group) * 12 + 6];
		tmp[3] = Kparams.Kangs[(kang_ind + group) * 12 + 7];
		SAVE_VAL_256(L2y, tmp, group);
	}

	u32 L1S2 = Kparams.L1S2[BLOCK_X * BLOCK_SIZE + THREAD_X];

    for (int step_ind = 0; step_ind < STEP_CNT; step_ind++)
    {
#if USE_JACOBIAN
        // [Jacobian path] — usamos L2x=L2X, L2y=L2Y, L2s=L2Z
        __align__(16) u64 X[4], Y[4], Z[4];
        // 1) Cargar Z=1 si es la primera iteración
        for (int group = 0; group < PNT_GROUP_CNT; group++) {
            LOAD_VAL_256(X, L2x, group);
            LOAD_VAL_256(Y, L2y, group);
            if (step_ind==0) { Z[0]=1; Z[1]=Z[2]=Z[3]=0; }
            else { LOAD_VAL_256(Z, L2s, group); }
            // salto
            u64* jmp_table;
            __align__(16) u64 jmp_x[4], jmp_y[4];
            u16 jmp_ind_loc = X[0] % JMP_CNT;
            jmp_table = ((L1S2 >> group) & 1) ? jmp2_table : jmp1_table;
            Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind_loc);
            Copy_int4_x2(jmp_y, jmp_table + 8 * jmp_ind_loc + 4);
            // manejo de inversión de y según bit
            u32 inv_flag = (u32)Y[0] & 1u;
            if (inv_flag) { jmp_ind_loc |= INV_FLAG; NegModP(jmp_y); }
            u64 X3[4],Y3[4],Z3[4];
            JacobianAddMixed(X3,Y3,Z3,X,Y,Z,jmp_x,jmp_y);
            SAVE_VAL_256(L2x, X3, group);
            SAVE_VAL_256(L2y, Y3, group);
            SAVE_VAL_256(L2s, Z3, group);

            // Conversión batched a afín sólo para DP (una inversión total por hilo)
        }
        // Batch inversion: Z^-1 para todos los grupos (Montgomery trick)
        __align__(16) u64 prod[4] = {1,0,0,0};
        for (int group=0; group<PNT_GROUP_CNT; group++) {
            LOAD_VAL_256(Z, L2s, group);
            MulModP(prod, prod, Z);
            SAVE_VAL_256(L2s, prod, group); // prefijos
        }
        InvModP((u32*)prod); // prod = (∏Z)^-1
        for (int group=PNT_GROUP_CNT-1; group>=0; group--) {
            __align__(16) u64 prev[4] = {1,0,0,0};
            if (group>0) LOAD_VAL_256(prev, L2s, group-1);
            LOAD_VAL_256(Z, L2s, group);
            __align__(16) u64 invZ[4]; MulModP(invZ, prod, prev); // Z_i^-1
            // actualizar prod = prod * Z_i
            MulModP(prod, prod, Z);
            // x_aff = X * invZ^2
            __align__(16) u64 Xg[4], invZ2[4];
            LOAD_VAL_256(Xg, L2x, group);
            MulModP(invZ2, invZ, invZ);
            MulModP(Xg, Xg, invZ2);
            // check DP
            if ((Xg[3] & dp_mask64) == 0) {
                u32 kang_ind = (THREAD_X + BLOCK_X * BLOCK_SIZE) * PNT_GROUP_CNT + group;
                u32 ind = atomicAdd(Kparams.DPTable + kang_ind, 1);
                ind = min(ind, DPTABLE_MAX_CNT - 1);
                int4 *pdst = (int4*)(&Kparams.DPs_out[(u64)kang_ind * DPTABLE_MAX_CNT * 8 + 8 * ind]);
                ((int4*)pdst)[0] = ((int4*)Xg)[0];
                ((int4*)pdst)[1] = ((int4*)Xg)[1];
                // acción de salto (igual al camino original)
                // NOTA: reusamos jmp_ind_loc de arriba no disponible aquí; reconstituimos:
                LOAD_VAL_256(X, L2x, group);
                u16 ji = X[0] % JMP_CNT;
                ji |= ((u32)Y[0] & 1u) ? 0 : INV_FLAG;
                ((u16*)pdst)[8+0] = ji; // almacena índice salto+flags
                ((u16*)pdst)[8+1] = 0;
            }
        }
        continue; // saltamos el camino afín original
#endif // USE_JACOBIAN

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
			
			if ((x[3] & dp_mask64) == 0)
			{
				u32 kang_ind = (THREAD_X + BLOCK_X * BLOCK_SIZE) * PNT_GROUP_CNT + group;
				u32 ind = atomicAdd(Kparams.DPTable + kang_ind, 1);
				ind = min(ind, DPTABLE_MAX_CNT - 1);
				int4* dst = (int4*)(Kparams.DPTable + Kparams.KangCnt + (kang_ind * DPTABLE_MAX_CNT + ind) * 4);
				dst[0] = ((int4*)x)[0];
				jmp_ind |= DP_FLAG;
			}

			lds_jlist[8 * THREAD_X + (group % 8)] = jmp_ind;
			if ((group % 8) == 0)
				st_cs_v4_b32(&jlist[(group / 8) * 32 + (THREAD_X % 32)], *(int4*)&lds_jlist[8 * THREAD_X]); //skip L2 cache

			if (step_ind + MD_LEN >= STEP_CNT) //store last kangs to be able to find loop exit point
			{
				int n = step_ind + MD_LEN - STEP_CNT;
				u64* x_last = x_last0 + n * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
				u64* y_last = y_last0 + n * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
				SAVE_VAL_256(x_last, x, group);
				SAVE_VAL_256(y_last, y, group);
			}
        }
		jlist += PNT_GROUP_CNT * BLOCK_SIZE / 8;
    } 

	Kparams.L1S2[BLOCK_X * BLOCK_SIZE + THREAD_X] = L1S2;
	//copy kangs from L2 to global
	kang_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE);
	for (u32 group = 0; group < PNT_GROUP_CNT; group++)
	{
		LOAD_VAL_256(tmp, L2x, group);
		Kparams.Kangs[(kang_ind + group) * 12 + 0] = tmp[0];
		Kparams.Kangs[(kang_ind + group) * 12 + 1] = tmp[1];
		Kparams.Kangs[(kang_ind + group) * 12 + 2] = tmp[2];
		Kparams.Kangs[(kang_ind + group) * 12 + 3] = tmp[3];
		LOAD_VAL_256(tmp, L2y, group);
		Kparams.Kangs[(kang_ind + group) * 12 + 4] = tmp[0];
		Kparams.Kangs[(kang_ind + group) * 12 + 5] = tmp[1];
		Kparams.Kangs[(kang_ind + group) * 12 + 6] = tmp[2];
		Kparams.Kangs[(kang_ind + group) * 12 + 7] = tmp[3];
	}
} 

#else

#define LOAD_VAL_256_m(dst,p,i) { *((int4*)&(dst)[0]) = *((int4*)&(p)[4 * (i)]); *((int4*)&(dst)[2]) = *((int4*)&(p)[4 * (i) + 2]); }
#define SAVE_VAL_256_m(p,src,i) { *((int4*)&(p)[4 * (i)]) = *((int4*)&(src)[0]); *((int4*)&(p)[4 * (i) + 2]) = *((int4*)&(src)[2]); }


//this kernel performs main jumps for old cards
//not good but works
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelA(const TKparams Kparams)
{
	__align__(16) u64 Lx[4 * PNT_GROUP_CNT];
	__align__(16) u64 Ly[4 * PNT_GROUP_CNT];
	__align__(16) u64 Ls[4 * PNT_GROUP_CNT / 2]; //we store only half so need only half mem

	//list of distances of performed jumps for KernelB
	int4* jlist = (int4*)(Kparams.JumpsList + (u64)BLOCK_X * STEP_CNT * PNT_GROUP_CNT * BLOCK_SIZE / 4);
	jlist += (THREAD_X / 32) * 32 * PNT_GROUP_CNT / 8;
	//list of last visited points for KernelC
	u64* x_last0 = Kparams.LastPnts + 2 * THREAD_X + 4 * BLOCK_SIZE * BLOCK_X;
	u64* y_last0 = x_last0 + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;

	u64* jmp1_table = LDS; //32KB
	u16* lds_jlist = (u16*)&LDS[8 * JMP_CNT]; //8KB, must be aligned 16bytes

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

	__align__(16) u64 inverse[5];
	__align__(16) u64 x[4], y[4], tmp[4], tmp2[4];
	u64 dp_mask64 = ~((1ull << (64 - Kparams.DP)) - 1);
	u16 jmp_ind;

	//copy kangs from global to local
	u32 kang_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE);
	for (u32 group = 0; group < PNT_GROUP_CNT; group++)
	{
		tmp[0] = Kparams.Kangs[(kang_ind + group) * 12 + 0];
		tmp[1] = Kparams.Kangs[(kang_ind + group) * 12 + 1];
		tmp[2] = Kparams.Kangs[(kang_ind + group) * 12 + 2];
		tmp[3] = Kparams.Kangs[(kang_ind + group) * 12 + 3];
		SAVE_VAL_256_m(Lx, tmp, group);
		tmp[0] = Kparams.Kangs[(kang_ind + group) * 12 + 4];
		tmp[1] = Kparams.Kangs[(kang_ind + group) * 12 + 5];
		tmp[2] = Kparams.Kangs[(kang_ind + group) * 12 + 6];
		tmp[3] = Kparams.Kangs[(kang_ind + group) * 12 + 7];
		SAVE_VAL_256_m(Ly, tmp, group);
	}

	u64 L1S2 = ((u64*)Kparams.L1S2)[BLOCK_X * BLOCK_SIZE + THREAD_X];
	u64* jmp_table;
	__align__(16) u64 jmp_x[4];
	__align__(16) u64 jmp_y[4];

	//preparations (first calc for inv)
	for (int group = 0; group < PNT_GROUP_CNT; group++)
	{
		LOAD_VAL_256_m(x, Lx, group);
		jmp_ind = x[0] % JMP_CNT;
		jmp_table = ((L1S2 >> group) & 1) ? jmp2_table : jmp1_table;
		Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
		SubModP(tmp, x, jmp_x);
		if (group == 0)
		{
			Copy_u64_x4(inverse, tmp);
			SAVE_VAL_256_m(Ls, tmp, 0);
		}
		else
		{
			MulModP(inverse, inverse, tmp);
			if ((group & 1) == 0)
				SAVE_VAL_256_m(Ls, inverse, group / 2);
		}
	}

	//main loop
	int g_beg = PNT_GROUP_CNT - 1; //start val
	int g_end = -1; //first val after range
	int g_inc = -1;
	int s_mask = 1;
	int jlast_add = 0;
	__align__(16) u64 t_cache[4], x0_cache[4], jmpx_cached[4];
	t_cache[0] = t_cache[1] = t_cache[2] = t_cache[3] = 0;
	x0_cache[0] = x0_cache[1] = x0_cache[2] = x0_cache[3] = 0;

	for (int step_ind = 0; step_ind < STEP_CNT; step_ind++)
	{
		__align__(16) u64 next_inv[4];

		InvModP((u32*)inverse);

		int group = g_beg;
		bool cached = false;
		while (group != g_end)
		{
			__align__(16) u64 dx[4], x0[4], y0[4], dx0[4];
			if (cached)
			{
				Copy_u64_x4(x0, x0_cache);
			}
			else
			{
				LOAD_VAL_256_m(x0, Lx, group);
			}
			LOAD_VAL_256_m(y0, Ly, group);

			jmp_ind = x0[0] % JMP_CNT;
			jmp_table = ((L1S2 >> group) & 1) ? jmp2_table : jmp1_table;
			if (cached)
			{
				Copy_u64_x4(jmp_x, jmpx_cached); 
			}
			else
			{
				Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
			}
			Copy_int4_x2(jmp_y, jmp_table + 8 * jmp_ind + 4);
			u32 inv_flag = (u32)y0[0] & 1;
			if (inv_flag)
			{
				jmp_ind |= INV_FLAG;
				NegModP(jmp_y);
			}

			if (group == g_end - g_inc)
			{
				Copy_u64_x4(dx0, inverse);
			}
			else
			{
				if ((group & 1) == s_mask) //simple case
				{
					if (cached)
					{
						Copy_u64_x4(tmp, t_cache);
						cached = false;
					}
					else
					{
						LOAD_VAL_256_m(tmp, Ls, (group + g_inc) / 2);
					}
				}
				else //no s(-1), need to calc it
				{
					LOAD_VAL_256_m(t_cache, Ls, (group + g_inc + g_inc) / 2);
					cached = true;				
					LOAD_VAL_256_m(x0_cache, Lx, group + g_inc);
					u32 jmp_tmp = x0_cache[0] % JMP_CNT;
					__align__(16) u64 dx2[4];
					u64* jmp_table_tmp = ((L1S2 >> (group + g_inc)) & 1) ? jmp2_table : jmp1_table;
					Copy_int4_x2(jmpx_cached, jmp_table_tmp + 8 * jmp_tmp);
					SubModP(dx2, x0_cache, jmpx_cached);
					MulModP(tmp, t_cache, dx2); //t = s(-1)
				}

				SubModP(dx, x0, jmp_x);
				MulModP(dx0, tmp, inverse);
				MulModP(inverse, inverse, dx);
			}

			SubModP(tmp2, y0, jmp_y);
			MulModP(tmp, tmp2, dx0);
			SqrModP(tmp2, tmp);

			SubModP(x, tmp2, jmp_x);
			SubModP(x, x, x0);
			SAVE_VAL_256_m(Lx, x, group);

			SubModP(y, x0, x);
			MulModP(y, y, tmp);
			SubModP(y, y, y0);
			SAVE_VAL_256_m(Ly, y, group);

			if (((L1S2 >> group) & 1) == 0) //normal mode, check L1S2 loop
			{
				u32 jmp_next = x[0] % JMP_CNT;
				jmp_next |= ((u32)y[0] & 1) ? 0 : INV_FLAG; //inverted
				L1S2 |= (jmp_ind == jmp_next) ? (1ull << group) : 0; //loop L1S2 detected
			}
			else
			{
				L1S2 &= ~(1ull << group);
				jmp_ind |= JMP2_FLAG;
			}

			if ((x[3] & dp_mask64) == 0)
			{
				u32 kang_ind = (THREAD_X + BLOCK_X * BLOCK_SIZE) * PNT_GROUP_CNT + group;
				u32 ind = atomicAdd(Kparams.DPTable + kang_ind, 1);
				ind = min(ind, DPTABLE_MAX_CNT - 1);
				int4* dst = (int4*)(Kparams.DPTable + Kparams.KangCnt + (kang_ind * DPTABLE_MAX_CNT + ind) * 4);
				dst[0] = ((int4*)x)[0];
				jmp_ind |= DP_FLAG;
			}

			lds_jlist[8 * THREAD_X + (group % 8)] = jmp_ind;
			if (((group + jlast_add) % 8) == 0)
				st_cs_v4_b32(&jlist[(group / 8) * 32 + (THREAD_X % 32)], *(int4*)&lds_jlist[8 * THREAD_X]); //skip L2 cache

			if (step_ind + MD_LEN >= STEP_CNT) //store last kangs to be able to find loop exit point
			{
				int n = step_ind + MD_LEN - STEP_CNT;
				u64* x_last = x_last0 + n * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
				u64* y_last = y_last0 + n * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
				SAVE_VAL_256(x_last, x, group);
				SAVE_VAL_256(y_last, y, group);
			}
		
			//preps to calc next inv
			jmp_ind = x[0] % JMP_CNT;
			jmp_table = ((L1S2 >> group) & 1) ? jmp2_table : jmp1_table;
			Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
			SubModP(dx, x, jmp_x);
			if (group == g_beg)
			{
				Copy_u64_x4(next_inv, dx);
				SAVE_VAL_256_m(Ls, dx, g_beg / 2);
			}
			else
			{
				MulModP(next_inv, next_inv, dx);
				if ((group & 1) == s_mask)
				{
					SAVE_VAL_256_m(Ls, next_inv, group / 2);
				}
			}

			group += g_inc;
		} //group
		jlist += PNT_GROUP_CNT * BLOCK_SIZE / 8;
		Copy_u64_x4(inverse, next_inv);
		if (g_inc < 0) //invert direction
		{
			g_beg = 0;
			g_end = PNT_GROUP_CNT;
			g_inc = 1;
			s_mask = 0;
			jlast_add = 1;
		}
		else
		{
			g_beg = PNT_GROUP_CNT - 1;
			g_end = -1;
			g_inc = -1;
			s_mask = 1;
			jlast_add = 0;
		}
	}

	((u64*)Kparams.L1S2)[BLOCK_X * BLOCK_SIZE + THREAD_X] = L1S2;
	//copy kangs from local to global
	kang_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE);
	for (u32 group = 0; group < PNT_GROUP_CNT; group++)
	{
		LOAD_VAL_256_m(tmp, Lx, group);
		Kparams.Kangs[(kang_ind + group) * 12 + 0] = tmp[0];
		Kparams.Kangs[(kang_ind + group) * 12 + 1] = tmp[1];
		Kparams.Kangs[(kang_ind + group) * 12 + 2] = tmp[2];
		Kparams.Kangs[(kang_ind + group) * 12 + 3] = tmp[3];
		LOAD_VAL_256_m(tmp, Ly, group);
		Kparams.Kangs[(kang_ind + group) * 12 + 4] = tmp[0];
		Kparams.Kangs[(kang_ind + group) * 12 + 5] = tmp[1];
		Kparams.Kangs[(kang_ind + group) * 12 + 6] = tmp[2];
		Kparams.Kangs[(kang_ind + group) * 12 + 7] = tmp[3];
	}
}

#endif

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
	DPs[10] = 3 * kang_ind / Kparams.KangCnt; //kang type
}

__device__ __forceinline__ bool ProcessJumpDistance(u32 step_ind, u32 d_cur, u64* d, u32 kang_ind, u64* jmp1_d, u64* jmp2_d, const TKparams& Kparams, u64* table, u32* cur_ind, u8 iter)
{
	u64* jmp_d = (d_cur & JMP2_FLAG) ? jmp2_d : jmp1_d;

	__align__(16) u64 jmp[3];
	((int4*)(jmp))[0] = ((int4*)(jmp_d + 4 * (d_cur & JMP_MASK)))[0];
	jmp[2] = *(jmp_d + 4 * (d_cur & JMP_MASK) + 2);

	if (d_cur & INV_FLAG)
		Sub192from192(d, jmp)
	else
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

	// --- Warp-aggregated emisión de DPs (reemplaza BuildDP) ---
if (found_ind < 0)
{
    // 1) Cada hilo decide si «quiere emitir» un DP en este paso
    //    (antes llamábamos a BuildDP)
    bool emit = false;
    int4 rx_local;  // X parcial del DP (lo que lectura BuildDP hacía)
    if (d_cur & DP_FLAG) {
        // Replicamos la parte de BuildDP que obtiene el X guardado en DPTable
        int idx = atomicAdd(Kparams.DPTable + kang_ind, 0x10000);
        idx >>= 16;  // índice de lectura (alto de 16 bits)
        if (idx < DPTABLE_MAX_CNT) {
            rx_local = *(int4*)(Kparams.DPTable
                       + Kparams.KangCnt
                       + (kang_ind * DPTABLE_MAX_CNT + idx) * 4);
            emit = true;
        }
    }

    // 2) Warp-aggregated atomic: una sola atomicAdd por warp
    const unsigned active = __activemask();
    const int lane = threadIdx.x & 31;
    const int leader = __ffs(active) - 1;              // primer lane activo
    unsigned mask_emit = __ballot_sync(active, emit);  // quiénes emiten en el warp
    int hits = __popc(mask_emit);                      // cuántos DPs emiten

    if (hits) {
        // Sólo el líder reserva 'hits' slots contiguos en la salida
        unsigned base = 0;
        if (lane == leader) {
            base = atomicAdd(Kparams.DPs_out, (unsigned)hits);
        }
        // Broadcast del 'base' al warp
        base = __shfl_sync(active, base, leader);

        if (emit) {
            // Posición compacta de este hilo dentro del bloque reservado
            unsigned laneMask = mask_emit & ((1u << lane) - 1u);
            unsigned pos = __popc(laneMask);

            unsigned outIdx = base + pos;
            outIdx = min(outIdx, (unsigned)(MAX_DP_CNT - 1));

            // Escribir el registro exactamente igual que BuildDP
            u32* DPs = Kparams.DPs_out + 4 + outIdx * (GPU_DP_SIZE / 4);
            *(int4*)&DPs[0] = rx_local;              // X parcial (int4)
            *(int4*)&DPs[4] = ((int4*)d)[0];         // dist[0..1] (128 bits)
            *(u64*)&DPs[8] = d[2];                   // dist[2]   (64  bits)
            DPs[10] = 3 * kang_ind / Kparams.KangCnt; // tipo de kanguro
        }
    }

    return false;
}


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
		LoopedA = ProcessJumpDistance(step_ind, cur_dA, dA, kang_ind, jmp1_d, jmp2_d, Kparams, RegsA, &cur_indA, iter); \
	if (!LoopedB) \
		LoopedB = ProcessJumpDistance(step_ind, cur_dB, dB, kang_ind + 1, jmp1_d, jmp2_d, Kparams, RegsB, &cur_indB, iter); \
	jlist += BLOCK_SIZE * PNT_GROUP_CNT / 2; \
	step_ind++; \
}

//this kernel counts distances and detects loops Size>2
//Loops Level1 statistics for JMP_CNT=512: L1S2 = 1/1024 (so one loop every 1024 jumps), L1S4 = L1S2/1024, L1S6 = L1S4/256, L1S8 = L1S6/158, L1S10 = L1S8/82. L1S12 = L1S10/50. 
// For RTX4090 at 8HG/s for 24 hours and JMP_CNT=512: jumps = 691200bln, L1S2 = 682bln, L1S4 = 666mln, L1S6 = 2.6mln, L1S8 = 16.5k, L1S10 = 201. L1S12 = 4.
// I don't see any reasons to catch L1S12 because we have 786432 kangs, if we lose 4 kangs every day, we lose 1460 kangs a year which is about 0.19%.
// This degradation depends only on speed of a single kangaroo, so it's about the same for all 40xx GPUs (50xx GPUs will have +20% clock speed may be).
// Since we lose kangs gradually, for a year we lose 0.19/2 = 0.1% of speed, so you should catch L1S12 only if you are going to solve same point for decades.
// Or you can check all kangs for L1S12 on CPU once a day and restart looped kangs.
// Level2 loops are very rare and they have even size too so they will be handled by the same code. We don't know what loop level we catch so we use JmpTable3 for escaping.
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelB(const TKparams Kparams)
{
	u64* jmp1_d = LDS; //16KB, 192bit jumps
	u64* jmp2_d = LDS + 4 * JMP_CNT; //16KB, 192bit jumps

	int i = THREAD_X;
	while (i < JMP_CNT)
	{
		//192bits but we need align 128 so use 256
		jmp1_d[4 * i + 0] = Kparams.Jumps1[12 * i + 8];
		jmp1_d[4 * i + 1] = Kparams.Jumps1[12 * i + 9];
		jmp1_d[4 * i + 2] = Kparams.Jumps1[12 * i + 10];
		jmp2_d[4 * i + 0] = Kparams.Jumps2[12 * i + 8];
		jmp2_d[4 * i + 1] = Kparams.Jumps2[12 * i + 9];
		jmp2_d[4 * i + 2] = Kparams.Jumps2[12 * i + 10];
		i += BLOCK_SIZE;
	}

	u32* jlist0 = (u32*)(Kparams.JumpsList + (u64)BLOCK_X * STEP_CNT * PNT_GROUP_CNT * BLOCK_SIZE / 4);

	__syncthreads();

	u64 RegsA[MD_LEN], RegsB[MD_LEN];

	//we process two kangs at once
	for (u32 gr_ind2 = 0; gr_ind2 < PNT_GROUP_CNT/2; gr_ind2++)
	{	
		#pragma unroll
		for (int i = 0; i < MD_LEN; i++)
		{
			RegsA[i] = Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X + 2 * MD_LEN * BLOCK_SIZE * gr_ind2 + i * BLOCK_SIZE + BLOCK_X];
			RegsB[i] = Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X + 2 * MD_LEN * BLOCK_SIZE * gr_ind2 + (i + MD_LEN) * BLOCK_SIZE + BLOCK_X];
		}
		u32 cur_indA = 0;
		u32 cur_indB = 0;

		u32* jlist = jlist0 + gr_ind2 * BLOCK_SIZE;

		//calc original kang_ind
		u32 tind = (THREAD_X + gr_ind2 * BLOCK_SIZE); //0..3071
		u32 warp_ind = tind / (32 * PNT_GROUP_CNT / 2); // 0..7	
		u32 thr_ind = (tind / 4) % 32; //index in warp 0..31
		u32 g8_ind = (tind % (32 * PNT_GROUP_CNT / 2)) / 128; // 0..2
		u32 gr_ind = 2 * (tind % 4); // 0, 2, 4, 6

		u32 kang_ind = (BLOCK_X * BLOCK_SIZE) * PNT_GROUP_CNT;
		kang_ind += (32 * warp_ind + thr_ind) * PNT_GROUP_CNT + 8 * g8_ind + gr_ind;

		__align__(8) u64 dA[3], dB[3];
		dA[0] = Kparams.Kangs[kang_ind * 12 + 8];
		dA[1] = Kparams.Kangs[kang_ind * 12 + 9];
		dA[2] = Kparams.Kangs[kang_ind * 12 + 10];
		dB[0] = Kparams.Kangs[(kang_ind + 1) * 12 + 8];
		dB[1] = Kparams.Kangs[(kang_ind + 1) * 12 + 9];
		dB[2] = Kparams.Kangs[(kang_ind + 1) * 12 + 10];

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

		Kparams.Kangs[kang_ind * 12 + 8] = dA[0];
		Kparams.Kangs[kang_ind * 12 + 9] = dA[1];
		Kparams.Kangs[kang_ind * 12 + 10] = dA[2];
		Kparams.Kangs[(kang_ind + 1) * 12 + 8] = dB[0];
		Kparams.Kangs[(kang_ind + 1) * 12 + 9] = dB[1];
		Kparams.Kangs[(kang_ind + 1) * 12 + 10] = dB[2];

		//store so cur_ind is 0 at next loading
		#pragma unroll
		for (int i = 0; i < MD_LEN; i++)
		{
			int ind = (i + MD_LEN - cur_indA) % MD_LEN;
			Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X + 2 * MD_LEN * BLOCK_SIZE * gr_ind2 + ind * BLOCK_SIZE + BLOCK_X] = RegsA[i];
			ind = (i + MD_LEN - cur_indB) % MD_LEN;
			Kparams.LoopTable[MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * BLOCK_X + 2 * MD_LEN * BLOCK_SIZE * gr_ind2 + (ind + MD_LEN) * BLOCK_SIZE + BLOCK_X] = RegsB[i];
		}
	}
}

//this kernel performes single jump3 for looped kangs
extern "C" __launch_bounds__(BLOCK_SIZE, 1)
__global__ void KernelC(const TKparams Kparams)
{
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

		u64* x_last0 = Kparams.LastPnts;
		u64* y_last0 = x_last0 + 4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE;

		u32 block_ind = kang_ind / (BLOCK_SIZE * PNT_GROUP_CNT);
		u32 thr_ind = (kang_ind - block_ind * (BLOCK_SIZE * PNT_GROUP_CNT)) / PNT_GROUP_CNT;
		u32 gr_ind = (kang_ind - block_ind * (BLOCK_SIZE * PNT_GROUP_CNT) - thr_ind * PNT_GROUP_CNT);

		y_last0 += 2 * thr_ind + 4 * BLOCK_SIZE * block_ind;
		x_last0 += 2 * thr_ind + 4 * BLOCK_SIZE * block_ind;
		u64* x_last = x_last0 + last_ind * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
		u64* y_last = y_last0 + last_ind * 2 * (4 * PNT_GROUP_CNT * BLOCK_CNT * BLOCK_SIZE);
		LOAD_VAL_256(x0, x_last, gr_ind);
		LOAD_VAL_256(y0, y_last, gr_ind);

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
		Kparams.Kangs[kang_ind * 12 + 0] = x[0];
		Kparams.Kangs[kang_ind * 12 + 1] = x[1];
		Kparams.Kangs[kang_ind * 12 + 2] = x[2];
		Kparams.Kangs[kang_ind * 12 + 3] = x[3];
		Kparams.Kangs[kang_ind * 12 + 4] = y[0];
		Kparams.Kangs[kang_ind * 12 + 5] = y[1];
		Kparams.Kangs[kang_ind * 12 + 6] = y[2];
		Kparams.Kangs[kang_ind * 12 + 7] = y[3];

		//add distance
		u64 d[3];
		d[0] = Kparams.Kangs[kang_ind * 12 + 8];
		d[1] = Kparams.Kangs[kang_ind * 12 + 9];
		d[2] = Kparams.Kangs[kang_ind * 12 + 10];
		if (inv_flag)
			Sub192from192(d, jmp3_table + 12 * jmp_ind + 8)
		else
			Add192to192(d, jmp3_table + 12 * jmp_ind + 8);
		Kparams.Kangs[kang_ind * 12 + 8] = d[0];
		Kparams.Kangs[kang_ind * 12 + 9] = d[1];
		Kparams.Kangs[kang_ind * 12 + 10] = d[2];

#ifndef OLD_GPU
		atomicAnd(&Kparams.L1S2[block_ind * BLOCK_SIZE + thr_ind], ~(1u << gr_ind));
#else
		atomicAnd(&((u64*)Kparams.L1S2)[block_ind * BLOCK_SIZE + thr_ind], ~(1ull << gr_ind));
#endif
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
	for (u32 group = 0; group < PNT_GROUP_CNT; group++)
	{
		__align__(16) u64 x0[4], y0[4], d[3];
		__align__(16) u64 x[4], y[4];
		__align__(16) u64 tx[4], ty[4];
		__align__(16) u64 t2x[4], t2y[4];

		u32 kang_ind = PNT_GROUP_CNT * (THREAD_X + BLOCK_X * BLOCK_SIZE) + group;
		x0[0] = Kparams.Kangs[kang_ind * 12 + 0];
		x0[1] = Kparams.Kangs[kang_ind * 12 + 1];
		x0[2] = Kparams.Kangs[kang_ind * 12 + 2];
		x0[3] = Kparams.Kangs[kang_ind * 12 + 3];
		y0[0] = Kparams.Kangs[kang_ind * 12 + 4];
		y0[1] = Kparams.Kangs[kang_ind * 12 + 5];
		y0[2] = Kparams.Kangs[kang_ind * 12 + 6];
		y0[3] = Kparams.Kangs[kang_ind * 12 + 7];
		d[0] = Kparams.Kangs[kang_ind * 12 + 8];
		d[1] = Kparams.Kangs[kang_ind * 12 + 9];
		d[2] = Kparams.Kangs[kang_ind * 12 + 10];
		
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

		Kparams.Kangs[kang_ind * 12 + 0] = x[0];
		Kparams.Kangs[kang_ind * 12 + 1] = x[1];
		Kparams.Kangs[kang_ind * 12 + 2] = x[2];
		Kparams.Kangs[kang_ind * 12 + 3] = x[3];
		Kparams.Kangs[kang_ind * 12 + 4] = y[0];
		Kparams.Kangs[kang_ind * 12 + 5] = y[1];
		Kparams.Kangs[kang_ind * 12 + 6] = y[2];
		Kparams.Kangs[kang_ind * 12 + 7] = y[3];
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CallGpuKernelABC(TKparams Kparams)
{
	KernelA <<< Kparams.BlockCnt, Kparams.BlockSize, Kparams.KernelA_LDS_Size >>> (Kparams);
	KernelB <<< Kparams.BlockCnt, Kparams.BlockSize, Kparams.KernelB_LDS_Size >>> (Kparams);
	KernelC <<< Kparams.BlockCnt, Kparams.BlockSize, Kparams.KernelC_LDS_Size >>> (Kparams);
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
