// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once

#include "Ec.h"

#ifdef USE_OPENCL
#include "OCLGpuUtils.h"
#else
#include "RCGpuUtils.h"
#endif

#define STATS_WND_SIZE	16

struct EcJMP
{
	EcPoint p;
	EcInt dist;
};

//96bytes size
struct TPointPriv
{
	u64 x[4];
	u64 y[4];
	u64 priv[4];
};

class RCGpuKang
{
private:
	bool StopFlag;
	EcPoint PntToSolve;
	int Range; //in bits
	int DP; //in bits
	Ec ec;

	u32* DPs_out;
	TKparams Kparams;

	EcInt HalfRange;
	EcPoint PntHalfRange;
	EcPoint NegPntHalfRange;
	TPointPriv* RndPnts;
	EcJMP* EcJumps1;
	EcJMP* EcJumps2;
	EcJMP* EcJumps3;

	EcPoint PntA;
	EcPoint PntB;

#ifdef USE_OPENCL
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    // Kernels
    cl_kernel kernel_A = NULL; // Will point to KernelA_main or KernelA_oldgpu
    cl_kernel kernel_B = NULL;
    cl_kernel kernel_C = NULL;
    cl_kernel kernel_Gen = NULL;
    // Memory Objects
    cl_mem d_Kangs_ocl = NULL;
    cl_mem d_Jumps1_ocl = NULL;
    cl_mem d_Jumps2_ocl = NULL;
    cl_mem d_Jumps3_ocl = NULL;
    cl_mem d_DPTable_ocl = NULL;
    cl_mem d_DPs_out_ocl = NULL;
    cl_mem d_L1S2_ocl = NULL;
    cl_mem d_LoopTable_ocl = NULL;
    cl_mem d_JumpsList_ocl = NULL;
    cl_mem d_LastPnts_ocl = NULL;
    cl_mem d_LoopedKangs_ocl = NULL;
    cl_mem d_dbg_buf_ocl = NULL;
    cl_mem d_jmp2_table_ocl = NULL;
    cl_mem d_L2_ocl = NULL; // For the L2 cache emulation buffer if used by non-OLD_GPU KernelA
#endif

	int cur_stats_ind;
	int SpeedStats[STATS_WND_SIZE];

	void GenerateRndDistances();
	bool Start();
	void Release();
#ifdef DEBUG_MODE
	int Dbg_CheckKangs();
#endif
public:
	int persistingL2CacheMaxSize;
	int CudaIndex; //gpu index in cuda
	int mpCnt;
	int KangCnt;
	bool Failed;
	bool IsOldGpu;

	int CalcKangCnt();
	bool Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3);
	void Stop();
	void Execute();

	u32 dbg[256];

	int GetStatsSpeed();
};
