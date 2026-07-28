// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"

#include "GpuKang.h"

cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table);
void CallGpuKernelGen(TKparams Kparams);
void CallGpuKernelA(TKparams Kparams);
void CallGpuKernelB(TKparams Kparams);
void CallGpuKernelC(TKparams Kparams);

void AddPointsToList(u32* data, int cnt, u32 KangCnt, u64 ops_cnt, int JumperInd);
extern bool gGenMode; //tames generation mode

int RCGpuKang::CalcKangCnt()
{
	Kparams.BlockCnt = mpCnt;
	Kparams.BlockSize = BLOCK_SIZE;
	Kparams.GroupCnt = PNT_GROUP_CNT;
	return Kparams.BlockSize* Kparams.GroupCnt* Kparams.BlockCnt;
}

//executes in main thread
bool RCGpuKang::Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3)
{
	PntToSolve = _PntToSolve;
	Range = _Range;
	DP = _DP;
	EcJumps1 = _EcJumps1;
	EcJumps2 = _EcJumps2;
	EcJumps3 = _EcJumps3;
	StopFlag = false;
	Failed = false;
	u64 total_mem = 0;
	memset(dbg, 0, sizeof(dbg));
	memset(SpeedStats, 0, sizeof(SpeedStats));
	cur_stats_ind = 0;

	cudaError_t err;
	err = cudaSetDevice(CudaIndex);
	if (err != cudaSuccess)
		return false;

	char path[500];
	path[0] = 0;
//	GetExeDir(path, 500);
//	strcat(path, "/");
	if (Is5xxx)
		strcat(path, "kernel_sm120.cubin");
	else
		strcat(path, "kernel_sm89.cubin");
	if (!cc.LoadCubin(path))
		return false;

	Kparams.BlockCnt = mpCnt - sm_inv_cnt;
	Kparams.BlockSize = BLOCK_SIZE;
	Kparams.GroupCnt = PNT_GROUP_CNT;
	KangCnt = Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
	Kparams.KangCnt = KangCnt;
	Kparams.DP = DP;
	Kparams.KernelA_LDS_Size = 98 * 1024;
	Kparams.KernelB_LDS_Size = 48 * 1024;
	Kparams.KernelC_LDS_Size = 96 * JMP_CNT;
	Kparams.IsGenMode = gGenMode;
	Kparams.dp_mask = (u32)((1ull << DP) - 1);
	Kparams.iter_cnt = STEP_CNT;
	Kparams.StopThr = (int)(0.5 * (Kparams.BlockCnt * 8)); //at the end, work will be stopped when number of finished producers is higher than this value. Must be <(WarpCnt-32)

//allocate gpu mem
	u64 size;
	//L2	
	//additional data for asm kernels: about 650KB only
	Inv_DataSize = 1024; //a few atomic counters (better to place them at 0x00 and at 0x80 to avoid same L2 line) plus some reserved (padding), 1024 bytes total //0x400
	Inv_DataSize += 4 * 256 * 8; //atomic iter counters for warps(producers) //0x2000
	Inv_DataSize += 4 * (32 * 1024); //queue 32K cells
	Inv_DataSize += (32 * 4) * 256 * 8; //plus mailboxes OUT (send), 256 - max number of SM, every SM has 8 warps, so 2K producers
	Inv_DataSize += (32 * 4) * 256 * 8; //plus mailboxes IN (recv), 256 - max number of SM, every SM has 8 warps, so 2K producers
	Inv_DataSize += 4 * 256 * 8; //plus ReadyFlag for mailboxes

	int L2size = Kparams.KangCnt * (3 * 32) + Inv_DataSize;
	total_mem += L2size;
	err = cudaMalloc((void**)&Kparams.L2, L2size);
	if (err != cudaSuccess)
	{
		printf("GPU %d, Allocate L2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	size = L2size;
	if (size > persistingL2CacheMaxSize)
		size = persistingL2CacheMaxSize;
	err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); // set max allowed size for L2
	//persisting for L2

	cudaStreamAttrValue stream_attribute;                                                   
	stream_attribute.accessPolicyWindow.base_ptr = Kparams.L2;
	stream_attribute.accessPolicyWindow.num_bytes = size;										
	stream_attribute.accessPolicyWindow.hitRatio = 1.0;                                     
	stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;             
	stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;  	
	err = cudaStreamSetAttribute(NULL, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaStreamSetAttribute failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	
	size = MAX_DP_CNT * GPU_DP_SIZE + 16;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.DPs_out, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate GpuOut memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.Jumps1, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.Jumps2, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.Jumps3, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps3 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = 2 * (u64)KangCnt * (STEP_CNT + MD_LEN);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.JumpsList, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate JumpsList memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * (16 * DPTABLE_MAX_CNT + sizeof(u32)); //we store 16bytes of X
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.DPTable, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate DPTable memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = mpCnt * Kparams.BlockSize * sizeof(u64);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.L1S2, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate L1S2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * MD_LEN * (2 * 32);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.LastPnts, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * MD_LEN * sizeof(u64);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.LoopTable, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += 1024;
	err = cudaMalloc((void**)&Kparams.dbg_buf, 1024);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate dbg_buf memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = sizeof(u32) * KangCnt + 8;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.LoopedKangs, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate LoopedKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = 32 * KangCnt;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.dists, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate dists memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	DPs_out = (u32*)malloc(MAX_DP_CNT * GPU_DP_SIZE);
	/////////////////
	size = JMP_CNT * 32 * 2 * 3;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.Jumps12, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps12 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	u32* pJumps12 = (u32*)malloc(JMP_CNT * 32 * 2 * 3); //96KB
	int part_ofs = 4 * JMP_CNT;
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(pJumps12 + i * 4, EcJumps1[i].p.x.data, 16);
		memcpy(pJumps12 + i * 4 + part_ofs, EcJumps1[i].p.x.data + 2, 16);
		memcpy(pJumps12 + i * 4 + 2 * part_ofs, EcJumps1[i].p.y.data, 16);
		memcpy(pJumps12 + i * 4 + 3 * part_ofs, EcJumps1[i].p.y.data + 2, 16);

		EcInt ng = EcJumps1[i].p.y;
		ng.NegModP();
		memcpy(pJumps12 + i * 4 + 4 * part_ofs, ng.data, 16);
		memcpy(pJumps12 + i * 4 + 5 * part_ofs, ng.data + 2, 16);
	}
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(pJumps12 + 48 * 1024 / 4 + i * 4, EcJumps2[i].p.x.data, 16);
		memcpy(pJumps12 + 48 * 1024 / 4 + i * 4 + part_ofs, EcJumps2[i].p.x.data + 2, 16);
		memcpy(pJumps12 + 48 * 1024 / 4 + i * 4 + 2 * part_ofs, EcJumps2[i].p.y.data, 16);
		memcpy(pJumps12 + 48 * 1024 / 4 + i * 4 + 3 * part_ofs, EcJumps2[i].p.y.data + 2, 16);

		EcInt ng = EcJumps2[i].p.y;
		ng.NegModP();
		memcpy(pJumps12 + 48 * 1024 / 4 + i * 4 + 4 * part_ofs, ng.data, 16);
		memcpy(pJumps12 + 48 * 1024 / 4 + i * 4 + 5 * part_ofs, ng.data + 2, 16);
	}
	err = cudaMemcpy(Kparams.Jumps12, pJumps12, JMP_CNT * 32 * 2 * 3, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps1 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	/////////////////
	free(pJumps12);


	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.JmpDists12, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate JmpDists12 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	EcInt val193b;
	val193b.SetHexStr("1000000000000000000000000000000000000000000000000");
	u32* jd12 = (u32*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(jd12 + i * 4, EcJumps1[i].dist.data, 16);
		memcpy(jd12 + i * 2 + (32 * 1024 / 4), EcJumps1[i].dist.data + 2, 8);
		EcInt neg = val193b;
		neg.Sub(EcJumps1[i].dist);
		memcpy(jd12 + i * 4 + (8 * 1024 / 4), neg.data, 16);
		memcpy(jd12 + i * 2 + (32 * 1024 / 4) + (4 * 1024 / 4), neg.data + 2, 8);

		memcpy(jd12 + i * 4 + (16 * 1024 / 4), EcJumps2[i].dist.data, 16);
		memcpy(jd12 + i * 2 + (32 * 1024 / 4) + (8 * 1024 / 4), EcJumps2[i].dist.data + 2, 8);
		neg = val193b;
		neg.Sub(EcJumps2[i].dist);
		memcpy(jd12 + i * 4 + (24 * 1024 / 4), neg.data, 16);
		memcpy(jd12 + i * 2 + (32 * 1024 / 4) + (12 * 1024 / 4), neg.data + 2, 8);
	}
	err = cudaMemcpy(Kparams.JmpDists12, jd12, JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy JmpDists12 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(jd12);
//jmp1
	u64* buf = (u64*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps1[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps1[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps1[i].dist.data, 32);
	}
	err = cudaMemcpy(Kparams.Jumps1, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps1 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(buf);
//jmp2
	buf = (u64*)malloc(JMP_CNT * 96);
	u64* jmp2_table = (u64*)malloc(JMP_CNT * 64);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps2[i].p.x.data, 32);
		memcpy(jmp2_table + i * 8, EcJumps2[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps2[i].p.y.data, 32);
		memcpy(jmp2_table + i * 8 + 4, EcJumps2[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps2[i].dist.data, 32);
	}
	err = cudaMemcpy(Kparams.Jumps2, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps2 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(buf);

	err = cuSetGpuParams(Kparams, jmp2_table);
	if (err != cudaSuccess)
	{
		free(jmp2_table);
		printf("GPU %d, cuSetGpuParams failed: %s!\r\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(jmp2_table);
//jmp3
	buf = (u64*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps3[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps3[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps3[i].dist.data, 32);
	}
	err = cudaMemcpy(Kparams.Jumps3, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps3 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(buf);

	printf("GPU %d: allocated %llu MB, %d kangaroos.\r\n", CudaIndex, total_mem / (1024 * 1024), KangCnt);
	return true;
}

void RCGpuKang::Release()
{
	free(RndPnts);
	free(DPs_out);
	cudaFree(Kparams.LoopedKangs);
	cudaFree(Kparams.dbg_buf);
	cudaFree(Kparams.LoopTable);
	cudaFree(Kparams.LastPnts);
	cudaFree(Kparams.L1S2);
	cudaFree(Kparams.DPTable);
	cudaFree(Kparams.JumpsList);
	cudaFree(Kparams.Jumps3);
	cudaFree(Kparams.Jumps2);
	cudaFree(Kparams.Jumps1);
	cudaFree(Kparams.DPs_out);
	cudaFree(Kparams.L2);
	cudaFree(Kparams.dists);
	cudaFree(Kparams.Jumps12);
	cudaFree(Kparams.JmpDists12);
}

void RCGpuKang::Stop()
{
	StopFlag = true;
}

void RCGpuKang::DoRestartKangs()
{
	cr.Enter();
	if (lsToRestart.empty())
	{
		cr.Leave();
		return;
	}

	EcInt WildRange, x32;
	x32.Set(1);
	x32.ShiftLeft(Range - 5);
	WildRange.Set(0);
	for (int i = 0; i < 32 + 2; i++) // +2 to smooth edges, PntToSolve must be RealPntToSolve+x32
		WildRange.Add(x32);

	cudaError_t err;
	u64 t0 = GetTickCount64();
	for (int i = 0; i < (int)lsToRestart.size(); i++)
	{
		int KangInd = lsToRestart[i];

		EcInt d;
		if (KangInd < KangCnt / 3)
			d.RndMax(x32); //TAME kangs
		else
		{
			d.RndMax(WildRange);
			d.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		}
		memcpy(RndPnts[KangInd].priv, d.data, 32);

#ifdef DEBUG_MODE
		EcPoint pnt = ec.MultiplyG_Fast(d);
#else
		EcPoint pnt = ec.MultiplyG(d);
#endif
		if (KangInd >= KangCnt / 3)
			pnt = ec.AddPoints(pnt, PntWild);
		pnt.SaveToBuffer64((u8*)RndPnts[KangInd].x);

		////copy pnt to gpu
		err = cudaMemcpy(Kparams.L2 + 4 * KangInd, RndPnts[KangInd].x, 32, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
			cr.Leave();
			return;
		}
		err = cudaMemcpy(Kparams.L2 + 4 * KangCnt + 4 * KangInd, RndPnts[KangInd].y, 32, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
			cr.Leave();
			return;
		}
		err = cudaMemcpy(Kparams.dists + 4 * KangInd, RndPnts[KangInd].priv, 24, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
			cr.Leave();
			return;
		}
	}

	lsToRestart.clear();
	cr.Leave();
//	printf("DoRestart %d ms\r\n", GetTickCount64() - t0);
}

void RCGpuKang::GenerateRndDistances()
{
	EcInt WildRange, x32;
	x32.Set(1);
	x32.ShiftLeft(Range - 5);
	WildRange.Set(0);
	for (int i = 0; i < 32 + 2; i++) // +2 to smooth edges, PntToSolve must be RealPntToSolve+x32
		WildRange.Add(x32);

	for (int i = 0; i < KangCnt; i++)
	{
		EcInt d;
		if (i < KangCnt / 3)
			d.RndMax(x32); //TAME kangs
		else
		{
			d.RndMax(WildRange);
			d.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		}
		memcpy(RndPnts[i].priv, d.data, 24);
	}
}

bool RCGpuKang::Start()
{
	if (Failed)
		return false;

	cudaError_t err;
	err = cudaSetDevice(CudaIndex);
	if (err != cudaSuccess)
		return false;

	HalfRange.Set(1);
	HalfRange.ShiftLeft(Range - 1);
	PntHalfRange = ec.MultiplyG(HalfRange);
	NegPntHalfRange = PntHalfRange;
	NegPntHalfRange.y.NegModP();

	PntWild = PntToSolve; //to smooth edges PntToSolve = RealPnt+x32 (added in caller)
	PntWild.y.NegModP(); //negate

	RndPnts = (TPointPriv*)malloc(KangCnt * 96);
	GenerateRndDistances();
/* 
	//we can calc start points on CPU
	for (int i = 0; i < KangCnt; i++)
	{
		EcInt d;
		memcpy(d.data, RndPnts[i].priv, 24);
		d.data[3] = 0;
		d.data[4] = 0;
		EcPoint p = ec.MultiplyG(d);
		memcpy(RndPnts[i].x, p.x.data, 32);
		memcpy(RndPnts[i].y, p.y.data, 32);
	}
	for (int i = KangCnt / 3; i < KangCnt; i++)
	{
		EcPoint p;
		p.LoadFromBuffer64((u8*)RndPnts[i].x);
		p = ec.AddPoints(p, PntWild);
		p.SaveToBuffer64((u8*)RndPnts[i].x);
	}
	//copy to gpu
	err = cudaMemcpy(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
/**/
	//but it's faster to calc them on GPU
	u8 buf_PntWild[64];
	PntWild.SaveToBuffer64(buf_PntWild);
	for (int i = 0; i < KangCnt; i++)
	{
		if (i < KangCnt / 3)
			memset(RndPnts[i].x, 0, 64);
		else
			memcpy(RndPnts[i].x, buf_PntWild, 64);
	}

	u8* gpu_pnts = (u8*)malloc(96 * KangCnt);
	for (int i = 0; i < KangCnt; i++)
	{
		memcpy(gpu_pnts + 32 * i, RndPnts[i].x, 32);
		memcpy(gpu_pnts + 32 * i + 32 * KangCnt, RndPnts[i].y, 32);
		memcpy(gpu_pnts + 32 * i + 64 * KangCnt, RndPnts[i].priv, 32);
	}

	//copy to gpu
	err = cudaMemcpy(Kparams.L2, gpu_pnts, KangCnt * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		free(gpu_pnts);
		printf("GPU %d, cudaMemcpy gpu_pnts failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}	
	CallGpuKernelGen(Kparams);
	err = cudaMemcpy(Kparams.dists, gpu_pnts + 64 * KangCnt, KangCnt * 32, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(gpu_pnts);

	err = cudaMemset(Kparams.L1S2, 0, mpCnt * Kparams.BlockSize * 8);
	if (err != cudaSuccess)
		return false;
	cudaMemset(Kparams.dbg_buf, 0, 1024);
	cudaMemset(Kparams.LoopTable, 0, KangCnt * MD_LEN * sizeof(u64));
	return true;
}

#ifdef DEBUG_MODE
int RCGpuKang::Dbg_CheckKangs()
{
	u32 PartStride = PNT_GROUP_CNT * (Kparams.BlockCnt * 256 * 32);
	u64* kangs = (u64*)malloc(Kparams.KangCnt * 64);
	u64* dists = (u64*)malloc(Kparams.KangCnt * 32);
	cudaError_t err = cudaMemcpy(kangs, Kparams.L2, Kparams.KangCnt * 64, cudaMemcpyDeviceToHost);
	err = cudaMemcpy(dists, Kparams.dists, Kparams.KangCnt * 32, cudaMemcpyDeviceToHost);
	int res = 0;
	for (int i = 0; i < KangCnt; i++)
	{
		EcPoint Pnt, p;
		memcpy(Pnt.x.data, &kangs[i * 4 + 0], 32);
		memcpy(Pnt.y.data, &kangs[i * 4 + 0 + PartStride / 8], 32);

		EcInt dist;
		dist.Set(0);
		memcpy(dist.data, &dists[i * 4], 24);
		bool neg = false;
		if (dist.data[2] >> 63)
		{
			neg = true;
			memset(((u8*)dist.data) + 24, 0xFF, 16);
			dist.Neg();
		}
		p = ec.MultiplyG_Fast(dist);
		if (neg)
			p.y.NegModP();
		if (i < KangCnt / 3)
			p = p;
		else
			p = ec.AddPoints(PntWild, p);
		if (!p.IsEqual(Pnt))
			res++;
	}
	free(kangs);
	free(dists);
	return res;
}

#endif

extern u32 gTotalErrors;

//executes in separate thread
void RCGpuKang::Execute()
{
	cudaSetDevice(CudaIndex);

	if (!Start())
	{
		gTotalErrors++;
		return;
	}
#ifdef DEBUG_MODE
	u64 iter = 1;
#endif
	cudaError_t err;	
	while (!StopFlag)
	{
		u64 t1 = GetTickCount64();
		cudaMemset(Kparams.DPs_out, 0, 4);
		cudaMemset(Kparams.DPTable, 0, KangCnt * sizeof(u32));
		cudaMemset(Kparams.LoopedKangs, 0, 8);

		if (sm_inv_cnt) //use turbo asm kernels
			Asm_CallGpuKernelAB();
		else
		{
			CallGpuKernelA(Kparams);
			CallGpuKernelB(Kparams);
		}

		CallGpuKernelC(Kparams);

		int cnt;
		err = cudaMemcpy(&cnt, Kparams.DPs_out, 4, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			printf("GPU %d, CallGpuKernel failed: %s\r\n", CudaIndex, cudaGetErrorString(err));
			gTotalErrors++;
			break;
		}
		
		if (cnt >= MAX_DP_CNT)
		{
			cnt = MAX_DP_CNT;
			printf("GPU %d, gpu DP buffer overflow, some points lost, increase DP value!\r\n", CudaIndex);
		}
		u64 pnt_cnt = (u64)KangCnt * STEP_CNT;

		if (cnt)
		{
			err = cudaMemcpy(DPs_out, Kparams.DPs_out + 4, cnt * GPU_DP_SIZE, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess)
			{
				gTotalErrors++;
				break;
			}
			AddPointsToList(DPs_out, cnt, KangCnt, (u64)KangCnt * STEP_CNT, JumperInd);
		}

		//dbg
		cudaMemcpy(dbg, Kparams.dbg_buf, 1024, cudaMemcpyDeviceToHost);

		u32 lcnt;
		cudaMemcpy(&lcnt, Kparams.LoopedKangs, 4, cudaMemcpyDeviceToHost);
		//printf("GPU %d, Looped: %d\r\n", CudaIndex, lcnt);

		DoRestartKangs();

		u64 t2 = GetTickCount64();
		u64 tm = t2 - t1;
		if (!tm)
			tm = 1;
		int cur_speed = (int)(pnt_cnt / (tm * 1000));
		//printf("GPU %d kernel time %d ms, speed %d MH\r\n", CudaIndex, (int)tm, cur_speed);

		SpeedStats[cur_stats_ind] = cur_speed;
		cur_stats_ind = (cur_stats_ind + 1) % STATS_WND_SIZE;

#ifdef DEBUG_MODE
		if ((iter % 300) == 0)
		{
			int corr_cnt = Dbg_CheckKangs();
			if (corr_cnt)
			{
				printf("DBG: GPU %d, KANGS CORRUPTED: %d\r\n", CudaIndex, corr_cnt);
				gTotalErrors++;
			}
			else
				printf("DBG: GPU %d, ALL KANGS OK!\r\n", CudaIndex);
		}
		iter++;
#endif

		
	}

	Release();
}

void RCGpuKang::ToRestartKangaroo(int KangInd)
{
	cr.Enter();
	lsToRestart.push_back(KangInd);
	cr.Leave();
}

int RCGpuKang::GetStatsSpeed()
{
	int res = SpeedStats[0];
	for (int i = 1; i < STATS_WND_SIZE; i++)
		res += SpeedStats[i];
	return res / STATS_WND_SIZE;
}

void RCGpuKang::Asm_CallGpuKernelAB()
{
	TCallKernelParams kp;

	strcpy(kp.kernel_name, "KernelA");
	kp.blockSize = Kparams.BlockSize;
	kp.blockCnt = Kparams.BlockCnt + sm_inv_cnt;
	kp.stream = NULL;
	kp.kernel_param_ptr = &Kparams;
	kp.kernel_param_size = sizeof(Kparams);
	kp.sharedSize = Kparams.KernelA_LDS_Size;
	cudaMemset(((u8*)Kparams.L2) + 96 * Kparams.KangCnt, 0, Inv_DataSize);
	if (!cc.CallKernel(kp))
		fprintf(stderr, "KernelA failed!");

	strcpy(kp.kernel_name, "KernelB");
	kp.blockCnt = Kparams.BlockCnt;
	kp.sharedSize = Kparams.KernelB_LDS_Size;
	if (!cc.CallKernel(kp))
		fprintf(stderr, "KernelB failed!");
}