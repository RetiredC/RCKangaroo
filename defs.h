// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once 

#pragma warning(disable : 4996)

typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef int i32;
typedef unsigned short u16;
typedef short i16;
typedef unsigned char u8;
typedef char i8;



#define MAX_GPU_CNT			32

//must be divisible by MD_LEN
#define STEP_CNT			1000

#define JMP_CNT				512

#define BLOCK_SIZE			256	
#define PNT_GROUP_CNT		24

// kang type
#define TAME				0  // Tame kangs
#define WILD				1  // Wild kangs 

#define GPU_DP_SIZE			48
#define MAX_DP_CNT			(256 * 1024)

#define JMP_MASK			(JMP_CNT-1)
#define JMP_MASK_ADV		(2048 - 1) //including INV_FLAG and JMP2_FLAG

#define DPTABLE_MAX_CNT		16

#define MAX_CNT_LIST		(512 * 1024)

#define DP_FLAG				0x0800
#define INV_FLAG			0x0200
#define JMP2_FLAG			0x0400

#define MD_LEN				10

//#define DEBUG_MODE

//gpu kernel parameters
struct TKparams
{
	u64* L2;
	u32* Jumps12;
	u32* DPTable;
	u32* Reserved1;
	u64* JumpsList;
	u64* LastPnts;
	u32* dbg_buf;
	u32* L1S2;
	u64* Reserved2;

	u32 iter_cnt;
	u32 BlockCnt;
	u32 StopThr;

	u32 dp_mask;
	///////////////////////////////////////////
	u32* DPs_out;
	u64* LoopTable;
	u32* LoopedKangs;
	u64* dists;
	u64* JmpDists12;
	u32 KangCnt;
	u64* Jumps1;
	u64* Jumps2;
	u64* Jumps3;
	u32 BlockSize;
	u32 GroupCnt;
	u64 DP;
	bool IsGenMode; //tames generation mode
	u32 KernelA_LDS_Size;
	u32 KernelB_LDS_Size;
	u32 KernelC_LDS_Size;
};

