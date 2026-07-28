// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC

#pragma once

#include "cuda_runtime.h"
#include "cuda.h"

struct TCallKernelParams
{
	char kernel_name[256];
	int blockSize;
	int blockCnt;
	int sharedSize; //LDS size
	cudaStream_t stream;
	void* kernel_param_ptr;
	int kernel_param_size;
};

class TCubinCall
{
private:
	CUmodule cuModule;

public:
	TCubinCall();
	~TCubinCall();

	bool LoadCubin(const char* fn);
	bool CallKernel(TCallKernelParams params);
	bool CopyToSymbol(const char* sym_name, void* data, int size);
};

