// This is the OpenCL initialization block for RCGpuKang::Prepare
// It should be placed inside the #ifdef USE_OPENCL block in RCGpuKang::Prepare

// Standard C++ headers for file operations, vectors, strings
#include <vector>
#include <fstream>
#include <sstream>
#include <string> // For std::to_string with compile options
#include <cstring> // For memcpy (host buffers to device)

// OpenCL Headers
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Helper function to load kernel source (should be available or defined in GpuKang.cpp)
// static std::string oclLoadKernelSource(const char* filename) { ... }
// (Assuming it's defined elsewhere or will be added to GpuKang.cpp)

// --- Start of OpenCL Initialization Logic ---
PntToSolve = _PntToSolve;
Range = _Range;
DP = _DP;
EcJumps1 = _EcJumps1;
EcJumps2 = _EcJumps2;
EcJumps3 = _EcJumps3;
StopFlag = false;
Failed = false;
u64 total_mem_ocl = 0;
memset(dbg, 0, sizeof(dbg));
memset(SpeedStats, 0, sizeof(SpeedStats));
cur_stats_ind = 0;

cl_int ret; // OpenCL error code

// 1. Platform and Device Setup
cl_uint num_platforms;
ret = clGetPlatformIDs(0, NULL, &num_platforms);
if (ret != CL_SUCCESS || num_platforms == 0) {
    printf("GPU %d (OCL): clGetPlatformIDs failed or no platforms found: %d\n", CudaIndex, ret);
    return false;
}
std::vector<cl_platform_id> platforms(num_platforms);
ret = clGetPlatformIDs(num_platforms, platforms.data(), NULL);
if (ret != CL_SUCCESS) {
    printf("GPU %d (OCL): clGetPlatformIDs failed to get platform data: %d\n", CudaIndex, ret);
    return false;
}
platform_id = platforms[0]; // Using the first platform by default

cl_uint num_devices;
ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
if (ret != CL_SUCCESS || num_devices == 0) {
    printf("GPU %d (OCL): No GPU devices found, trying CL_DEVICE_TYPE_DEFAULT. Error: %d\n", CudaIndex, ret);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
    if (ret != CL_SUCCESS || num_devices == 0) { // num_devices check was from GPU path, should be for default here
         printf("GPU %d (OCL): clGetDeviceIDs for CL_DEVICE_TYPE_DEFAULT also failed: %d\n", CudaIndex, ret);
        return false;
    }
} else {
    std::vector<cl_device_id> devices(num_devices);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), NULL);
     if (ret != CL_SUCCESS) {
        printf("GPU %d (OCL): clGetDeviceIDs failed to get device data: %d\n", CudaIndex, ret);
        return false;
    }
    device_id = devices[0]; // Using the first GPU device by default
}
// TODO: Add logic to select specific device if CudaIndex > 0 and multiple OpenCL devices exist.

// 2. Context Creation
context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
if (!context || ret != CL_SUCCESS) {
    printf("GPU %d (OCL): clCreateContext failed: %d\n", CudaIndex, ret);
    return false;
}

// 3. Command Queue Creation
#ifdef CL_VERSION_2_0
command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
#else
// Deprecated in OpenCL 2.0, but required for OpenCL 1.x
command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
#endif
if (!command_queue || ret != CL_SUCCESS) {
    printf("GPU %d (OCL): clCreateCommandQueue failed: %d\n", CudaIndex, ret);
    return false;
}

// 4. Program Creation and Build
// Assuming oclLoadKernelSource is defined in GpuKang.cpp or a utility header
std::string kernel_source = oclLoadKernelSource("OCLGpuCore.cl");
if (kernel_source.empty()) {
    printf("GPU %d (OCL): Cannot open/read OCLGpuCore.cl for program creation.\n", CudaIndex);
    return false;
}
const char* source_str = kernel_source.c_str();
size_t source_size = kernel_source.length();

program = clCreateProgramWithSource(context, 1, &source_str, &source_size, &ret);
if (!program || ret != CL_SUCCESS) {
    printf("GPU %d (OCL): clCreateProgramWithSource failed: %d\n", CudaIndex, ret);
    return false;
}

std::string compile_options = "-I. "; // Include current directory for OCLGpuUtils.h
if (IsOldGpu) { // IsOldGpu is a member of RCGpuKang
    compile_options += "-DOLD_GPU ";
}
// Pass PNT_GROUP_CNT_KERNEL and BLOCK_SIZE_KERNEL as defines for clarity,
// matching how they are used in OCLGpuCore.cl
compile_options += "-DPNT_GROUP_CNT_KERNEL=" + std::to_string(IsOldGpu ? 64 : 24) + " ";
compile_options += "-DBLOCK_SIZE_KERNEL=" + std::to_string(IsOldGpu ? 512 : 256) + " ";


ret = clBuildProgram(program, 1, &device_id, compile_options.c_str(), NULL, NULL);
if (ret != CL_SUCCESS) {
    printf("GPU %d (OCL): clBuildProgram failed: %d\n", CudaIndex, ret);
    size_t log_size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    if (log_size > 1) {
        std::vector<char> log_buffer(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log_buffer.data(), NULL);
        printf("Build Log:\n%s\n", log_buffer.data());
    }
    return false;
}

// 5. Kernel Creation
const char* kernel_a_name = IsOldGpu ? "KernelA_oldgpu" : "KernelA_main";
kernel_A = clCreateKernel(program, kernel_a_name, &ret);
if (!kernel_A || ret != CL_SUCCESS) { printf("GPU %d (OCL): clCreateKernel %s failed: %d\n", CudaIndex, kernel_a_name, ret); return false; }
kernel_B = clCreateKernel(program, "KernelB_main", &ret);
if (!kernel_B || ret != CL_SUCCESS) { printf("GPU %d (OCL): clCreateKernel KernelB_main failed: %d\n", CudaIndex, ret); return false; }
kernel_C = clCreateKernel(program, "KernelC_main", &ret);
if (!kernel_C || ret != CL_SUCCESS) { printf("GPU %d (OCL): clCreateKernel KernelC_main failed: %d\n", CudaIndex, ret); return false; }
kernel_Gen = clCreateKernel(program, "KernelGen_main", &ret);
if (!kernel_Gen || ret != CL_SUCCESS) { printf("GPU %d (OCL): clCreateKernel KernelGen_main failed: %d\n", CudaIndex, ret); return false; }

// Populate Kparams (host-side struct) with sizes and config values for OpenCL
// These values are used by the host to manage buffers and enqueue kernels.
// The Kparams struct itself is not directly copied to device in OpenCL.
// Kernel arguments are set individually.
Kparams.BlockCnt = mpCnt;
Kparams.BlockSize = IsOldGpu ? 512 : 256;
Kparams.GroupCnt = IsOldGpu ? 64 : 24; // This is PNT_GROUP_CNT
KangCnt = Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
Kparams.KangCnt = KangCnt;
Kparams.DP = DP;
Kparams.IsGenMode = gGenMode;
// Kparams.KernelA_LDS_Size etc. are not used directly by OpenCL kernels in the same way.
// Local memory is an argument to clEnqueueNDRangeKernel or sized in kernel.

// 6. Memory Allocation (Buffer Creation)
u64 ocl_buf_size; // For clCreateBuffer size argument

// L2 (if used by non-OLD_GPU KernelA)
if (!IsOldGpu) {
    ocl_buf_size = (u64)Kparams.KangCnt * (3 * 32);
    total_mem_ocl += ocl_buf_size;
    d_L2_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, ocl_buf_size, NULL, &ret);
    if (!d_L2_ocl || ret != CL_SUCCESS) { printf("GPU %d (OCL): Buffer d_L2_ocl failed: %d\n", CudaIndex, ret); return false; }
}

ocl_buf_size = (u64)MAX_DP_CNT * GPU_DP_SIZE + 16; total_mem_ocl += ocl_buf_size;
d_DPs_out_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, ocl_buf_size, NULL, &ret); if (!d_DPs_out_ocl || ret != CL_SUCCESS) { printf("GPU %d (OCL): Buffer d_DPs_out_ocl failed: %d\n", CudaIndex, ret); return false; }

ocl_buf_size = (u64)KangCnt * 96; total_mem_ocl += ocl_buf_size;
d_Kangs_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, ocl_buf_size, NULL, &ret); if (!d_Kangs_ocl || ret != CL_SUCCESS) { printf("GPU %d (OCL): Buffer d_Kangs_ocl failed: %d\n", CudaIndex, ret); return false; }

// Temporary host buffer for Jumps data (x, y, dist)
// Each Jumps entry is 12 u64s (96 bytes) = x[4], y[4], d[3] (actual) + padding for d if needed.
ocl_buf_size = (u64)JMP_CNT * 96;
u64* host_jumps_buf = (u64*)malloc(ocl_buf_size);
if (!host_jumps_buf) { printf("GPU %d (OCL): Malloc host_jumps_buf failed\n", CudaIndex); return false;}

// Jumps1
for (int j = 0; j < JMP_CNT; j++) { memcpy(host_jumps_buf + j * 12, EcJumps1[j].p.x.data, 32); memcpy(host_jumps_buf + j * 12 + 4, EcJumps1[j].p.y.data, 32); memcpy(host_jumps_buf + j * 12 + 8, EcJumps1[j].dist.data, 24); if(12*j+11 < JMP_CNT*12) memset(host_jumps_buf+j*12+11,0,8); /*Pad last u64 of dist if necessary*/ }
d_Jumps1_ocl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ocl_buf_size, host_jumps_buf, &ret); if (!d_Jumps1_ocl || ret != CL_SUCCESS) { free(host_jumps_buf); printf("GPU %d (OCL): Buffer d_Jumps1_ocl failed: %d\n", CudaIndex, ret); return false; }
total_mem_ocl += ocl_buf_size;

// Jumps2 and jmp2_table (constant table for KernelA)
u64* host_jmp2_table_xy = (u64*)malloc(JMP_CNT * 64); // 8 ulongs (x,y coords only) per entry
if (!host_jmp2_table_xy) { free(host_jumps_buf); printf("GPU %d (OCL): Malloc host_jmp2_table_xy failed\n", CudaIndex); return false;}
for (int j = 0; j < JMP_CNT; j++) { memcpy(host_jumps_buf + j * 12, EcJumps2[j].p.x.data, 32); memcpy(host_jmp2_table_xy + j * 8, EcJumps2[j].p.x.data, 32); memcpy(host_jumps_buf + j * 12 + 4, EcJumps2[j].p.y.data, 32); memcpy(host_jmp2_table_xy + j * 8 + 4, EcJumps2[j].p.y.data, 32); memcpy(host_jumps_buf + j * 12 + 8, EcJumps2[j].dist.data, 24); if(12*j+11 < JMP_CNT*12) memset(host_jumps_buf+j*12+11,0,8); }
d_Jumps2_ocl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ocl_buf_size, host_jumps_buf, &ret); if (!d_Jumps2_ocl || ret != CL_SUCCESS) { free(host_jumps_buf); free(host_jmp2_table_xy); printf("GPU %d (OCL): Buffer d_Jumps2_ocl failed: %d\n", CudaIndex, ret); return false; }
total_mem_ocl += ocl_buf_size;
d_jmp2_table_ocl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, JMP_CNT * 64, host_jmp2_table_xy, &ret); if (!d_jmp2_table_ocl || ret != CL_SUCCESS) { free(host_jumps_buf); free(host_jmp2_table_xy); printf("GPU %d (OCL): Buffer d_jmp2_table_ocl failed: %d\n", CudaIndex, ret); return false; }
total_mem_ocl += (u64)JMP_CNT * 64;
free(host_jmp2_table_xy);

// Jumps3
for (int j = 0; j < JMP_CNT; j++) { memcpy(host_jumps_buf + j * 12, EcJumps3[j].p.x.data, 32); memcpy(host_jumps_buf + j * 12 + 4, EcJumps3[j].p.y.data, 32); memcpy(host_jumps_buf + j * 12 + 8, EcJumps3[j].dist.data, 24); if(12*j+11 < JMP_CNT*12) memset(host_jumps_buf+j*12+11,0,8); }
d_Jumps3_ocl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ocl_buf_size, host_jumps_buf, &ret); if (!d_Jumps3_ocl || ret != CL_SUCCESS) { free(host_jumps_buf); printf("GPU %d (OCL): Buffer d_Jumps3_ocl failed: %d\n", CudaIndex, ret); return false; }
total_mem_ocl += ocl_buf_size;
free(host_jumps_buf);

ocl_buf_size = 2 * (u64)KangCnt * STEP_CNT; total_mem_ocl += ocl_buf_size;
d_JumpsList_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, ocl_buf_size, NULL, &ret); if (!d_JumpsList_ocl || ret != CL_SUCCESS) { printf("GPU %d (OCL): Buffer d_JumpsList_ocl failed: %d\n", CudaIndex, ret); return false; }

ocl_buf_size = (u64)KangCnt * (16 * DPTABLE_MAX_CNT + sizeof(cl_uint)); total_mem_ocl += ocl_buf_size; // KangCnt cl_uint counters + data
d_DPTable_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, ocl_buf_size, NULL, &ret); if (!d_DPTable_ocl || ret != CL_SUCCESS) { printf("GPU %d (OCL): Buffer d_DPTable_ocl failed: %d\n", CudaIndex, ret); return false; }

ocl_buf_size = (u64)mpCnt * Kparams.BlockSize * sizeof(cl_ulong); total_mem_ocl += ocl_buf_size; // L1S2 uses ulong in TKparams_ocl
d_L1S2_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, ocl_buf_size, NULL, &ret); if (!d_L1S2_ocl || ret != CL_SUCCESS) { printf("GPU %d (OCL): Buffer d_L1S2_ocl failed: %d\n", CudaIndex, ret); return false; }

ocl_buf_size = (u64)KangCnt * MD_LEN * 2 * 4 * sizeof(cl_ulong);  total_mem_ocl += ocl_buf_size; // KangCnt * MD_LEN * (2 points * 4 ulongs/point)
d_LastPnts_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, ocl_buf_size, NULL, &ret); if (!d_LastPnts_ocl || ret != CL_SUCCESS) { printf("GPU %d (OCL): Buffer d_LastPnts_ocl failed: %d\n", CudaIndex, ret); return false; }

ocl_buf_size = (u64)KangCnt * MD_LEN * sizeof(cl_ulong); total_mem_ocl += ocl_buf_size;
d_LoopTable_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, ocl_buf_size, NULL, &ret); if (!d_LoopTable_ocl || ret != CL_SUCCESS) { printf("GPU %d (OCL): Buffer d_LoopTable_ocl failed: %d\n", CudaIndex, ret); return false; }

ocl_buf_size = 1024; total_mem_ocl += ocl_buf_size; // Matched to CUDA allocation
d_dbg_buf_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, ocl_buf_size, NULL, &ret); if (!d_dbg_buf_ocl || ret != CL_SUCCESS) { printf("GPU %d (OCL): Buffer d_dbg_buf_ocl failed: %d\n", CudaIndex, ret); return false; }

ocl_buf_size = (2 + KangCnt) * sizeof(cl_uint); total_mem_ocl += ocl_buf_size; // 2 global counters + KangCnt data elements
d_LoopedKangs_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, ocl_buf_size, NULL, &ret); if (!d_LoopedKangs_ocl || ret != CL_SUCCESS) { printf("GPU %d (OCL): Buffer d_LoopedKangs_ocl failed: %d\n", CudaIndex, ret); return false; }

DPs_out = (u32*)malloc(MAX_DP_CNT * GPU_DP_SIZE); // Host buffer for DPs_out results
if (!DPs_out) { printf("GPU %d (OCL): Malloc DPs_out (host) failed\n", CudaIndex); return false; }


printf("GPU %d (OpenCL): allocated %llu MB, %d kangaroos. OldGpuMode: %s\r\n", CudaIndex, total_mem_ocl / (1024 * 1024), KangCnt, IsOldGpu ? "Yes" : "No");
return true;
// --- End of OpenCL Initialization Logic ---
