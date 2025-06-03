// This is the OpenCL cleanup block for RCGpuKang::Release
// It should be placed inside the #ifdef USE_OPENCL block in RCGpuKang::Release

// Host allocated memory (RndPnts was malloc'd in Start(), DPs_out in Prepare())
if (RndPnts) {
    free(RndPnts);
    RndPnts = NULL;
}
if (DPs_out) {
    free(DPs_out);
    DPs_out = NULL;
}

// Release OpenCL memory objects
if(d_Kangs_ocl) { clReleaseMemObject(d_Kangs_ocl); d_Kangs_ocl = NULL; }
if(d_Jumps1_ocl) { clReleaseMemObject(d_Jumps1_ocl); d_Jumps1_ocl = NULL; }
if(d_Jumps2_ocl) { clReleaseMemObject(d_Jumps2_ocl); d_Jumps2_ocl = NULL; }
if(d_Jumps3_ocl) { clReleaseMemObject(d_Jumps3_ocl); d_Jumps3_ocl = NULL; }
if(d_DPTable_ocl) { clReleaseMemObject(d_DPTable_ocl); d_DPTable_ocl = NULL; }
if(d_DPs_out_ocl) { clReleaseMemObject(d_DPs_out_ocl); d_DPs_out_ocl = NULL; }
if(d_L1S2_ocl) { clReleaseMemObject(d_L1S2_ocl); d_L1S2_ocl = NULL; }
if(d_LoopTable_ocl) { clReleaseMemObject(d_LoopTable_ocl); d_LoopTable_ocl = NULL; }
if(d_JumpsList_ocl) { clReleaseMemObject(d_JumpsList_ocl); d_JumpsList_ocl = NULL; }
if(d_LastPnts_ocl) { clReleaseMemObject(d_LastPnts_ocl); d_LastPnts_ocl = NULL; }
if(d_LoopedKangs_ocl) { clReleaseMemObject(d_LoopedKangs_ocl); d_LoopedKangs_ocl = NULL; }
if(d_dbg_buf_ocl) { clReleaseMemObject(d_dbg_buf_ocl); d_dbg_buf_ocl = NULL; }
if(d_jmp2_table_ocl) { clReleaseMemObject(d_jmp2_table_ocl); d_jmp2_table_ocl = NULL; }
if(d_L2_ocl) { clReleaseMemObject(d_L2_ocl); d_L2_ocl = NULL; }

// Release OpenCL kernels
if(kernel_A) { clReleaseKernel(kernel_A); kernel_A = NULL; }
// kernel_A_oldgpu is not a separate member; kernel_A points to one or the other.
if(kernel_B) { clReleaseKernel(kernel_B); kernel_B = NULL; }
if(kernel_C) { clReleaseKernel(kernel_C); kernel_C = NULL; }
if(kernel_Gen) { clReleaseKernel(kernel_Gen); kernel_Gen = NULL; }

// Release OpenCL program
if(program) { clReleaseProgram(program); program = NULL; }

// Release OpenCL command queue
if(command_queue) { clReleaseCommandQueue(command_queue); command_queue = NULL; }

// Release OpenCL context
if(context) { clReleaseContext(context); context = NULL; }

// Platform and device IDs do not need to be released by clRelease calls.
platform_id = NULL;
device_id = NULL;
