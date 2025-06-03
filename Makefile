# Makefile for RCKangaroo (OpenCL version)

CC := g++
# NVCC := /usr/local/cuda-12.0/bin/nvcc # Not used for OpenCL device code compilation by make
# CUDA_PATH ?= /usr/local/cuda-12.0 # Not used for OpenCL build

# Add -DUSE_OPENCL to enable OpenCL code paths
# Add -Wall for more warnings, -std=c++11 or newer if needed by C++ code
CCFLAGS := -O3 -Wall -DUSE_OPENCL -std=c++11 # Assuming C++11 for std::to_string etc.
# NVCCFLAGS := -O3 ... # Not used

# Link with OpenCL library
LDFLAGS := -lOpenCL -pthread

CPU_SRC := RCKangaroo.cpp GpuKang.cpp Ec.cpp utils.cpp
# GPU_SRC is not applicable here as OpenCL kernels are compiled at runtime
# GPU_SRC :=
GPU_OBJECTS := # Was CU_OBJECTS

CPP_OBJECTS := $(CPU_SRC:.cpp=.o)

TARGET := rckangaroo_opencl

all: $(TARGET)

$(TARGET): $(CPP_OBJECTS) $(GPU_OBJECTS)
	$(CC) $(CCFLAGS) -o $@ $(CPP_OBJECTS) $(LDFLAGS) # Removed $(GPU_OBJECTS) from linking line

%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

# Rule for .cu files is not needed for a pure OpenCL build
# %.o: %.cu
#	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(CPP_OBJECTS) $(GPU_OBJECTS) $(TARGET)
	rm -f rckangaroo # Remove old CUDA target too, just in case
