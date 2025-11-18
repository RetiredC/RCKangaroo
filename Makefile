# RCKangaroo Makefile (robusto y autodetecta CUDA)
# Uso:
#   make clean
#   make SM=86 USE_JACOBIAN=1 PROFILE=release -j
#   ./rckangaroo -h

TARGET := rckangaroo

# Toolchains
CC    := g++
NVCC  := /usr/bin/nvcc

# CUDA
CUDA_PATH ?= /usr/local/cuda-12.0
SM        ?= 86
USE_JACOBIAN ?= 1
PROFILE   ?= release

# Optimización separada: host vs device
HOST_COPT_release := -O3 -DNDEBUG -ffunction-sections -fdata-sections
HOST_COPT_debug   := -O0 -g
HOST_COPT := $(HOST_COPT_$(PROFILE))

DEV_COPT_release := -O3
DEV_COPT_debug   := -O0 -g
DEV_COPT := $(DEV_COPT_$(PROFILE))

# Flags
CCFLAGS    := -std=c++17 -I$(CUDA_PATH)/include $(HOST_COPT) -DUSE_JACOBIAN=$(USE_JACOBIAN)
NVCCFLAGS  := -std=c++17 -arch=sm_$(SM) $(DEV_COPT) -Xptxas -O3 -Xptxas -dlcm=ca -Xfatbin=-compress-all -DUSE_JACOBIAN=$(USE_JACOBIAN)
NVCCXCOMP  := -Xcompiler -ffunction-sections -Xcompiler -fdata-sections

LDFLAGS   := -L$(CUDA_PATH)/lib64 -lcudart -pthread

# Fuentes
SRC_CPP := RCKangaroo.cpp GpuKang.cpp Ec.cpp utils.cpp

# Directorio donde está el .cu (por defecto, raíz)
CU_DIR ?= .
SRC_CU := $(wildcard $(CU_DIR)/RCGpuCore.cu)

OBJ_CPP := $(SRC_CPP:.cpp=.o)
OBJ_CU  := $(patsubst %.cu,%.o,$(SRC_CU))

ifeq ($(strip $(OBJ_CU)),)
  $(warning [Makefile] No se encontró RCGpuCore.cu en $(CU_DIR). Se construirá solo CPU.)
  OBJS := $(OBJ_CPP)
else
  OBJS := $(OBJ_CPP) $(OBJ_CU)
endif

.PHONY: all clean print-vars

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CCFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

# Regla genérica CUDA (.cu -> .o) con flags host vía -Xcompiler
$(CU_DIR)/%.o: $(CU_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCXCOMP) -c $< -o $@

# Regla explícita (por si tu make ignora patrones)
$(CU_DIR)/RCGpuCore.o: $(CU_DIR)/RCGpuCore.cu RCGpuUtils.h Ec.h defs.h
	$(NVCC) $(NVCCFLAGS) $(NVCCXCOMP) -c $< -o $@

clean:
	rm -f $(OBJ_CPP) $(OBJ_CU) $(TARGET)

print-vars:
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "SM=$(SM)"
	@echo "USE_JACOBIAN=$(USE_JACOBIAN)"
	@echo "PROFILE=$(PROFILE)"
	@echo "SRC_CPP=$(SRC_CPP)"
	@echo "CU_DIR=$(CU_DIR)"
	@echo "SRC_CU=$(SRC_CU)"
	@echo "OBJ_CPP=$(OBJ_CPP)"
	@echo "OBJ_CU=$(OBJ_CU)"
	@echo "OBJS=$(OBJS)"
	@echo "NVCCFLAGS=$(NVCCFLAGS)"
	@echo "NVCCXCOMP=$(NVCCXCOMP)"
