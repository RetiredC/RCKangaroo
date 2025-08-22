#!/usr/bin/env bash
set -euo pipefail

SM="${1:-86}"
USE_JACOBIAN="${2:-1}"
PROFILE="${3:-release}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-12.0}"

HOST_COPT_release="-O3 -DNDEBUG -ffunction-sections -fdata-sections"
HOST_COPT_debug="-O0 -g"
DEV_COPT_release="-O3"
DEV_COPT_debug="-O0 -g"

if [[ "$PROFILE" == "release" ]]; then 
  HOST_COPT="$HOST_COPT_release"
  DEV_COPT="$DEV_COPT_release"
else 
  HOST_COPT="$HOST_COPT_debug"
  DEV_COPT="$DEV_COPT_debug"
fi

CCFLAGS="-std=c++17 -I${CUDA_PATH}/include ${HOST_COPT} -DUSE_JACOBIAN=${USE_JACOBIAN}"
NVCCFLAGS="-std=c++17 -arch=sm_${SM} ${DEV_COPT} -Xptxas -O3 -Xptxas -dlcm=ca -Xfatbin=-compress-all -DUSE_JACOBIAN=${USE_JACOBIAN}"
NVCCXCOMP="-Xcompiler -ffunction-sections -Xcompiler -fdata-sections"
LDFLAGS="-L${CUDA_PATH}/lib64 -lcudart -pthread"

echo "== CCFLAGS:   ${CCFLAGS}"
echo "== NVCCFLAGS: ${NVCCFLAGS} ${NVCCXCOMP}"

# Compile C++
g++ ${CCFLAGS} -c RCKangaroo.cpp -o RCKangaroo.o
g++ ${CCFLAGS} -c GpuKang.cpp    -o GpuKang.o
g++ ${CCFLAGS} -c Ec.cpp         -o Ec.o
g++ ${CCFLAGS} -c utils.cpp      -o utils.o

# Compile CUDA (if present)
if [[ -f "RCGpuCore.cu" ]]; then
  /usr/bin/nvcc ${NVCCFLAGS} ${NVCCXCOMP} -c RCGpuCore.cu -o RCGpuCore.o
  g++ ${CCFLAGS} -o rckangaroo RCKangaroo.o GpuKang.o Ec.o utils.o RCGpuCore.o ${LDFLAGS}
else
  echo "WARN: RCGpuCore.cu no existe; enlazando CPU-only"
  g++ ${CCFLAGS} -o rckangaroo RCKangaroo.o GpuKang.o Ec.o utils.o ${LDFLAGS}
fi
echo "== Listo: ./rckangaroo"
