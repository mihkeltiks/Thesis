# NVIDIA H100 (compute_90)

CUDA_ARCH_FLAGS ?= -gencode arch=compute_90,code=sm_90

OPTIONS  =
OPTIONS +=  -DSIZE=32768
OPTIONS +=  -DNTIMES=300
OPTIONS +=  -DMETRICS
# OPTIONS +=  -DTENSOR

# ---------------------------------------------------------------------------
# Default collective buffer size (override with -m at runtime)
# ---------------------------------------------------------------------------
OPTIONS +=  -DCOMM_MSG_SIZE=33554432   # 32 MB — typical gradient AllReduce chunk
