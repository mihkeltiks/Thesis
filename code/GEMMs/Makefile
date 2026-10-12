include config.mk
include mk/include_$(GPU).mk

NVCC      = nvcc
CC_FLAGS += -O3 --std=c++17 $(CUDA_ARCH_FLAGS) $(OPTIONS)
CUDA_LIBS = -lcublas -lnvidia-ml -lcuda

# Rpath so -ccbin mpicxx links against the toolkit's cuBLAS, not the
# compiler wrapper's mismatched math libs.
CUDA_HOME  ?= $(shell dirname $$(dirname $$(command -v nvcc)))
RPATH_CUDA = -L$(CUDA_HOME)/lib64 -Xlinker -rpath,$(CUDA_HOME)/lib64

# NCCL location: picked up from the module env (NCCL_HOME / EBROOTNCCL),
# else assumed to live with the CUDA toolkit. The library dir is lib64 or lib.
NCCL_HOME ?= $(or $(EBROOTNCCL),$(CUDA_HOME))
NCCL_LIB  := $(if $(wildcard $(NCCL_HOME)/lib64/libnccl.so*),lib64,lib)
NCCL_LINK  = -I$(NCCL_HOME)/include -L$(NCCL_HOME)/$(NCCL_LIB) -Xlinker -rpath,$(NCCL_HOME)/$(NCCL_LIB)

# PRECISION in config.mk only affects the base targets; the per-datatype
# targets below set the datatype explicitly, so strip any -DDOUBLE it added.
ifeq ($(PRECISION), DP)
  CC_FLAGS += -DDOUBLE
endif
BASE_FLAGS = $(filter-out -DDOUBLE,$(CC_FLAGS))

DTYPES    = sp dp tf32 bf16
DTYPE_sp   =
DTYPE_dp   = -DDOUBLE
DTYPE_tf32 = -DTENSOR
DTYPE_bf16 = -DBFLOAT16

all: sweep sweep_nccl

gemm_sleep: gemm_sleep.cu
	$(NVCC) -DGEMM_SLEEP $(CC_FLAGS) $(CUDA_LIBS) -o $@ $<

gemm_comms_nccl: gemm_comms_nccl.cu
	$(NVCC) -DGEMM_COMMS -DGEMM_COMMS_NCCL -ccbin mpicxx $(CC_FLAGS) $(NCCL_LINK) $(RPATH_CUDA) $(CUDA_LIBS) -lnccl -lmpi -o $@ $<

gemm_sleep_%: gemm_sleep.cu
	$(NVCC) -DGEMM_SLEEP -DMETRICS $(DTYPE_$*) $(BASE_FLAGS) $(CUDA_LIBS) -o $@ $<

gemm_comms_nccl_%: gemm_comms_nccl.cu
	$(NVCC) -DGEMM_COMMS -DGEMM_COMMS_NCCL -DMETRICS -ccbin mpicxx $(DTYPE_$*) $(BASE_FLAGS) $(NCCL_LINK) $(RPATH_CUDA) $(CUDA_LIBS) -lnccl -lmpi -o $@ $<

sweep:     $(DTYPES:%=gemm_sleep_%)
sweep_nccl: $(DTYPES:%=gemm_comms_nccl_%)

clean:
	rm -f gemm_sleep gemm_comms_nccl $(DTYPES:%=gemm_sleep_%) $(DTYPES:%=gemm_comms_nccl_%)

.PHONY: all sweep sweep_nccl clean
