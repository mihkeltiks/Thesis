#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include "cli.h"
#include "metrics.h"
#include <nccl.h>
#include <mpi.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include <time.h>
#include <fstream>
#include <filesystem>
#include <vector>
#include <tuple>
#include <string>

using namespace std;

#if defined(DOUBLE)
  typedef double         GEMM_FLOAT;
#elif defined(BFLOAT16)
  typedef __nv_bfloat16  GEMM_FLOAT;
#else
  typedef float          GEMM_FLOAT;
#endif

// For(--compute triad|eltwise): replaces the
// GEMM inside the per-lap event bracket. bf16, fp32 math, 128-bit accesses;
struct alignas(16) bf16x8 { __nv_bfloat162 v[4]; };   // 8 bf16 = one 128-bit access

__global__ void init_bf16_random(__nv_bfloat16 *X, size_t count, unsigned long long seed) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  curandState state;
  curand_init(seed, idx, 0, &state);
  for (; idx < count; idx += stride)
    X[idx] = __float2bfloat16(curand_uniform(&state) * 2.0f - 1.0f);
}

__global__ void triad_bf16_kernel(bf16x8 *__restrict__ a,
                                  const bf16x8 *__restrict__ b,
                                  const bf16x8 *__restrict__ c,
                                  size_t n8, float q) {
  const __nv_bfloat162 qq = __float2bfloat162_rn(q);
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n8; i += stride) {
    bf16x8 bb = b[i], cc = c[i], out;
#pragma unroll
    for (int j = 0; j < 4; j++) out.v[j] = __hfma2(qq, cc.v[j], bb.v[j]);
    a[i] = out;
  }
}

// Megatron-style fused bias + GeLU: read x, add a broadcast bias row, apply gelu, write y
__global__ void bias_gelu_bf16_kernel(bf16x8 *__restrict__ y,
                                      const bf16x8 *__restrict__ x,
                                      const __nv_bfloat162 *__restrict__ bias,
                                      size_t n8, unsigned bias_mask) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n8; i += stride) {
    bf16x8 xx = x[i], out;
    float e[8];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      __nv_bfloat162 bp = bias[(unsigned)(i * 4 + j) & bias_mask];
      float2 t = __bfloat1622float2(xx.v[j]);
      float2 bv = __bfloat1622float2(bp);
      e[2 * j] = t.x + bv.x;
      e[2 * j + 1] = t.y + bv.y;
    }
    // gelu_tanh(v) = 0.5v(1 + tanh(0.7978845608 * (v + 0.044715 v^3)))
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float e0 = e[2 * j], e1 = e[2 * j + 1];
      float g0 = 0.5f * e0 * (1.0f + tanhf(0.7978845608f * (e0 + 0.044715f * e0 * e0 * e0)));
      float g1 = 0.5f * e1 * (1.0f + tanhf(0.7978845608f * (e1 + 0.044715f * e1 * e1 * e1)));
      out.v[j] = __floats2bfloat162_rn(g0, g1);
    }
    y[i] = out;
  }
}

// Comm buffer type mirrors GEMM storage; NCCL dtype matches.
#if defined(DOUBLE)
  typedef double         COMM_FLOAT;
  #define NCCL_DTYPE     ncclDouble
  #define COMM_DTYPE_STR "fp64"
#elif defined(BFLOAT16)
  typedef __nv_bfloat16  COMM_FLOAT;
  #define NCCL_DTYPE     ncclBfloat16
  #define COMM_DTYPE_STR "bf16"
#else  // FP32 or TF32
  typedef float          COMM_FLOAT;
  #define NCCL_DTYPE     ncclFloat
  #define COMM_DTYPE_STR "fp32"
#endif

const char *cublasGetErrorString(cublasStatus_t status)
{
  switch (status)
  {
  case CUBLAS_STATUS_SUCCESS:            return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:       return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:      return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:      return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:      return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:   return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:     return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "unknown error";
}

inline void checkCuda(cudaError_t result)
{
  if (result != cudaSuccess)
  {
    fprintf(stderr, "GPU Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
}

inline void checkCublas(cublasStatus_t result)
{
  if (result != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr, "BLAS Error: %s\n", cublasGetErrorString(result));
    assert(result == CUBLAS_STATUS_SUCCESS);
  }
}

inline void checkNccl(ncclResult_t result)
{
  if (result != ncclSuccess)
  {
    fprintf(stderr, "NCCL Error: %s\n", ncclGetErrorString(result));
    assert(result == ncclSuccess);
  }
}

static unsigned long long now_us() {
  return (unsigned long long)std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch()).count();
}

inline void checkMpi(int result)
{
  if (result != MPI_SUCCESS)
  {
    fprintf(stderr, "MPI Error: code %d\n", result);
    assert(result == MPI_SUCCESS);
  }
}

static void make_nccl_comm(ncclComm_t &comm, int rank, int nranks, bool zero_cta) {
  ncclUniqueId id;
  if (rank == 0) checkNccl(ncclGetUniqueId(&id));
  checkMpi(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  if (zero_cta) {
    ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
    cfg.CTAPolicy = NCCL_CTA_POLICY_ZERO;
    checkNccl(ncclCommInitRankConfig(&comm, nranks, id, rank, &cfg));
  } else {
    checkNccl(ncclCommInitRank(&comm, nranks, id, rank));
  }
}

__global__ void init_data_flat(GEMM_FLOAT *X, size_t count, unsigned long long seed)
{
  size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (idx >= count) return;
  curandState state;
  curand_init(seed, idx, 0, &state);
  X[idx] = (GEMM_FLOAT)curand_uniform(&state);
}
__global__ void init_data_flat_constant(GEMM_FLOAT *X, size_t count, GEMM_FLOAT val)
{
  size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (idx >= count) return;
  X[idx] = val;
}

int main(int argc, char **argv)
{
  checkMpi(MPI_Init(&argc, &argv));
  int rank = 0, nranks = 1;
  checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

  BenchOpts opts = parse_bench_opts(argc, argv);
  const bool  use_barrier   = opts.use_barrier;
  const bool  use_comm_only = opts.comm_only;
  const bool  bw_compute    = (opts.compute_mode != COMPUTE_GEMM);
  const ComputeMode compute_mode = opts.compute_mode;
  const int   compute_passes = opts.compute_passes;
  const size_t compute_bytes = opts.compute_bytes;

  GemmArgs args  = parseArguments(argc, argv);
  CommArgs cargs = parseCommArguments(argc, argv);

  const bool skip_comm = (cargs.msg_size == 0);
  const bool need_nccl = !skip_comm || use_barrier;
  const bool overlap   = (cargs.overlap != 0);  // -o: 0 = sequential, >0 overlap

  size_t   matrix_dimension = args.matrix_size;
  int      repeats        = args.repeats;
  InitMode init_mode      = args.init_mode;

  int ndevices = 0;
  checkCuda(cudaGetDeviceCount(&ndevices));
  int      device_id = rank % ndevices;
  int      gpu_id    = device_id;


  // Align to 256 (tensor-core-friendly GEMM dims)
  const size_t ALIGN = 256;
  matrix_dimension = (matrix_dimension + ALIGN - 1) / ALIGN * ALIGN;
  const size_t m_dim = matrix_dimension, n_dim = matrix_dimension, k_dim = matrix_dimension;

  checkCuda(cudaSetDevice(device_id));

  if (rank == 0) {

    cout << HLINE;
#if defined(DOUBLE)
    cout << "Double Precision (FP64) GEMM + NCCL collective results:" << endl;
#elif defined(BFLOAT16)
    cout << "BFloat16 (BF16 in/out, FP32 accumulate) GEMM + NCCL collective results:" << endl;
#elif defined(TENSOR)
    cout << "Single Precision TF32 (FP32 in/out, TF32 TC math) GEMM + NCCL collective results:" << endl;
#else
    cout << "Single Precision (FP32) GEMM + NCCL collective results:" << endl;
#endif
    cout << HLINE;

    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, device_id));

    cout << "Summary:" << endl
         << "MPI ranks: "        << nranks             << endl
         << "GPU ID: "           << gpu_id           << endl
         << "Device Name: "      << prop.name          << endl
         << "Matrix dims (M,N,K): " << m_dim << "x" << n_dim << "x" << k_dim << endl;
          cout << "Repeats: " << repeats << endl;
    cout << "Init mode: "      << (init_mode == INIT_RANDOM ? "random" : "constant") << endl;
    if (cargs.msg_size == 0) {
      cout << "Collective: disabled (GEMM only, -m 0)" << endl;
    } else {
      cout << "Collective: "
           << (cargs.pattern == COMM_ALLGATHER      ? "allgather (SM)"
              : cargs.pattern == COMM_ALLREDUCE     ? "allreduce (SM)"
              : cargs.pattern == COMM_REDUCESCATTER ? "reducescatter (SM)"
              :                                     "ce_allgather (copy engines)")
           << endl
           << "Msg size: "       << cargs.msg_size / 1024.0 / 1024.0 << " MB" << endl;
    }
    if (!skip_comm)
      cout << "Overlap: " << (overlap ? "comm-first (collective overlaps the GEMM)"
                                     : "sequential (collective completes, then GEMM)") << endl;
    cout << "Barrier: " << (use_barrier ? "yes (every lap starts simultaneously on all ranks)"
                                        : "no (ranks progress independently)") << endl;
    cout << HLINE;
  }

  cublasHandle_t handle;
  checkCublas(cublasCreate(&handle));


#if !defined(DOUBLE) && !defined(BFLOAT16) && defined(TENSOR)
  checkCublas(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
#endif

  cudaStream_t gemm_stream;
  checkCuda(cudaStreamCreateWithFlags(&gemm_stream, cudaStreamNonBlocking));
  checkCublas(cublasSetStream(handle, gemm_stream));

#ifdef METRICS
  GpuMonitor monitor;
  if (!monitor.init(device_id, gpu_id)) { return 1; }
#endif

  size_t elems_A = (size_t)m_dim * (size_t)k_dim;
  size_t elems_B = (size_t)k_dim * (size_t)n_dim;
  size_t elems_C = (size_t)m_dim * (size_t)n_dim;
  size_t bytes_A = elems_A * sizeof(GEMM_FLOAT);
  size_t bytes_B = elems_B * sizeof(GEMM_FLOAT);
  size_t bytes_C = elems_C * sizeof(GEMM_FLOAT);
  GEMM_FLOAT *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  if (!use_comm_only && !bw_compute) {
    if (rank == 0) {
      cout << "Allocating device variables: A=" << bytes_A/1e9 << " B=" << bytes_B/1e9
           << " C=" << bytes_C/1e9 << " GB (total " << (bytes_A+bytes_B+bytes_C)/1e9 << " GB)" << endl;
      cout << HLINE;
    }
    checkCuda(cudaMalloc(&d_A, bytes_A));
    checkCuda(cudaMalloc(&d_B, bytes_B));
    checkCuda(cudaMalloc(&d_C, bytes_C));
    long threads = 256;
    auto launch_init = [&](GEMM_FLOAT *dst, size_t count, unsigned long long seed) {
      long blocks = ((long)count + (long)threads - 1) / (long)threads;
      if (init_mode == INIT_RANDOM)
        init_data_flat<<<blocks, threads>>>(dst, count, seed);
      else
        init_data_flat_constant<<<blocks, threads>>>(dst, count, (GEMM_FLOAT)0.1529);
    };
    launch_init(d_A, elems_A, (unsigned long long)time(NULL) + rank);
    launch_init(d_B, elems_B, (unsigned long long)time(NULL) + rank + 1);
    launch_init(d_C, elems_C, (unsigned long long)time(NULL) + rank + 2);
    checkCuda(cudaDeviceSynchronize());
  } else if (rank == 0) {
    cout << "Comm-only mode: GEMM skipped (characterizing collective performance).\n" << HLINE;
  }

#if defined(DOUBLE)
  const GEMM_FLOAT alf = 1.012, bet = 4.019;
  const GEMM_FLOAT *alpha = &alf, *beta = &bet;
#elif defined(BFLOAT16)
  // GemmEx scale factors follow the compute type (fp32), not the bf16 storage.
  const float alf = 1.0579f, bet = 0.0127f;
  const float *alpha = &alf, *beta = &bet;
#else
  const GEMM_FLOAT alf = 1.0579f, bet = 0.0127f;
  const GEMM_FLOAT *alpha = &alf, *beta = &bet;
#endif

  // do_gemm_call(...): one dtype-dispatched cuBLAS GEMM call (OP_N/OP_N, column-major).
  auto do_gemm_call = [&](GEMM_FLOAT *A_ptr, GEMM_FLOAT *B_ptr, GEMM_FLOAT *C_ptr,
                          int64_t mm, int64_t nn, int64_t kk,
                          int64_t lda_, int64_t ldb_, int64_t ldc_,
                          decltype(alpha) a_ptr, decltype(beta) b_ptr) -> cublasStatus_t {
#if defined(DOUBLE)
    return cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, mm, nn, kk,
                        a_ptr, A_ptr, lda_, B_ptr, ldb_, b_ptr, C_ptr, ldc_);
#elif defined(BFLOAT16)
    return cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)mm, (int)nn, (int)kk, a_ptr,
                         A_ptr, CUDA_R_16BF, (int)lda_, B_ptr, CUDA_R_16BF, (int)ldb_, b_ptr,
                         C_ptr, CUDA_R_16BF, (int)ldc_, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#else
    return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, mm, nn, kk,
                        a_ptr, A_ptr, lda_, B_ptr, ldb_, b_ptr, C_ptr, ldc_);
#endif
  };

  // --------------------------------------------------------------------------
  // Bandwidth-kernel arrays: only in --compute mode (A/B/C not allocated
  // there). --cpasses is a linear duration knob (fixed bytes per pass).
  // --------------------------------------------------------------------------
  constexpr int BW_BLOCKS = 32768;   // fixed grid size, like PyTorch's eltwise kernels
  bf16x8 *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
  __nv_bfloat162 *d_bias = nullptr;
  const unsigned  bias_pairs = 4096u;                    
  size_t elems_bw = compute_bytes / sizeof(__nv_bfloat16);
  elems_bw = (elems_bw / 8) * 8;                          // whole 128-bit vectors
  const size_t n8_bw = elems_bw / 8;
  // bytes of DRAM traffic per pass: triad reads b,c writes a (3 arrays);
  // eltwise reads x writes y (2 arrays, bias is cache-resident).
  const double bytes_per_pass = (double)elems_bw * sizeof(__nv_bfloat16)
                              * ((compute_mode == COMPUTE_TRIAD) ? 3.0 : 2.0);
  const double flops_per_pass = (double)elems_bw
                              * ((compute_mode == COMPUTE_TRIAD) ? 2.0 : 15.0);
  if (bw_compute && !use_comm_only) {
    checkCuda(cudaMalloc(&d_x, elems_bw * sizeof(__nv_bfloat16)));
    checkCuda(cudaMalloc(&d_y, elems_bw * sizeof(__nv_bfloat16)));
    init_bf16_random<<<8192, 256>>>(reinterpret_cast<__nv_bfloat16 *>(d_x), elems_bw, 1234ull + rank);
    init_bf16_random<<<8192, 256>>>(reinterpret_cast<__nv_bfloat16 *>(d_y), elems_bw, 5678ull + rank);
    if (compute_mode == COMPUTE_TRIAD) {
      checkCuda(cudaMalloc(&d_z, elems_bw * sizeof(__nv_bfloat16)));
      init_bf16_random<<<8192, 256>>>(reinterpret_cast<__nv_bfloat16 *>(d_z), elems_bw, 9012ull + rank);
    } else {
      checkCuda(cudaMalloc(&d_bias, bias_pairs * sizeof(__nv_bfloat162)));
      checkCuda(cudaMemset(d_bias, 0, bias_pairs * sizeof(__nv_bfloat162)));
    }
    checkCuda(cudaDeviceSynchronize());
    if (rank == 0)
      cout << "Compute phase: " << (compute_mode == COMPUTE_TRIAD ? "triad" : "eltwise(bias+gelu)")
           << " | " << compute_passes << " passes/lap"
           << " | " << (compute_bytes / (1024.0 * 1024.0)) << " MiB/array"
           << " | " << (bytes_per_pass / 1e9) << " GB traffic/pass"
           << " | AI(est) " << (flops_per_pass / bytes_per_pass) << " flop/byte"
           << " | init random" << endl;
  }
  // One compute phase = the GEMM, or compute_passes launches
  auto run_compute_phase = [&](int64_t mm, int64_t nn, int64_t kk,
                               int64_t lda_, int64_t ldb_, int64_t ldc_) -> cublasStatus_t {
    if (!bw_compute) return do_gemm_call(d_A, d_B, d_C, mm, nn, kk, lda_, ldb_, ldc_, alpha, beta);
    for (int p = 0; p < compute_passes; p++) {
      if (compute_mode == COMPUTE_TRIAD)
        triad_bf16_kernel<<<BW_BLOCKS, 256, 0, gemm_stream>>>(d_x, d_y, d_z, n8_bw, 1.0000001f);
      else
        bias_gelu_bf16_kernel<<<BW_BLOCKS, 256, 0, gemm_stream>>>(d_y, d_x, d_bias, n8_bw,
                                                                  bias_pairs - 1u);
    }
    return CUBLAS_STATUS_SUCCESS;
  };

  // -m 0: NCCL is still needed if --barrier is set.

  size_t       comm_count = 0;  // size_t: avoids int overflow for large bf16/half messages (>4GB → count>2^31)
  size_t       comm_bytes = 0;
  COMM_FLOAT  *comm_buf   = nullptr;
  COMM_FLOAT  *comm_send  = nullptr;
  COMM_FLOAT  *d_barrier_buf = nullptr;
  // ce_allgather needs NCCL 2.28+ symmetric memory + a zero-CTA communicator.
  const bool is_ce  = (cargs.pattern == COMM_CE_ALLGATHER);
  const bool is_rs  = (cargs.pattern == COMM_REDUCESCATTER);
  if (!skip_comm) {
    comm_count = cargs.msg_size / sizeof(COMM_FLOAT);
    if (comm_count < 1)
      comm_count = 1;
    comm_bytes = (size_t)nranks * (size_t)comm_count * sizeof(COMM_FLOAT);
    if (is_ce) {
      // Copy-engine collectives require symmetric memory (ncclMemAlloc), registered
      // into a window for CE-driven NVLink transfers.
      checkNccl(ncclMemAlloc((void **)&comm_buf, comm_bytes));
    } else {
      checkCuda(cudaMalloc((void **)&comm_buf, comm_bytes));
    }
    checkCuda(cudaMemset(comm_buf, 0, comm_bytes));
    // Send buffer: own chunk for the all-gather variants; reducescatter mirrors
    // it (nranks*count input in comm_buf, count-sized reduced output here).
    if (cargs.pattern == COMM_ALLGATHER || cargs.pattern == COMM_CE_ALLGATHER || is_rs) {
      size_t send_bytes = (size_t)comm_count * sizeof(COMM_FLOAT);
      if (is_ce) checkNccl(ncclMemAlloc((void **)&comm_send, send_bytes));
      else       checkCuda(cudaMalloc((void **)&comm_send, send_bytes));
      checkCuda(cudaMemset(comm_send, 0, send_bytes));
    }
  }

  cudaStream_t comm_stream = nullptr;
  ncclComm_t nccl_comm{};
  cudaStream_t barrier_stream = nullptr;
  ncclComm_t barrier_comm{};
  if (need_nccl) {
    checkCuda(cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking));
    make_nccl_comm(nccl_comm, rank, nranks, is_ce);
    if (use_barrier) {
      // Separate communicator/stream: NCCL serializes ops within one communicator.
      checkCuda(cudaStreamCreateWithFlags(&barrier_stream, cudaStreamNonBlocking));
      make_nccl_comm(barrier_comm, rank, nranks, false);
      checkCuda(cudaMalloc((void **)&d_barrier_buf, sizeof(COMM_FLOAT)));
      checkCuda(cudaMemset(d_barrier_buf, 0, sizeof(COMM_FLOAT)));
    }
  }

  // ---------------------------------------------------------------------------
  // Pattern-specific post-init setup
  // ---------------------------------------------------------------------------
  ncclWindow_t ce_win_send = nullptr, ce_win_buf = nullptr;
  if (!skip_comm && is_ce) {
    // Register the symmetric-allocated buffers into NCCL windows so the zero-CTA
    // AllGather dispatches to the copy engines. The per-iter ncclAllGather() call
    // is identical to the SM path — only setup (ncclMemAlloc + this registration) differs.
    checkNccl(ncclCommWindowRegister(nccl_comm, comm_send, (size_t)comm_count * sizeof(COMM_FLOAT),
                                     &ce_win_send, NCCL_WIN_COLL_SYMMETRIC));
    checkNccl(ncclCommWindowRegister(nccl_comm, comm_buf, comm_bytes,
                                     &ce_win_buf, NCCL_WIN_COLL_SYMMETRIC));
    if (rank == 0)
      cout << "ce_allgather: buffers registered into NCCL symmetric windows "
           << "(zero-CTA-policy comm).\n";
  }

  // launch_one_collective(stream): perform ONE collective on the given stream.
  // Dispatches by pattern. Captures setup state by reference.
  auto launch_one_collective = [&](cudaStream_t stream) {
    if (cargs.pattern == COMM_ALLGATHER || cargs.pattern == COMM_CE_ALLGATHER) {
      checkNccl(ncclAllGather((const void *)comm_send, (void *)comm_buf, (size_t)comm_count,
                              NCCL_DTYPE, nccl_comm, stream));
    } else if (is_rs) {
      // Mirror of allgather: reduce nranks*count input (comm_buf) into each rank's
      // own count-sized chunk (comm_send). recvcount = comm_count per NCCL semantics.
      checkNccl(ncclReduceScatter((const void *)comm_buf, (void *)comm_send, (size_t)comm_count,
                                  NCCL_DTYPE, ncclSum, nccl_comm, stream));
    } else {  // COMM_ALLREDUCE
      checkNccl(ncclAllReduce((const void *)comm_buf, (void *)comm_buf, (size_t)comm_count,
                              NCCL_DTYPE, ncclSum, nccl_comm, stream));
    }
  };

  // Device timeline origin (recorded after warmup): lap events are measured
  // relative to it; dev_t0_host_us aligns device to host-epoch time.
  cudaEvent_t dev_t0 = nullptr;
  unsigned long long dev_t0_host_us = 0;
  checkCuda(cudaEventCreate(&dev_t0));

  // Optional per-rank timeline CSV (GEMM_NCCL_TIMELINE_FILE=<dir>): device-aligned
  // event intervals + raw NVML samples, off by default.
  const char *tl_path = getenv("GEMM_NCCL_TIMELINE_FILE");
  bool do_timeline = (tl_path != nullptr && tl_path[0] != '\0');
  std::vector<std::tuple<int, double, double>> gemm_dev_ms;  // iter, start_ms, end_ms (rel dev_t0)

  // ---------------------------------------------------------------------------
  // Execution loop
  // ---------------------------------------------------------------------------
  double gemm_s         = 0.0;
  double wall_s    = 0.0;
  double comm_s    = 0.0;
  double barrier_s = 0.0;
  int actual_repeats = 0;
  int comm_laps      = 0;

  // Per-lap event pairs (GEMM and comm), created per-iter, read in the final
  // drain after a full device sync.
  std::vector<cudaEvent_t>        ev_comm_starts, ev_comm_stops;
  std::vector<unsigned long long> comm_start_us_vec;
  std::vector<cudaEvent_t>        ev_gemm_starts, ev_gemm_stops;

  // Enqueue one event-bracketed collective on comm_stream, at the top of each lap.
  auto enqueue_bg_comm = [&]() {
    unsigned long long ts = now_us();
    comm_start_us_vec.push_back(ts);
    cudaEvent_t cs, ce;
    checkCuda(cudaEventCreate(&cs));
    checkCuda(cudaEventCreate(&ce));
    checkCuda(cudaEventRecord(cs, comm_stream));
    launch_one_collective(comm_stream);
    checkCuda(cudaEventRecord(ce, comm_stream));
    ev_comm_starts.push_back(cs);
    ev_comm_stops.push_back(ce);
    // No cudaStreamSynchronize — comm runs in the background 
  };

#ifdef METRICS
  std::vector<TimeWindow> kernel_windows;
  std::vector<TimeWindow> barrier_windows;
  std::vector<TimeWindow> comm_windows;
#endif

  // Warm up cuBLAS and NCCL so first-call overhead stays out of the measurements.
  if (!use_comm_only) {
    int64_t m = (int64_t)m_dim, n = (int64_t)n_dim, k = (int64_t)k_dim;
    int64_t lda = m, ldb = k, ldc = m;
    run_compute_phase(m, n, k, lda, ldb, ldc);
    checkCuda(cudaStreamSynchronize(gemm_stream));
  }
  if (need_nccl && !skip_comm) {
    launch_one_collective(comm_stream);
    checkCuda(cudaStreamSynchronize(comm_stream));
  }
  if (use_barrier) {
    checkNccl(ncclAllReduce(d_barrier_buf, d_barrier_buf, 1, NCCL_DTYPE,
                            ncclSum, barrier_comm, barrier_stream));
    checkCuda(cudaStreamSynchronize(barrier_stream));
  }
  checkCuda(cudaDeviceSynchronize());

  // Record the device timeline origin (dev_t0) on the GEMM stream after warmup,
  // and capture the matching host timestamp for cross-clock alignment. Both
  // happen after a full sync, so dev_t0 completes immediately.
  checkCuda(cudaEventRecord(dev_t0, gemm_stream));
  checkCuda(cudaStreamSynchronize(gemm_stream));
  dev_t0_host_us = now_us();

#ifdef METRICS
  // Comm-only: skip the background monitor (polling perturbs short comm
  // kernels) unless GEMM_NCCL_COMMONLY_MONITOR=1 (power trace wanted).
  bool commonly_monitor = false;
  { const char *e = getenv("GEMM_NCCL_COMMONLY_MONITOR"); if (e && e[0]=='1') commonly_monitor = true; }
  long long ec_total_start = -1;
  if (use_comm_only) {
    ec_total_start = monitor.energy_counter_mj();   // single bracket call, no polling
    if (use_comm_only && commonly_monitor) monitor.start();
    if (rank == 0) printf("iter | rank | comm_s\n");
  } else {
    ec_total_start = monitor.energy_counter_mj();
    monitor.start();
    printf("iter | rank | gemm_s | barrier_s | comm_s | f_sm_pre_MHz | f_sm_avg_MHz | f_sm_post_MHz | kernel_J\n");
  }
#endif

  while (true)
  {
    if (actual_repeats >= repeats) break;
    auto iter_t0 = chrono::high_resolution_clock::now();

    if (use_comm_only) {
      if (skip_comm) break;
      comm_laps++;
      enqueue_bg_comm();                                     // same event bookkeeping as the main loop
      checkCuda(cudaStreamSynchronize(comm_stream));         // bounded: one collective in flight
      float cms = 0.0f;
      checkCuda(cudaEventElapsedTime(&cms, ev_comm_starts.back(), ev_comm_stops.back()));
      double comm_iter = cms / 1000.0;
      // comm_s is accumulated by the end-of-loop drain (uniform path); don't double-count here.
#ifdef METRICS
      printf("%4d | %d | %7.4f\n", actual_repeats, rank, comm_iter);
#endif
      wall_s += chrono::duration<double>(chrono::high_resolution_clock::now() - iter_t0).count();
      actual_repeats++;
      continue;
    }

    int64_t m = (int64_t)m_dim, n = (int64_t)n_dim, k = (int64_t)k_dim;
    int64_t lda = m, ldb = k, ldc = m;

    if (!skip_comm) {
      comm_laps++;
      enqueue_bg_comm();                                    
      if (!overlap)
        checkCuda(cudaStreamSynchronize(comm_stream));     // -o 0: exposed comm completes before the GEMM
    }

#ifdef METRICS
    unsigned long long t_kernel_start_us = now_us();        // for the pre-GEMM clock sample
#endif

    // Per-lap event pair around the compute phase (read in the drain).
    cudaEvent_t gs, ge;
    checkCuda(cudaEventCreate(&gs));
    checkCuda(cudaEventCreate(&ge));
    checkCuda(cudaEventRecord(gs, gemm_stream));

    cublasStatus_t stat = run_compute_phase(m, n, k, lda, ldb, ldc);

    checkCuda(cudaEventRecord(ge, gemm_stream));
    ev_gemm_starts.push_back(gs);
    ev_gemm_stops.push_back(ge);
    assert(stat == CUBLAS_STATUS_SUCCESS);
    assert(!cudaGetLastError());

    // End-of-lap sync (LOCAL): this rank's comm and GEMM complete before the
    // next lap; --barrier below adds the cross-rank wait. (-o 0 already
    // synced the comm at the lap top; the sync here returns immediately.)
    if (!skip_comm)
      checkCuda(cudaStreamSynchronize(comm_stream));
    checkCuda(cudaStreamSynchronize(gemm_stream));

    // This lap's GEMM duration (complete — synced above)
    float gemm_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&gemm_ms, ev_gemm_starts.back(), ev_gemm_stops.back()));
    double gemm_iter = gemm_ms / 1000.0;

    // ---- Barrier (--barrier): lap-start rank sync. barrier_s = wait for the
    // slowest rank; separate communicator so data comms never block it.
    double barrier_iter = 0.0;
    unsigned long long t_barrier_start_us = 0, t_barrier_end_us = 0;
    if (use_barrier) {
      t_barrier_start_us = now_us();
      checkNccl(ncclAllReduce(d_barrier_buf, d_barrier_buf, 1, NCCL_DTYPE,
                              ncclSum, barrier_comm, barrier_stream));
      checkCuda(cudaStreamSynchronize(barrier_stream));
      t_barrier_end_us = now_us();
      barrier_iter = (t_barrier_end_us - t_barrier_start_us) * 1e-6;
    }
    barrier_s += barrier_iter;

    // This lap's comm duration (final — synced above).
    double comm_iter = 0.0;
    if (!skip_comm) {
      float cms = 0.0f;
      checkCuda(cudaEventElapsedTime(&cms, ev_comm_starts.back(), ev_comm_stops.back()));
      comm_iter = cms / 1000.0;
    }

#ifdef METRICS
    {
      // Execution window in host-epoch µs, from the device timeline origin.
      float ds = 0.0f, de = 0.0f;
      checkCuda(cudaEventElapsedTime(&ds, dev_t0, ev_gemm_starts.back()));
      checkCuda(cudaEventElapsedTime(&de, dev_t0, ev_gemm_stops.back()));
      unsigned long long ws = dev_t0_host_us + (unsigned long long)(ds * 1000.0);
      unsigned long long we = dev_t0_host_us + (unsigned long long)(de * 1000.0);
      kernel_windows.push_back({ws, we});
      if (use_barrier && t_barrier_end_us > t_barrier_start_us)
        barrier_windows.push_back({t_barrier_start_us, t_barrier_end_us});
      unsigned int f_sm_pre  = monitor.clock_near_us(t_kernel_start_us);
      unsigned int f_sm_avg  = (unsigned int)monitor.avg_clock_in_window(ws, we);
      unsigned int f_sm_post = monitor.clock_near_us(we);
      double iter_kernel_J   = monitor.energy_in_window_locked({ws, we});
      printf("%4d | %d | %7.4f | %7.4f | %7.4f | %u | %u | %u | %.4f\n",
             actual_repeats, rank, gemm_iter, barrier_iter, comm_iter,
             f_sm_pre, f_sm_avg, f_sm_post, iter_kernel_J);
    }
#endif

    wall_s += chrono::duration<double>(
        chrono::high_resolution_clock::now() - iter_t0).count();

    actual_repeats++;
  }

  // Final drain: complete every pending compute phase and collective, then
  // read the event-based timings (gemm_s, comm_s) and fill the timeline rows.
  std::vector<std::pair<double,double>> comm_dev_ms;  // (start_ms, end_ms)
  if (!ev_gemm_starts.empty()) {
    checkCuda(cudaDeviceSynchronize());
    for (size_t i = 0; i < ev_gemm_starts.size(); i++) {
      float ms = 0.0f;
      checkCuda(cudaEventElapsedTime(&ms, ev_gemm_starts[i], ev_gemm_stops[i]));
      gemm_s += ms / 1000.0;
      if (do_timeline) {
        float ds = 0.0f, de = 0.0f;
        checkCuda(cudaEventElapsedTime(&ds, dev_t0, ev_gemm_starts[i]));
        checkCuda(cudaEventElapsedTime(&de, dev_t0, ev_gemm_stops[i]));
        gemm_dev_ms.emplace_back((int)i, (double)ds, (double)de);
      }
      checkCuda(cudaEventDestroy(ev_gemm_starts[i]));
      checkCuda(cudaEventDestroy(ev_gemm_stops[i]));
    }
  }
  if (!ev_comm_starts.empty()) {
    checkCuda(cudaDeviceSynchronize());
    for (size_t i = 0; i < ev_comm_starts.size(); i++) {
      float ms = 0.0f;
      checkCuda(cudaEventElapsedTime(&ms, ev_comm_starts[i], ev_comm_stops[i]));
      comm_s += ms / 1000.0;
#ifdef METRICS
      unsigned long long cs_us = comm_start_us_vec[i];
      unsigned long long ce_us = cs_us + (unsigned long long)(ms * 1000.0);
      comm_windows.push_back({cs_us, ce_us});
#endif
      if (do_timeline) {
        float cms_s = 0.0f, cms_e = 0.0f;
        cudaEventElapsedTime(&cms_s, dev_t0, ev_comm_starts[i]);
        cudaEventElapsedTime(&cms_e, dev_t0, ev_comm_stops[i]);
        comm_dev_ms.emplace_back((double)cms_s, (double)cms_e);
      }
      checkCuda(cudaEventDestroy(ev_comm_starts[i]));
      checkCuda(cudaEventDestroy(ev_comm_stops[i]));
    }
  }

#ifdef METRICS
  long long ec_total_end = -1;
  double kernel_energy_J = 0.0, barrier_energy_J = 0.0, comm_energy_J = 0.0;
  double kernel_power_stddev = 0.0, kernel_avg_clock = 0.0, kernel_avg_mempower = 0.0;
  if (!use_comm_only) {
    monitor.stop();
    ec_total_end     = monitor.energy_counter_mj();
    kernel_energy_J  = monitor.energy_in_windows(kernel_windows);
    barrier_energy_J = monitor.energy_in_windows(barrier_windows);
    comm_energy_J    = monitor.energy_in_windows(comm_windows);
    kernel_power_stddev = monitor.power_stddev_in_windows(kernel_windows);
    kernel_avg_clock    = monitor.avg_clock_in_windows(kernel_windows);
    kernel_avg_mempower = monitor.avg_mempower_in_windows(kernel_windows);
    monitor.drain();  // flush tail power samples so the timeline trace is complete
  } else {
    if (use_comm_only && commonly_monitor) { monitor.stop(); monitor.drain(); }
    ec_total_end = monitor.energy_counter_mj();
    // comm-only: per-window energy not computed (no kernel windows); total energy from counter.
  }
#endif

  // ---- Write per-rank timeline trace (GEMM_NCCL_TIMELINE_FILE) ----------------
  // Device-aligned event intervals (true overlap) + raw NVML power/clock samples.
  if (do_timeline) {
    std::string dir = tl_path;
    // Treat tl_path as a directory; create it if needed. One file per rank so each
    // GPU's power stream stays self-contained and aligns with that rank's events.
    std::error_code ec_mkdir;
    std::filesystem::create_directories(dir, ec_mkdir);
    std::string fpath = dir + "/timeline_rank" + std::to_string(rank) + ".csv";
    std::ofstream ofs(fpath, std::ios::trunc);
    if (ofs) {
      ofs << "# GEMM_NCCL timeline rank=" << rank
          << " dev_t0_host_us=" << dev_t0_host_us
          << " size=" << m_dim << "x" << n_dim << "x" << k_dim
          << " repeats=" << actual_repeats
          << "\n";
      ofs << "# columns: kind,iter,dev_ms,host_us,value,unit\n";
      ofs << "# events: dev_ms = ms from dev_t0; comm_start is the ENQUEUE time, not kernel start\n";
      ofs << "# units: power=mW clock=MHz mempower=W (HBM rail, ~1s smoothed)\n";
      // GEMM intervals (one row per start/end per iter)
      for (const auto& g : gemm_dev_ms)
        ofs << "gemm_start," << std::get<0>(g) << "," << std::get<1>(g)
            << "," << (dev_t0_host_us + (unsigned long long)(std::get<1>(g)*1000.0)) << ",,\n"
            << "gemm_end," << std::get<0>(g) << "," << std::get<2>(g)
            << "," << (dev_t0_host_us + (unsigned long long)(std::get<2>(g)*1000.0)) << ",,\n";
      // Comm intervals (lap index = row order)
      for (size_t i = 0; i < comm_dev_ms.size(); i++)
        ofs << "comm_start," << i << "," << comm_dev_ms[i].first
            << "," << (dev_t0_host_us + (unsigned long long)(comm_dev_ms[i].first*1000.0)) << ",,\n"
            << "comm_end," << i << "," << comm_dev_ms[i].second
            << "," << (dev_t0_host_us + (unsigned long long)(comm_dev_ms[i].second*1000.0)) << ",,\n";
#ifdef METRICS
      // Raw power + clock sample history (NVML monitor — METRICS builds only)
      monitor.write_trace_rows(ofs);
#endif
      if (rank == 0)
        cout << "Timeline trace written: " << fpath << " (" << gemm_dev_ms.size()
             << " gemm iters, " << comm_dev_ms.size() << " comm laps)\n";
    } else if (rank == 0) {
      std::cerr << "WARN: could not open timeline file " << fpath << "\n";
    }
  }

  // Cleanup
  if (dev_t0) checkCuda(cudaEventDestroy(dev_t0));

  if (actual_repeats == 0) actual_repeats = 1;

  checkCublas(cublasSetStream(handle, nullptr));
  if (d_x)    checkCuda(cudaFree(d_x));
  if (d_y)    checkCuda(cudaFree(d_y));
  if (d_z)    checkCuda(cudaFree(d_z));
  if (d_bias) checkCuda(cudaFree(d_bias));
  checkCuda(cudaStreamDestroy(gemm_stream));
  if (!skip_comm && is_ce) {
    if (ce_win_send) checkNccl(ncclCommWindowDeregister(nccl_comm, ce_win_send));
    if (ce_win_buf)  checkNccl(ncclCommWindowDeregister(nccl_comm, ce_win_buf));
    if (comm_send) checkNccl(ncclMemFree(comm_send));
    if (comm_buf)  checkNccl(ncclMemFree(comm_buf));
  }
  if (need_nccl) {
    checkNccl(ncclCommDestroy(nccl_comm));
    checkCuda(cudaStreamDestroy(comm_stream));
    if (use_barrier) {
      checkNccl(ncclCommDestroy(barrier_comm));
      checkCuda(cudaStreamDestroy(barrier_stream));
      if (d_barrier_buf) checkCuda(cudaFree(d_barrier_buf));
    }
  }
  if (!skip_comm && !is_ce) { 
    if ((cargs.pattern == COMM_ALLGATHER || cargs.pattern == COMM_REDUCESCATTER) && comm_send)
      checkCuda(cudaFree(comm_send));
    if (comm_buf)
      checkCuda(cudaFree(comm_buf));
  }
  checkCuda(cudaFree(d_A));
  checkCuda(cudaFree(d_B));
  checkCuda(cudaFree(d_C));

  // Bandwidth phases: count the filler's real flops
  double totalFlops  = bw_compute
      ? flops_per_pass * compute_passes * actual_repeats
      : 2.0 * (double)m_dim * (double)n_dim * (double)k_dim * actual_repeats;
  // Achieved bandwidth of the compute phase (0 for a GEMM).
  double compute_bw_gbs = 0.0;
  double comm_time_report = comm_s;
  double comm_lat_us =
      (skip_comm || comm_laps <= 0)
          ? 0.0
          : (comm_time_report / comm_laps * 1e6);
  // Overlap estimate: comm time hidden under the wall clock.
  double overlap_s = std::max(0.0, gemm_s + comm_s - wall_s);

  // Derived comm bandwidth (no NVML — pure arithmetic from CUDA event times).
  // All-gather: each rank receives nranks*msg_size per op (algorithmic volume).
  // comm_lat_us is per-op; per-op seconds = comm_lat_us/1e6.
  double per_op_s = comm_lat_us / 1e6;
  double comm_bw_alg_gbs  = (per_op_s > 0.0) ? ((double)nranks * cargs.msg_size) / per_op_s / 1e9 : 0.0;

  double perf_gemm_tf   = (gemm_s > 0.0) ? (totalFlops / gemm_s / 1e12) : 0.0;
  if (bw_compute && gemm_s > 0.0)
    compute_bw_gbs = bytes_per_pass * compute_passes * actual_repeats / gemm_s / 1e9;
  double perf_total_tf  = (wall_s > 0.0) ? (totalFlops / wall_s / 1e12) : 0.0;

  // ---------------------------------------------------------------------------
  // Output — all ranks print so per-rank imbalance is visible.
  // Prefix [rank=N] lets analysis scripts select rank=0 as canonical.
  // ---------------------------------------------------------------------------
  {
#ifdef METRICS
    MetricsAvg avg = monitor.averages();
    double total_energy_ctr_j = (ec_total_start >= 0 && ec_total_end >= 0)
        ? (double)(ec_total_end - ec_total_start) / 1000.0 : -1.0;
    double kernel_flops_gj = (kernel_energy_J    > 0.0) ? totalFlops / kernel_energy_J    / 1e9 : 0.0;
    double total_flops_gj  = (total_energy_ctr_j > 0.0) ? totalFlops / total_energy_ctr_j / 1e9 : 0.0;
    double kernel_avg_pwr  = (gemm_s > 0.0) ? kernel_energy_J / gemm_s : 0.0;
    cout << "[rank=" << rank << "] size " << m_dim << "x" << n_dim << "x" << k_dim
         << " | Total time: "    << wall_s << " s"
         << " | GEMM time: "     << gemm_s << " s"
         << " | Barrier time: "  << barrier_s << " s"
         << " | Comm time: "     << comm_time_report << " s"
         << " | Repeats: "       << actual_repeats
         << " | Perf (GEMM): "   << perf_gemm_tf << " TFlop/s"
         << " | Perf (total): "  << perf_total_tf << " TFlop/s"
         << " | Comm overlap: "              << overlap_s << " s"
         << " | Comm lat (µs/msg op): "      << comm_lat_us
         << " | Comm BW (alg, GB/s): "      << comm_bw_alg_gbs
         << " | kernel_energy_J: "      << kernel_energy_J
         << " | barrier_energy_J: "     << barrier_energy_J
         << " | comm_energy_J: "        << comm_energy_J
         << " | total_energy_J (ctr): " << total_energy_ctr_j
         << " | GFLOP_per_J (kernel): " << kernel_flops_gj
         << " | GFLOP_per_J (total): "  << total_flops_gj
         << " | Avg Power (kernel): "        << kernel_avg_pwr << " W"
         << " | kernel_power_stddev_W: "     << kernel_power_stddev
         << " | Avg Clock (kernel): "        << kernel_avg_clock << " MHz"
         << " | Avg Clock: "                 << avg.clock << " MHz"
         << " | Avg Temp: "                  << avg.temp << " C"
         << " | " << GpuMonitor::util_label() << ": " << avg.gpu_util << " %"
         << " | Mem Util: "                  << avg.mem_util << " %"
         // HBM-rail power (~1s trailing avg from the driver): kernel-window
         // value only meaningful when GEMM iterations are >=1s.
         << " | Avg Mem Power: "             << avg.mem_power << " W"
         << " | Avg Mem Power (kernel): "    << kernel_avg_mempower << " W"
         << " | comm_dtype: "                << COMM_DTYPE_STR
         << " | barrier: "                   << (use_barrier ? "yes" : "no")
         << " | comm_only: "                 << (use_comm_only ? "yes" : "no")
         << " | compute_kernel: "            << (compute_mode == COMPUTE_GEMM ? "gemm" :
                                                 compute_mode == COMPUTE_TRIAD ? "triad" : "eltwise")
         << " | compute_passes: "            << (bw_compute ? compute_passes : 0)
         << " | compute_BW (GB/s): "         << compute_bw_gbs
         << endl;
#else
    cout << "[rank=" << rank << "] size " << m_dim << "x" << n_dim << "x" << k_dim
         << " | Total time: "    << wall_s << " s"
         << " | GEMM time: "     << gemm_s << " s"
         << " | Barrier time: "  << barrier_s << " s"
         << " | Comm time: "     << comm_time_report << " s"
         << " | Repeats: "       << actual_repeats
         << " | Perf (GEMM): "   << perf_gemm_tf << " TFlop/s"
         << " | Perf (total): "  << perf_total_tf << " TFlop/s"
         << " | Comm overlap: "         << overlap_s  << " s"
         << " | Comm lat (µs/msg op): " << comm_lat_us
         << " | Comm BW (alg, GB/s): " << comm_bw_alg_gbs
         << " | comm_dtype: "           << COMM_DTYPE_STR
         << " | barrier: "              << (use_barrier ? "yes" : "no")
         << " | comm_only: "            << (use_comm_only ? "yes" : "no")
         << " | compute_kernel: "       << (compute_mode == COMPUTE_GEMM ? "gemm" :
                                            compute_mode == COMPUTE_TRIAD ? "triad" : "eltwise")
         << " | compute_passes: "       << (bw_compute ? compute_passes : 0)
         << " | compute_BW (GB/s): "    << compute_bw_gbs
         << " | Energy: N/A (build with -DMETRICS)" << endl;
#endif
  }

#ifdef METRICS
  monitor.shutdown();
#endif

  checkMpi(MPI_Finalize());
  return 0;
}
