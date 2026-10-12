#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include "cli.h"
#include "metrics.h"
#include <iostream>
#include <vector>
#include <assert.h>
#include <chrono>
#include <thread>
#include <time.h>

using namespace std;

// Select precision via -DDOUBLE (double), -DBFLOAT16 (bf16 storage, fp32
// accumulate via cublasGemmEx), or default (single).
#if defined(DOUBLE)
typedef double GEMM_FLOAT;
#elif defined(BFLOAT16)
typedef __nv_bfloat16 GEMM_FLOAT;
#else
typedef float GEMM_FLOAT;
#endif

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Kernel: Initialize matrices with random values
// ---------------------------------------------------------------------------
__global__ void init_data(GEMM_FLOAT *A, GEMM_FLOAT *B, GEMM_FLOAT *C, size_t size, unsigned long long seed)
{
  size_t idx   = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t total = size * size;
  if (idx >= total) return;

  curandState state;
  curand_init(seed, idx, 0, &state);
  A[idx] = (GEMM_FLOAT)curand_uniform(&state);
  B[idx] = (GEMM_FLOAT)curand_uniform(&state);
  C[idx] = (GEMM_FLOAT)curand_uniform(&state);
}

// ---------------------------------------------------------------------------
// Kernel: Initialize matrices with a fixed constant value
// ---------------------------------------------------------------------------
__global__ void init_data_constant(GEMM_FLOAT *A, GEMM_FLOAT *B, GEMM_FLOAT *C, size_t size)
{
  size_t idx   = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  size_t total = size * size;
  if (idx >= total) return;

  A[idx] = (GEMM_FLOAT)0.1529;
  B[idx] = (GEMM_FLOAT)1.2631;
  C[idx] = (GEMM_FLOAT)0.0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
  GemmArgs args   = parseArguments(argc, argv);
  SleepArgs sargs = parseSleepArguments(argc, argv);

  size_t   matrix_dimension = args.matrix_size;
  int      repeats        = args.repeats;
  InitMode init_mode      = args.init_mode;

  int      device_id = args.device_id;
  int      gpu_id    = device_id;


  // Align to 256
  const size_t ALIGN = 256;
  matrix_dimension = (matrix_dimension + ALIGN - 1) / ALIGN * ALIGN;

  checkCuda(cudaSetDevice(device_id));

  cout << HLINE;
#if defined(DOUBLE)
  cout << "Double Precision GEMM + host sleep results:" << endl;
#elif defined(BFLOAT16)
  cout << "BFloat16 (BF16 in/out, FP32 accumulate via cublasGemmEx) GEMM + host sleep results:" << endl;
#elif defined(TENSOR)
  cout << "Single Precision TF32 (FP32 in/out, TF32 tensor core math via cublasSgemm + CUBLAS_TF32_TENSOR_OP_MATH) GEMM + host sleep results:" << endl;
#else
  cout << "Single Precision GEMM + host sleep results:" << endl;
#endif
  cout << HLINE;

  {
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, device_id));

    cout << "Summary:" << endl
         << "GPU ID: "           << gpu_id           << endl
         << "Device Name: "      << prop.name          << endl
         << "Matrix_dimension: " << matrix_dimension   << endl;
          cout << "Repeats: " << repeats << endl;
    cout << "Init mode: "      << (init_mode == INIT_RANDOM ? "random" : "constant") << endl;
    if (sargs.sleep_sec <= 0.0) {
      cout << "Sleep: disabled (GEMM only, -m 0)" << endl;
    } else {
      cout << "Sleep: "          << sargs.sleep_sec << " s per lap" << endl;
    }
    cout << HLINE;
  }

  cublasHandle_t handle;
  checkCublas(cublasCreate(&handle));


#if !defined(DOUBLE) && !defined(BFLOAT16) && defined(TENSOR)
  checkCublas(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
#endif

  cudaStream_t gemm_stream;
  checkCuda(cudaStreamCreate(&gemm_stream));
  checkCublas(cublasSetStream(handle, gemm_stream));

#ifdef METRICS
  GpuMonitor monitor;
  if (!monitor.init(device_id, gpu_id)) return 1;
  std::vector<TimeWindow> kernel_windows;
#endif

  // ---------------------------------------------------------------------------
  // Memory allocation and data initialization
  // ---------------------------------------------------------------------------
  size_t bytes = (size_t)matrix_dimension * (size_t)matrix_dimension * sizeof(GEMM_FLOAT);
  cout << "Allocating device variables with total size: " << 3 * bytes / 1e9 << " GB" << endl;
  cout << HLINE;

  std::vector<GEMM_FLOAT*> d_As(1), d_Bs(1), d_Cs(1);
  long threads = 256;
  long blocks  = ((long)matrix_dimension * (long)matrix_dimension) / threads;
  unsigned long long seed_base = (unsigned long long)time(NULL);
  checkCuda(cudaMalloc(&d_As[0], bytes));
  checkCuda(cudaMalloc(&d_Bs[0], bytes));
  checkCuda(cudaMalloc(&d_Cs[0], bytes));
  if (init_mode == INIT_RANDOM)
    init_data<<<blocks, threads>>>(d_As[0], d_Bs[0], d_Cs[0], matrix_dimension, seed_base);
  else
    init_data_constant<<<blocks, threads>>>(d_As[0], d_Bs[0], d_Cs[0], matrix_dimension);
  checkCuda(cudaDeviceSynchronize());

#if defined(DOUBLE)
  const GEMM_FLOAT alf = 1.012,   bet = 4.019;
  const GEMM_FLOAT *alpha = &alf, *beta = &bet;
#elif defined(BFLOAT16)
  // cublasGemmEx with CUBLAS_COMPUTE_32F expects fp32 alpha/beta (scale type
  // tracks the compute type, not the storage type of A/B/C).
  const float alf = 1.0579f, bet = 0.0127f;
  const float *alpha = &alf, *beta = &bet;
#else
  const GEMM_FLOAT alf = 1.0579f, bet = 0.0127f;
  const GEMM_FLOAT *alpha = &alf, *beta = &bet;
#endif

  // -m 0: pure GEMM timing — no sleep phase (same idea as gemm_comms -m 0).
  const bool skip_sleep = (sargs.sleep_sec <= 0.0);

  cudaEvent_t ev_start, ev_stop;
  checkCuda(cudaEventCreate(&ev_start));
  checkCuda(cudaEventCreate(&ev_stop));

  // ---------------------------------------------------------------------------
  // Execution loop
  // ---------------------------------------------------------------------------
  double sum      = 0.0;
  double wall_sum = 0.0;  // sum of per-iteration wall times (GEMM path + sleep + cudaDeviceSynchronize)
  int actual_repeats = 0;
  double sleep_sum = 0.0;

#ifdef METRICS
  monitor.start();
  printf("iter | gemm_s | f_sm_pre_MHz | f_sm_avg_MHz | f_sm_post_MHz | kernel_J\n");
#endif

  while (true)
  {
    if (actual_repeats >= repeats) break;

    auto iter_t0 = chrono::high_resolution_clock::now();

    int64_t m = matrix_dimension, n = matrix_dimension, k = matrix_dimension;
    int64_t lda = m, ldb = k, ldc = m;
    GEMM_FLOAT *d_A = d_As[0], *d_B = d_Bs[0], *d_C = d_Cs[0];

#ifdef METRICS
    unsigned long long t_kernel_start_us = (unsigned long long)
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
#endif

    checkCuda(cudaEventRecord(ev_start, gemm_stream));

#if defined(DOUBLE)
    cublasStatus_t stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
#elif defined(BFLOAT16)
    cublasStatus_t stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         (int)m, (int)n, (int)k,
                                         alpha,
                                         d_A, CUDA_R_16BF, (int)lda,
                                         d_B, CUDA_R_16BF, (int)ldb,
                                         beta,
                                         d_C, CUDA_R_16BF, (int)ldc,
                                         CUBLAS_COMPUTE_32F,
                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#else
    cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
#endif

    checkCuda(cudaEventRecord(ev_stop, gemm_stream));
    assert(stat == CUBLAS_STATUS_SUCCESS);
    assert(!cudaGetLastError());

    // Sync stream before recording GEMM time and appending window.
    checkCuda(cudaStreamSynchronize(gemm_stream));

    float elapsed_ms;
    checkCuda(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    double gemm_iter = elapsed_ms / 1000.0;
    sum += (float)gemm_iter;

#ifdef METRICS
    unsigned long long t_kernel_end_us =
        t_kernel_start_us + (unsigned long long)(gemm_iter * 1e6);
    kernel_windows.push_back({t_kernel_start_us, t_kernel_end_us});
    unsigned int f_sm_pre  = monitor.clock_near_us(t_kernel_start_us);
    unsigned int f_sm_avg  = (unsigned int)monitor.avg_clock_in_window(t_kernel_start_us, t_kernel_end_us);
    unsigned int f_sm_post = monitor.clock_near_us(t_kernel_end_us);
    double iter_kernel_J   = monitor.energy_in_window_locked({t_kernel_start_us, t_kernel_end_us});
    printf("%4d | %7.4f | %u | %u | %u | %.4f\n",
           actual_repeats, gemm_iter, f_sm_pre, f_sm_avg, f_sm_post, iter_kernel_J);
#endif

    double sleep_iter = 0.0;
    auto ct0 = chrono::high_resolution_clock::now();
    if (!skip_sleep) {
      std::this_thread::sleep_for(std::chrono::duration<double>(sargs.sleep_sec));
      sleep_iter =
          chrono::duration<double>(chrono::high_resolution_clock::now() - ct0).count();
    }
    sleep_sum += sleep_iter;

    wall_sum += chrono::duration<double>(
        chrono::high_resolution_clock::now() - iter_t0).count();

    actual_repeats++;
  }

#ifdef METRICS
  monitor.stop();
  double kernel_energy_J     = monitor.energy_in_windows(kernel_windows);
  double kernel_power_stddev = monitor.power_stddev_in_windows(kernel_windows);
  // Mean SM clock over the GEMM windows only.
  double kernel_avg_clock    = monitor.avg_clock_in_windows(kernel_windows);
#endif

  if (actual_repeats == 0) actual_repeats = 1;

  checkCublas(cublasSetStream(handle, nullptr));
  checkCuda(cudaStreamDestroy(gemm_stream));
  checkCuda(cudaFree(d_As[0]));
  checkCuda(cudaFree(d_Bs[0]));
  checkCuda(cudaFree(d_Cs[0]));

  double totalFlops  = 2.0 * matrix_dimension * matrix_dimension * matrix_dimension * actual_repeats;
  // Average sleep duration per lap, in microseconds (sleep_for accuracy).
  double sleep_lat_us =
      (skip_sleep || actual_repeats <= 0)
          ? 0.0
          : (sleep_sum / actual_repeats * 1e6);
  double perf_gemm_tf   = (sum > 0.0) ? (totalFlops / sum / 1e12) : 0.0;
  double perf_total_tf  = (wall_sum > 0.0) ? (totalFlops / wall_sum / 1e12) : 0.0;
  double sleep_pct_wall = (wall_sum > 0.0) ? (100.0 * sleep_sum / wall_sum) : 0.0;
  double nongemm_pct_w  = (wall_sum > 0.0) ? (100.0 * (wall_sum - sum) / wall_sum) : 0.0;

  // ---------------------------------------------------------------------------
  // Output
  // ---------------------------------------------------------------------------
  {
#ifdef METRICS
    MetricsAvg avg = monitor.averages();
    // Total energy: prefer timestamped sample-integral; fall back to avg*wall.
    double total_energy_J  = (avg.energy_J > 0.0) ? avg.energy_J : avg.power * wall_sum;
    double kernel_flops_GJ    = (kernel_energy_J > 0.0) ? totalFlops / kernel_energy_J / 1e9 : 0.0;
    double total_flops_GJ     = (total_energy_J  > 0.0) ? totalFlops / total_energy_J  / 1e9 : 0.0;
    double kernel_avg_power_W = (sum > 0.0 && kernel_energy_J > 0.0) ? kernel_energy_J / sum : 0.0;
    cout << "size "          << matrix_dimension
         << " | Total time: "    << wall_sum << " s"
         << " | GEMM time: "     << sum << " s"
         << " | Repeats: "       << actual_repeats
         << " | Perf (GEMM): "  << perf_gemm_tf << " TFlop/s"
         << " | Perf (total): " << perf_total_tf << " TFlop/s"
         << " | Sleep time: "    << sleep_sum << " s"
         << " | Sleep % of total: " << sleep_pct_wall << " %"
         << " | Non-GEMM wall %: " << nongemm_pct_w << " %"
         << " | Sleep lat (µs/sleep): " << sleep_lat_us
         << " | Energy (kernel): "  << kernel_energy_J << " J"
         << " | Energy (total): "   << total_energy_J  << " J"
         << " | FLOPS/J (kernel): "    << kernel_flops_GJ    << " GFlop/J"
         << " | FLOPS/J (total): "     << total_flops_GJ     << " GFlop/J"
         << " | Avg Power (kernel): " << kernel_avg_power_W << " W"
         << " | Power StdDev (kernel): " << kernel_power_stddev << " W"
         << " | Avg Power (total): "  << avg.power << " W"
         << " | Avg Clock (kernel): " << kernel_avg_clock << " MHz"
         << " | Avg Clock: " << avg.clock << " MHz"
         << " | Avg Temp: "  << avg.temp  << " C"
         << " | " << GpuMonitor::util_label() << ": " << avg.gpu_util << " %"
         << " | Mem Util: "  << avg.mem_util << " %" << endl;
#else
    cout << "size "          << matrix_dimension
         << " | Total time: "    << wall_sum << " s"
         << " | GEMM time: "     << sum << " s"
         << " | Repeats: "       << actual_repeats
         << " | Perf (GEMM): "  << perf_gemm_tf << " TFlop/s"
         << " | Perf (total): " << perf_total_tf << " TFlop/s"
         << " | Sleep time: "    << sleep_sum << " s"
         << " | Sleep % of total: " << sleep_pct_wall << " %"
         << " | Non-GEMM wall %: " << nongemm_pct_w << " %"
         << " | Sleep lat (µs/sleep): " << sleep_lat_us
         << " | Energy: N/A (build with -DMETRICS)" << endl;
#endif
  }

#ifdef METRICS
  monitor.shutdown();
#endif

  return 0;
}
