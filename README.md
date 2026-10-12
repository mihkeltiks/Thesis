Benchmark codes used for the Master thesis "An investigation of the energy behavior of multi-GPU workloads".

## GEMMS

Codes modified from the GEMM benchmark in the [RRZE-HPC benchmark collection](https://github.com/RRZE-HPC/benchmark-collection).
Includes two codes with added features:

- gemm_comms_nccl.cu - GEMM/Triad/elementwise compute kernels 
that can be combined with a NCCL collective in an alternating
or overlapping manner. Launched with MPI, collectives use NCCL (>= 2.28).
- gemm_sleep.cu - GEMM iterations alternated with periods of 
host sleep.

Code can be built with `make -j`. Both binaries will accept the flags:

| Flag | Description | Default |
|------|-------------|---------|
| `-h` | Show help | — |
| `-d <int>` | GPU device ID | `0` |
| `-s <int>` | Matrix size | from `config.mk` |
| `-n <int>` | Number of iterations | from `config.mk` |
| `-i <mode>` | Initialisation: `random` or `constant` | `random` |

The NCCL benchmark also allows these flags:

| Flag | Description |
|------|-------------|
| `-m <size>` | Collective buffer size per rank  (`32M`, `1G`); `0` disables collectives |
| `-p <pattern>` | `allreduce`, `allgather`, `reducescatter`, `ce_allgather` |
| `-o 0\|1` | `0` = sequential, `1` = overlapped |
| `--barrier` | Rank sync each lap |
| `--comm-only` | Skip the GEMM, loop the collective |
| `--compute <m>` | Compute phase: `gemm`, `triad` or `eltwise`  |
| `--cpasses <n>` | Bandwidth-kernel launches per lap |
| `--cbytes <b>` | Bytes per working array (default 1 GiB) |

For the sleep code, `-m` determines the sleep duration in seconds.

Supported datatypes include fp64, fp32, tf32 and bf16. batch_scripts directory gives two example scripts for running on Helma.

## ML operation microbenchmark

Includes a Python benchmark measuring power draw of individual ML operations.
Loops one operation for a given duration.


```bash
python3 microbench.py --op bn_bwd --shape s1 --dtype bf16 \
    --seconds 60 --discard 15 --axis cap --axisval 300 --outdir <dir>
```

| Flag | Description |
|------|-------------|
| `--op` | One of `bn_fwd`, `bn_bwd`, `conv_fwd`, `conv_bwd`, `elementwise`, `pool`, `gemm`, `layernorm`, `sdpa` (fwd+bwd for the last four) |
| `--dtype` | `bf16`, `fp32`, `tf32`, `fp64` |
| `--batch` | Batch size  |
| `--seconds` | Sustained measurement window |
| `--discard` | Leading seconds discarded away before measurement |
| `--axis`/`--axisval` | Record which external knob (clock lock or power cap) a wrapper set on this run |
| `--outdir` | Output directory |


## Energy-measurement accuracy probe

`probe_energy_counter.cu` tests the update frequency of different NVML energy and power polling methods. 

```bash
nvcc -lnvidia-ml -o probe_energy_counter probe_energy_counter.cu
./probe_energy_counter
```

### AI usage disclaimer

AI-assisted tools, Claude Code and Pi coding agent with GLM-5.3, were used during the development of the codes.