#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --exclusive
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --reservation=...


# Representative example of power cap sweep on the GEMM + sleep benchmark
# bf16 16k GEMM laps with a host sleep in between, for each power cap
# x sleep duration, on each GPU of the node. 3 reps per cell.

REPO_DIR=...
cd $REPO_DIR

module purge
module load cuda/13.0.2
module load openmpi/5.0.5-nvhpc24.11-cuda
module load nccl

BIN="gemm_sleep_bf16"

CAPS="200 300 400 500 600"
REPS="r1 r2 r3"
LENS="0 0.05 0.1 0.25 0.5 1.0"   # seconds of sleep per lap (0 = GEMM only)
N=20                             # laps per run (GEMM ~0.12 s + sleep each)

sudo nvidia-smi --reset-gpu-clocks 

for cap in $CAPS; do
  sudo nvidia-smi --power-limit=$cap 
  for len in $LENS; do
    for gpu in 0 1 2 3; do
      for r in $REPS; do
        ./"$BIN" -s 16384 -n $N -m $len -d $gpu >> "sleep_d${len}s_cap${cap}_gpu${gpu}_${r}.log"
      done
    done
  done
done

echo "ALL DONE ($(date +%H:%M:%S))"
