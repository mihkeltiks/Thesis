#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --exclusive
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --reservation=...

# Representative example of power cap sweep on the GEMM + NCCL benchmark
# bf16 32k GEMM + allgather, for each power cap x message size
# serial and overlapped+barrier. 3 reps per cell.

REPO_DIR=...
cd $REPO_DIR

module purge
module load cuda/13.0.2
module load openmpi/5.0.5-nvhpc24.11-cuda
module load nccl

BIN="gemm_comms_nccl_bf16"

CAPS="200 300 400 500 600"
REPS="r1 r2 r3"
SIZES="32M 128M"                 # collective buffer per lap per rank (-m)
N=60                             # laps per run (~6 s at 600 W; longer at 200 W)

sudo nvidia-smi --reset-gpu-clocks 

for cap in $CAPS; do
  sudo nvidia-smi --power-limit=$cap 
  for m in $SIZES; do
    for r in $REPS; do
      mpirun -n 4 "$BIN" -s 32768 -n $N -m $m -p allreduce -o 0           >> "serial_${m}_cap${cap}_${r}.log"
      mpirun -n 4 "$BIN" -s 32768 -n $N -m $m -p allreduce -o 1 --barrier >> "overlap_${m}_cap${cap}_${r}.log"
    done
  done
done

echo "ALL DONE ($(date +%H:%M:%S))"
