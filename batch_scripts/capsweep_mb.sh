#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --exclusive
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Representative power-cap sweep of the ML-op microbenchmark: one sustained
# run per op x dtype x power cap, on each GPU of the node. bf16 and tf32.

REPO_DIR=...
cd $REPO_DIR

module purge
module load cuda/13.0.2
# module load python/Apptainer or something else for torch


BIN="microbench.py"

CAPS="200 300 400 500 600"
OPS="bn_fwd bn_bwd conv_fwd conv_bwd elementwise pool gemm layernorm sdpa"
DTYPES="bf16 tf32"
SECS=60                 # sustained seconds per run
DISCARD=15              # leading seconds thrown away (clock settles)

sudo nvidia-smi --reset-gpu-clocks 

for cap in $CAPS; do
  sudo nvidia-smi --power-limit=$cap 
  for dt in $DTYPES; do
    for op in $OPS; do
      for gpu in 0 1 2 3; do
        out="mb_${op}_${dt}_cap${cap}_gpu${gpu}"
        CUDA_VISIBLE_DEVICES=$gpu \
          python3 "$BIN" --op "$op" --dtype "$dt" \
            --seconds $SECS --discard $DISCARD \
            --axis cap --axisval $cap \
            --outdir "$out" >> "$out.log"
      done
    done
  done
done

echo "ALL DONE ($(date +%H:%M:%S))"
