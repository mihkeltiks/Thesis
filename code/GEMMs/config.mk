# ---------------------------------------------------------------------------
# GPU target — selects the mk/include_$(GPU).mk file, which sets the
# architecture flags and default problem sizes.
#
# NVIDIA (CUDA):  A40 | A100 | H100
# ---------------------------------------------------------------------------
GPU ?= H100

# ---------------------------------------------------------------------------
# Floating-point precision
#   DP   — double precision      (cublasDgemm)
#   SP   — single precision      (cublasSgemm)
# ---------------------------------------------------------------------------
PRECISION ?= DP

