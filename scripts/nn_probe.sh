#!/bin/bash
# Per-case NN timings, one process per case.  $1 = binary
B=${1:-/tmp/bench_gemm}
T=harness/nanogpt_train/rocm_gemm_targets.csv
for c in attn_c_attn_dgrad attn_c_proj_dgrad mlp_c_fc_dgrad mlp_c_proj_dgrad lm_head_dgrad; do
  $B --targets=$T --case=$c "${@:2}" 2>&1 | sed -n 2p | awk '{printf "%-20s %10s %8s %8s  %s\n", $1, $8, $10, $11, $13}'
done
