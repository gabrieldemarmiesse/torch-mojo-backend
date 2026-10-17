#!/usr/bin/env bash
# Throwaway CI diagnostics: what the GPU runner's caches look like at job
# start and what this job added to them by the end.
#   gpu_cache_report.sh start|end
set -u
phase=$1
marker="$RUNNER_TEMP/cache-report-marker"
dirs=(
  "TORCH_MOJO_BACKEND_CACHE_DIR=${TORCH_MOJO_BACKEND_CACHE_DIR:-}"
  "MODULAR_CACHE_DIR=${MODULAR_CACHE_DIR:-}"
  "MODULAR_HOME=${MODULAR_HOME:-}"
  "UV_CACHE_DIR=${UV_CACHE_DIR:-}"
  "TMPDIR=${TMPDIR:-/tmp}"
)
echo "::group::cache report ($phase) on $RUNNER_NAME"
date -u +%FT%TZ
echo "HOME=$HOME TMPDIR=${TMPDIR:-} XDG_CACHE_HOME=${XDG_CACHE_HOME:-} uid=$(id -u) host=$(hostname)"
echo "MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-}"
echo "nproc=$(nproc) loadavg=$(cat /proc/loadavg)"
cat /sys/fs/cgroup/cpu.max 2>/dev/null | sed 's/^/cgroup cpu.max: /'
cat /sys/fs/cgroup/memory.max 2>/dev/null | sed 's/^/cgroup memory.max: /'
cat /proc/pressure/cpu 2>/dev/null | sed 's/^/cpu pressure: /'
free -g | head -2
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,clocks.sm --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | head -20
for kv in "${dirs[@]}"; do
  name=${kv%%=*}; d=${kv#*=}
  [ -n "$d" ] || continue
  echo "--- $name=$d"
  if [ ! -d "$d" ]; then echo "  (missing)"; continue; fi
  echo "  fs: $(df -h --output=source,fstype,size,used,avail,target "$d" | tail -1)"
  echo "  dev:inode=$(stat -c '%d:%i' "$d")"
  if [ "$name" != UV_CACHE_DIR ]; then
    echo "  files=$(find "$d" -type f 2>/dev/null | wc -l) size=$(du -sh "$d" 2>/dev/null | cut -f1)"
  fi
  if [ "$phase" = end ] && [ -f "$marker" ]; then
    new=$(find "$d" -type f -newer "$marker" 2>/dev/null | wc -l)
    echo "  new files this job: $new, $(find "$d" -type f -newer "$marker" -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {printf "%.1f MB", s/1e6}')"
  fi
done
mount | grep -E " /home| /tmp|cache" | head -20
if [ "$phase" = start ]; then
  touch "$marker"
  uv run --no-sync python - <<'PY'
from torch_mojo_backend import native
print("cache_dir:", native.cache_dir())
print("kernel_identity:", native.kernel_identity())
print("backend_source_hash:", native.backend_source_hash())
PY
  tmb="${TORCH_MOJO_BACKEND_CACHE_DIR:-}"
  if [ -d "$tmb" ]; then
    echo "--- newest 15 entries of $tmb"
    ls -lt --time-style=+%FT%T "$tmb" | head -16
    echo "--- entries per kind"
    ls "$tmb" | sed -E 's/\.hash-.*//; s/^(lib[a-z_]+|[a-z_]+)\..*/\1/' | sort | uniq -c | sort -rn | head -40
  fi
else
  tmb="${TORCH_MOJO_BACKEND_CACHE_DIR:-}"
  if [ -d "$tmb" ] && [ -f "$marker" ]; then
    echo "--- files this job added to $tmb (mtime size name)"
    find "$tmb" -maxdepth 2 -newer "$marker" -type f ! -name '.*.lock' \
      -printf '%TT %10s %P\n' 2>/dev/null | sort | head -300
  fi
  mc="${MODULAR_CACHE_DIR:-}"
  if [ -d "$mc" ] && [ -f "$marker" ]; then
    echo "--- files this job added under $mc, by top-level dir"
    find "$mc" -newer "$marker" -type f -printf '%P\n' 2>/dev/null | cut -d/ -f1-3 | sort | uniq -c | sort -rn | head -20
  fi
  echo "--- snapshot / scratch dirs in ${TMPDIR:-/tmp}"
  ls -d "${TMPDIR:-/tmp}"/torch-mojo-backend-* "${TMPDIR:-/tmp}"/modular-* 2>/dev/null | head
fi
echo "::endgroup::"
