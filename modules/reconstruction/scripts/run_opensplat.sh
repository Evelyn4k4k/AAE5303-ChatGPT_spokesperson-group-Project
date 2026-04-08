#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPEN_SPLAT_BIN="$ROOT_DIR/opensplat_cpu_src/build_cuda/opensplat"
RESULTS_DIR="$ROOT_DIR/results"

print_usage() {
  cat <<'EOF'
Usage:
  ./run_opensplat.sh <preset>

Presets:
  amtown02_quality
  amtown02_compact
  vo_safe

Examples:
  ./run_opensplat.sh amtown02_quality
  ./run_opensplat.sh vo_safe
EOF
}

main() {
  local preset="${1:-}"

  if [[ -z "$preset" ]]; then
    print_usage
    exit 1
  fi

  if [[ ! -x "$OPEN_SPLAT_BIN" ]]; then
    echo "OpenSplat binary not found: $OPEN_SPLAT_BIN" >&2
    exit 1
  fi

  case "$preset" in
    amtown02_quality)
      run_amtown02_quality
      ;;
    amtown02_compact)
      run_amtown02_compact
      ;;
    vo_safe)
      run_vo_safe
      ;;
    *)
      print_usage
      exit 1
      ;;
  esac
}

run_amtown02_quality() {
  mkdir -p "$RESULTS_DIR"
  "$OPEN_SPLAT_BIN" \
    "$ROOT_DIR/baseline/colmap_from_amtown02" \
    -n 6000 \
    -o "$RESULTS_DIR/amtown02_quality_balanced_6000.ply"
}

run_amtown02_compact() {
  mkdir -p "$RESULTS_DIR"
  "$OPEN_SPLAT_BIN" \
    "$ROOT_DIR/baseline/colmap_from_amtown02" \
    -n 2000 \
    -o "$RESULTS_DIR/amtown02_gpu_final_2000.splat"
}

run_vo_safe() {
  mkdir -p "$RESULTS_DIR/amtown02_vo_safe"
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/home/wym/libtorch-cu118-cxx11/lib:${LD_LIBRARY_PATH:-}"
  unset CUBLAS_WORKSPACE_CONFIG
  export CUDA_VISIBLE_DEVICES=0
  export LD_PRELOAD="/home/wym/libtorch-cu118-cxx11/lib/libcublas-3b81d170.so.11:/home/wym/libtorch-cu118-cxx11/lib/libcublasLt-b6d14a74.so.11:/home/wym/libtorch-cu118-cxx11/lib/libcudart-d0da41ae.so.11.0"

  "$OPEN_SPLAT_BIN" \
    "$ROOT_DIR/baseline/colmap_from_vo" \
    -n 500 \
    -d 3 \
    --num-downscales 4 \
    --resolution-schedule 1200 \
    --sh-degree 1 \
    -o "$RESULTS_DIR/amtown02_vo_safe/scene_gpu_safe_n500_d3.ply"
}

main "$@"
