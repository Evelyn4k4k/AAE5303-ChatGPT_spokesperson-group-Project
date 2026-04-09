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
  corrected_safe
  corrected_30000
  corrected_35000
  corrected_40000

Examples:
  ./run_opensplat.sh corrected_safe
  ./run_opensplat.sh corrected_35000
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
    corrected_safe)
      run_corrected_safe
      ;;
    corrected_30000)
      run_corrected_30000
      ;;
    corrected_35000)
      run_corrected_35000
      ;;
    corrected_40000)
      run_corrected_40000
      ;;
    *)
      print_usage
      exit 1
      ;;
  esac
}

run_corrected_safe() {
  mkdir -p "$RESULTS_DIR/amtown02_vo_amtown02_safe"
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/home/wym/libtorch-cu118-cxx11/lib:${LD_LIBRARY_PATH:-}"
  unset CUBLAS_WORKSPACE_CONFIG
  export CUDA_VISIBLE_DEVICES=0
  export LD_PRELOAD="/home/wym/libtorch-cu118-cxx11/lib/libcublas-3b81d170.so.11:/home/wym/libtorch-cu118-cxx11/lib/libcublasLt-b6d14a74.so.11:/home/wym/libtorch-cu118-cxx11/lib/libcudart-d0da41ae.so.11.0"

  "$OPEN_SPLAT_BIN" \
    "$ROOT_DIR/baseline/colmap_from_vo_amtown02" \
    -n 500 \
    -d 3 \
    --num-downscales 4 \
    --resolution-schedule 1200 \
    --sh-degree 1 \
    -o "$RESULTS_DIR/amtown02_vo_amtown02_safe/scene_gpu_safe_n500_d3.ply"
}

run_corrected_30000() {
  mkdir -p "$RESULTS_DIR/amtown02_vo_amtown02_30000"
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/home/wym/libtorch-cu118-cxx11/lib:${LD_LIBRARY_PATH:-}"
  unset CUBLAS_WORKSPACE_CONFIG
  export CUDA_VISIBLE_DEVICES=0
  export LD_PRELOAD="/home/wym/libtorch-cu118-cxx11/lib/libcublas-3b81d170.so.11:/home/wym/libtorch-cu118-cxx11/lib/libcublasLt-b6d14a74.so.11:/home/wym/libtorch-cu118-cxx11/lib/libcudart-d0da41ae.so.11.0"

  "$OPEN_SPLAT_BIN" \
    "$ROOT_DIR/baseline/colmap_from_vo_amtown02" \
    -n 30000 \
    -d 3 \
    --num-downscales 4 \
    --resolution-schedule 1200 \
    --sh-degree 1 \
    -o "$RESULTS_DIR/amtown02_vo_amtown02_30000/scene_gpu_safe_n30000_d3.ply"
}

run_corrected_35000() {
  mkdir -p "$RESULTS_DIR/amtown02_vo_amtown02_35000"
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/home/wym/libtorch-cu118-cxx11/lib:${LD_LIBRARY_PATH:-}"
  unset CUBLAS_WORKSPACE_CONFIG
  export CUDA_VISIBLE_DEVICES=0
  export LD_PRELOAD="/home/wym/libtorch-cu118-cxx11/lib/libcublas-3b81d170.so.11:/home/wym/libtorch-cu118-cxx11/lib/libcublasLt-b6d14a74.so.11:/home/wym/libtorch-cu118-cxx11/lib/libcudart-d0da41ae.so.11.0"

  "$OPEN_SPLAT_BIN" \
    "$ROOT_DIR/baseline/colmap_from_vo_amtown02" \
    -n 35000 \
    -d 3 \
    --num-downscales 4 \
    --resolution-schedule 1200 \
    --sh-degree 1 \
    -o "$RESULTS_DIR/amtown02_vo_amtown02_35000/scene_gpu_safe_n35000_d3.ply"
}

run_corrected_40000() {
  mkdir -p "$RESULTS_DIR/amtown02_vo_amtown02_40000"
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/home/wym/libtorch-cu118-cxx11/lib:${LD_LIBRARY_PATH:-}"
  unset CUBLAS_WORKSPACE_CONFIG
  export CUDA_VISIBLE_DEVICES=0
  export LD_PRELOAD="/home/wym/libtorch-cu118-cxx11/lib/libcublas-3b81d170.so.11:/home/wym/libtorch-cu118-cxx11/lib/libcublasLt-b6d14a74.so.11:/home/wym/libtorch-cu118-cxx11/lib/libcudart-d0da41ae.so.11.0"

  "$OPEN_SPLAT_BIN" \
    "$ROOT_DIR/baseline/colmap_from_vo_amtown02" \
    -n 40000 \
    -d 3 \
    --num-downscales 4 \
    --resolution-schedule 1200 \
    --sh-degree 1 \
    -o "$RESULTS_DIR/amtown02_vo_amtown02_40000/scene_gpu_safe_n40000_d3.ply"
}

main "$@"
