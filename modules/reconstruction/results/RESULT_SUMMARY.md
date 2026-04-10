# Reconstruction Result Summary

- Total outputs: 8
- Results directory: `results`

## Outputs

| File | Type | Size (MB) | Iteration | Vertices |
| --- | --- | ---: | ---: | ---: |
| `amtown02_colmap_baseline_hq2_35000/scene_colmap_baseline_hq2_n35000_d2_sh3.ply` | ply | 630.06 | 35000 | 2663956 |
| `amtown02_vo_amtown02_10000/scene_gpu_safe_n10000_d3.ply` | ply | 128.58 | 10000 | 1296361 |
| `amtown02_vo_amtown02_20000/scene_gpu_safe_n20000_d3.ply` | ply | 132.80 | 20000 | 1338941 |
| `amtown02_vo_amtown02_30000/scene_gpu_safe_n30000_d3.ply` | ply | 190.70 | 30000 | 1922747 |
| `amtown02_vo_amtown02_35000/scene_gpu_safe_n35000_d3.ply` | ply | 193.34 | 35000 | 1949324 |
| `amtown02_vo_amtown02_40000/scene_gpu_safe_n40000_d3.ply` | ply | 179.29 | 40000 | 1807727 |
| `amtown02_vo_amtown02_aggressive_hq2_35000/scene_aggressive_hq2_n35000_d2_sh3.ply` | ply | 449.19 | 35000 | 1899228 |
| `amtown02_vo_amtown02_safe/scene_gpu_safe_n500_d3.ply` | ply | 4.96 | 500 | 50000 |

## Camera Files

| File | Camera Count |
| --- | ---: |
| `amtown02_colmap_baseline_hq2_35000/cameras.json` | 292 |
| `amtown02_vo_amtown02_10000/cameras.json` | 622 |
| `amtown02_vo_amtown02_20000/cameras.json` | 622 |
| `amtown02_vo_amtown02_30000/cameras.json` | 622 |
| `amtown02_vo_amtown02_35000/cameras.json` | 622 |
| `amtown02_vo_amtown02_40000/cameras.json` | 622 |
| `amtown02_vo_amtown02_aggressive_hq2_35000/cameras.json` | 622 |
| `amtown02_vo_amtown02_safe/cameras.json` | 622 |

## Recommended Files For Presentation

- `amtown02_vo_amtown02_aggressive_hq2_35000/scene_aggressive_hq2_n35000_d2_sh3.ply`: the selected VO-connected AMtown02 result (aggressive HQ2 OpenSplat profile).
- `amtown02_vo_amtown02_35000/scene_gpu_safe_n35000_d3.ply`: the best conservative-profile rerun for browser viewing and size tradeoffs.
- `amtown02_vo_amtown02_40000/scene_gpu_safe_n40000_d3.ply`: the longest conservative rerun (viewer compatibility varies).
- `amtown02_colmap_baseline_hq2_35000/scene_colmap_baseline_hq2_n35000_d2_sh3.ply`: **V7** — aggressive HQ2 with **changed** COLMAP input (`colmap_from_amtown02`, **292** registered cameras in this snapshot).
- `amtown02_vo_amtown02_safe/scene_gpu_safe_n500_d3.ply`: smoke test for wiring and CUDA.

## Notes

- Conservative VO reruns use `baseline/colmap_from_vo_amtown02` with the stable low-SH / heavy-downscale preset.
- Aggressive HQ2 uses the **same** `colmap_from_vo_amtown02` registration (**622** cameras) as the conservative line; only OpenSplat hyperparameters differ.
- FAQ: OpenSplat only consumes cameras in `images.bin`; extra PNG files on disk are ignored until registered in COLMAP (this is not what changed between conservative and aggressive VO runs).
