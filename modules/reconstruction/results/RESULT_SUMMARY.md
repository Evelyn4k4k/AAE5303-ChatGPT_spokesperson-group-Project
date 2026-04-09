# Reconstruction Result Summary

- Total outputs: 6
- Results directory: `/home/wym/AAE5303-ChatGPT_spokesperson-group-Project/modules/reconstruction/results`

## Outputs

| File | Type | Size (MB) | Iteration | Vertices |
| --- | --- | ---: | ---: | ---: |
| `amtown02_vo_amtown02_10000/scene_gpu_safe_n10000_d3.ply` | ply | 128.58 | 10000 | 1296361 |
| `amtown02_vo_amtown02_20000/scene_gpu_safe_n20000_d3.ply` | ply | 132.80 | 20000 | 1338941 |
| `amtown02_vo_amtown02_30000/scene_gpu_safe_n30000_d3.ply` | ply | 190.70 | 30000 | 1922747 |
| `amtown02_vo_amtown02_35000/scene_gpu_safe_n35000_d3.ply` | ply | 193.34 | 35000 | 1949324 |
| `amtown02_vo_amtown02_40000/scene_gpu_safe_n40000_d3.ply` | ply | 179.29 | 40000 | 1807727 |
| `amtown02_vo_amtown02_safe/scene_gpu_safe_n500_d3.ply` | ply | 4.96 | 500 | 50000 |

## Camera Files

| File | Camera Count |
| --- | ---: |
| `amtown02_vo_amtown02_10000/cameras.json` | 622 |
| `amtown02_vo_amtown02_20000/cameras.json` | 622 |
| `amtown02_vo_amtown02_30000/cameras.json` | 622 |
| `amtown02_vo_amtown02_35000/cameras.json` | 622 |
| `amtown02_vo_amtown02_40000/cameras.json` | 622 |
| `amtown02_vo_amtown02_safe/cameras.json` | 622 |

## Recommended Files For Presentation

- `amtown02_vo_amtown02_35000/scene_gpu_safe_n35000_d3.ply`: the selected corrected AMtown02 rerun for GitHub-facing presentation.
- `amtown02_vo_amtown02_30000/scene_gpu_safe_n30000_d3.ply`: a viewer-friendly corrected rerun kept for side-by-side comparison.
- `amtown02_vo_amtown02_40000/scene_gpu_safe_n40000_d3.ply`: the raw longest corrected rerun kept for comparison and debugging.
- `amtown02_vo_amtown02_safe/scene_gpu_safe_n500_d3.ply`: the smoke-test rerun proving that the corrected AMtown02 input is wired correctly.

## Notes

- This summary lists the corrected reruns generated from `baseline/colmap_from_vo_amtown02`.
- The GitHub-facing result line for this module is `safe -> 10000 -> 20000 -> 30000 -> 35000 -> 40000`.
