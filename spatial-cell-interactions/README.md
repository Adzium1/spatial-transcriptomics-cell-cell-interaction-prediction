# Spatial Cell Interactions

Distance-aware Graph Attention Networks (GATv2) for predicting cell-cell interactions from 10x Genomics Visium data. The pipeline ingests Space Ranger `outs/`, builds a spatial neighbor graph with distance encodings, trains a self-supervised edge reconstruction model, and exports ranked edges, ligand-receptor summaries.

## Method overview
- Load Visium expression + spatial metadata via `scanpy.read_visium`.
- QC: in-tissue spot filter, lowly expressed gene filter, total-count normalize to 1e4, log1p, and select highly variable genes.
- Build a spatial graph (kNN or radius) with Gaussian RBF distance features on each edge.
- Encode nodes with a distance-aware GATv2 (arXiv:2105.14491) and train via self-supervised edge reconstruction with negative sampling and validation AP early stopping.
- Score edges and (optionally) compute ligand-receptor enrichment, then render spatial edge overlays and UMAP embeddings.

## Data
Assumes a standard Space Ranger `outs/` directory containing:
- `filtered_feature_bc_matrix.h5`
- `spatial/` with tissue positions (`tissue_positions.parquet` or `tissue_positions.csv`), `scalefactors_json.json`, and associated images.

`tissue_positions_list.csv` is regenerated verbatim from the native parquet/CSV if present; avoid hand-edited conversions that rescale or swap axes.

Loading relies on `scanpy.read_visium(path=..., count_file="filtered_feature_bc_matrix.h5", load_images=True)`. If files are missing, the CLI will raise a descriptive error.

Place datasets under `data/external/<sample>/outs/` (ignored by git). Processed artifacts are written to `data/processed/`.

## Quickstart
Install dependencies (Python 3.10+). Ensure you install a PyTorch + PyTorch Geometric wheel that matches your CUDA/CPU setup (see https://pytorch.org/get-started/ and https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html):
```bash
pip install -e .
```

CPU-friendly quickstart:
```bash
python scripts/01_prepare_data.py --visium_path data/external/visium_sample/outs/ --config configs/quickstart_cpu.yaml
python scripts/02_build_graph.py --h5ad data/processed/visium_sample.h5ad --config configs/quickstart_cpu.yaml
python scripts/03_train_ssl.py --graph data/processed/visium_sample_graph.pt --config configs/quickstart_cpu.yaml --out_dir results/run_cpu
python scripts/04_make_figures.py --h5ad data/processed/visium_sample.h5ad --graph data/processed/visium_sample_graph.pt --checkpoint results/run_cpu/checkpoints/best_model.pt --config configs/quickstart_cpu.yaml --out_dir results/run_cpu
```

One-shot run:
```bash
python scripts/05_run_end_to_end.py --visium_path data/external/visium_sample/outs/ --run_name demo --config configs/default.yaml
```

## Outputs
A completed run (`results/<run_name>/`) contains:
```
results/<run_name>/
├── checkpoints/
│   ├── best_model.pt
│   └── node_embeddings.pt
├── figures/
│   ├── spatial_interaction_edges.png
│   └── umap_embeddings.png
├── tables/
│   ├── top_edges.csv
│   └── top_lr_pairs.csv   # if lr_pairs.csv is available
└── metrics.json
```

Example figures (generated after running):
![Spatial edges](results/figures/spatial_interaction_edges.png)
![UMAP embeddings](results/figures/umap_embeddings.png)

## Model
- **Encoder:** Distance-aware GATv2 with edge attributes injected into attention (`edge_dim=RBF_dim`), 2-3 layers, ELU activations, dropout.
- **Objective:** Self-supervised edge reconstruction using observed spatial edges as positives and uniform negative sampling over non-edges. Loss is BCE on concatenated `[z_i, z_j, edge_attr_ij]` through an MLP head. Early stopping monitors validation average precision.
- **Graph construction:** kNN (default k=8) or radius-based edges on spot coordinates, with Gaussian RBF distance embeddings (default 16 channels).
- **Biology add-on:** Optional ligand-receptor ranking when `data/lr_db/lr_pairs.csv` (columns: `ligand,receptor`) is provided; reports pairs enriched among top predicted edges.
- **Troubleshooting: misaligned spots & histology:** 10x defines `pxl_row_in_fullres` as y and `pxl_col_in_fullres` as x. If spots and tissue image do not overlap, regenerate `tissue_positions_list.csv` directly from the native parquet/CSV (no scaling, no offsets) or run `python scripts/06_fix_spatial_alignment.py --outs_path data/external/<sample>/outs`. `crop_coord` in `sc.pl.spatial` is in pixel space. Then regenerate figures via `scripts/07_make_pretty_spatial_figures.py`.

## Demo (CytAssist breast) and Quickstart

This repo ships with a reproducible demo using the CytAssist FFPE breast sample (10x public dataset). Paths below assume repo root.

### One-shot Quickstart
```bash
# 1) Download Visium-formatted outs
mkdir -p data/external/breast_cytassist_ffpe/outs
cd data/external/breast_cytassist_ffpe/outs
curl -L -o filtered_feature_bc_matrix.h5 https://cf.10xgenomics.com/samples/spatial-exp/2.1.0/CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer/CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer_filtered_feature_bc_matrix.h5
curl -L -o spatial.tar.gz https://cf.10xgenomics.com/samples/spatial-exp/2.1.0/CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer/CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer_spatial.tar.gz
tar -xzf spatial.tar.gz   # creates outs/spatial/*
cd ../../../..

# Windows note: if curl errors on cert revocation, use `curl.exe --ssl-no-revoke -L -o ...`

# 2) Prepare AnnData
python spatial-cell-interactions/scripts/01_prepare_data.py \
  --visium_path data/external/breast_cytassist_ffpe/outs \
  --count_file filtered_feature_bc_matrix.h5 \
  --out_h5ad data/processed/breast_cytassist_ffpe.h5ad \
  --filter_in_tissue 1 --min_spots_frac 0.001 --n_hvg 2000

# 3) Build radius graph (pixel units)
python spatial-cell-interactions/scripts/02_build_graph.py \
  --h5ad data/processed/breast_cytassist_ffpe.h5ad \
  --out_graph data/processed/breast_cytassist_ffpe_radius_graph.pt \
  --graph_type radius --distance_unit pixel --radius auto --rbf_dim 16

# 4) Train SSL model (GATv2 self-supervised on graph)
python spatial-cell-interactions/scripts/03_train_ssl.py \
  --graph data/processed/breast_cytassist_ffpe_radius_graph.pt \
  --out_dir results/run_breast_ssl \
  --config spatial-cell-interactions/configs/default.yaml \
  --device cpu

# 5) Generate figures (examples below), or use the saved demo outputs in results/figures/
```

### Canonical demo outputs (for README/slides)
- `results/figures/breast_in_tissue_lowres_cropped.png`
- `results/figures/breast_total_counts_hires_cropped.png`
- `results/figures/breast_radius_graph_spots_only.png`
- `results/figures/breast_umap_leiden.png`
- `results/figures/breast_spatial_leiden_lowres_cropped.png`

All spatial plots use `crop_coord=(left,right,top,bottom)` in pixel space to avoid whitespace; `img_key` can be set to `none` for spots-only overlays if desired.

### Graph format (saved .pt)
Graphs are PyTorch Geometric `Data` objects with:
- `x`: node features (shape [n_nodes, n_hvgs])
- `pos`: spatial coordinates (pixels)
- `edge_index`: [2, n_edges]
- `edge_attr`: RBF-encoded distances (default dim 16)
- `obs_names`: node barcodes
- `distances`, `rbf_centers`: distance scalars used for encoding

Loading uses `torch.serialization.add_safe_globals` to allow these PyG classes; see `scripts/03_train_ssl.py`.

## Reproducibility
- Deterministic seeding across Python, NumPy, and PyTorch (`seed` in configs).
- Config-driven hyperparameters (`configs/*.yaml`) with CLI overrides.
- Checkpoints and embeddings saved under `results/<run_name>/checkpoints/`.

## Limitations
- Visium spots aggregate multiple cells; predicted “interactions” are spot-level proxies.
- Edge reconstruction is a structural surrogate and does not guarantee true ligand-receptor engagement; biological validation is required.
- Ligand-receptor enrichment depends on gene detection and provided LR catalogs; sparse expression can limit interpretability.

## Cite
If you use this repository, please cite:
```bibtex
@misc{spatialcellinteractions2025,
  title        = {Spatial Cell Interactions: Distance-aware GATv2 for Visium},
  author       = {Spatial Cell Interactions Developers},
  year         = {2025},
  note         = {https://github.com/your-org/spatial-cell-interactions}
}
```
- Graph defaults use a radius graph. When scalefactors allow, coordinates are converted to microns using spot_diameter_fullres (approximate per 10x guidance); otherwise pixel units are used with an auto radius heuristic (1.5× median nearest-neighbor distance, clamped).
