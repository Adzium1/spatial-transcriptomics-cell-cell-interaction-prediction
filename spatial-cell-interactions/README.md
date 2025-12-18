# Spatial graph self-supervised learning for Visium/CytAssist
Spatial graph self-supervised learning pipeline (GATv2) for modeling cell–cell neighborhood structure from 10x Visium/CytAssist spatial transcriptomics.

## What this repo does
Build a spatial graph (spots as nodes, spatial proximity as edges) from Space Ranger `outs/`, then train a self-supervised GNN (link prediction / edge reconstruction) to learn embeddings that capture local spatial context. Includes scripts to generate figures and an optional preprint-style report.

## Demo dataset (recommended)
10x CytAssist FFPE Protein Expression Human Breast Cancer (Space Ranger-style).  
Loaded size: **4,169 spots × 18,085 genes**.

## Quickstart (CPU)
> Note: Dataset is large; download the two files from the 10x dataset page (browser/manual if links change).

0) Setup
```bash
pip install -r requirements.txt
```

1) Download and stage the dataset
```bash
mkdir -p data/external/breast_cytassist_ffpe/outs
cd data/external/breast_cytassist_ffpe/outs
# Download manually from 10x then place/rename:
#   *_filtered_feature_bc_matrix.h5  -> filtered_feature_bc_matrix.h5
#   *_spatial.tar.gz                 -> spatial.tar.gz
tar -xzf spatial.tar.gz   # extracts spatial/ with tissue images, scalefactors, positions
cd ../../../..
```

Expected layout:
```
data/external/breast_cytassist_ffpe/outs/
  filtered_feature_bc_matrix.h5
  spatial/
    tissue_hires_image.png, tissue_lowres_image.png
    scalefactors_json.json
    tissue_positions*.csv
    ...
```

2) Prepare AnnData
```bash
python spatial-cell-interactions/scripts/01_prepare_data.py \
  --visium_path data/external/breast_cytassist_ffpe/outs \
  --count_file filtered_feature_bc_matrix.h5 \
  --out_h5ad data/processed/breast_cytassist_ffpe.h5ad \
  --filter_in_tissue 1 \
  --min_spots_frac 0.001 \
  --n_hvg 2000
```

3) Build pixel-space radius graph
```bash
python spatial-cell-interactions/scripts/02_build_graph.py \
  --h5ad data/processed/breast_cytassist_ffpe.h5ad \
  --out_graph data/processed/breast_cytassist_ffpe_radius_graph.pt \
  --graph_type radius \
  --distance_unit pixel \
  --radius auto \
  --rbf_dim 16
```

4) Train SSL (link prediction / edge reconstruction)
```bash
python spatial-cell-interactions/scripts/03_train_ssl.py \
  --graph data/processed/breast_cytassist_ffpe_radius_graph.pt \
  --out_dir results/run_breast_ssl \
  --config spatial-cell-interactions/configs/default.yaml \
  --device cpu
```

5) Figures
Use the figure scripts (e.g., `07_make_pretty_spatial_figures.py`) or rely on the provided demo figures in `results/figures/` (cropped to avoid whitespace).

## Expected demo outputs
- Graph: pixel-space **radius** graph; auto radius ~462 px (median NN ~308 px); ~24,150 edges; edge RBF dim 16.
- SSL training (`03_train_ssl.py`, CPU): early stop ~epoch 19; validation AUROC ~0.914, AP ~0.890 (link prediction / edge reconstruction).

“Blessed” figures (under `results/figures/`):
- `breast_total_counts_hires_cropped.png`
- `breast_in_tissue_lowres_cropped.png`
- `breast_radius_graph_spots_only.png`
- `breast_umap_leiden.png`
- `breast_spatial_leiden_lowres_cropped.png`

## Repository layout (current)
- `.gitignore` — excludes generated data/results artifacts; keeps code and documentation.
- `requirements.txt` — core Python dependencies for preprocessing, graph construction, and SSL training.

### Generated artifacts (gitignored; created locally)
- `data/external/breast_cytassist_ffpe/outs/` — 10x CytAssist FFPE breast cancer Space Ranger outputs (counts + `spatial/`).
- `data/processed/breast_cytassist_ffpe.h5ad` — processed AnnData.
- `data/processed/breast_cytassist_ffpe_radius_graph.pt` — PyTorch/PyG graph object saved via torch (`x`, `pos`, `edge_index`, `edge_attr`, …); radius ~462 px, ~24,150 edges, RBF dim 16.
- `results/figures/` — demo figures (see above).
- `results/run_breast_ssl/` — SSL training outputs (`config_used.yaml`, logs, checkpoints).
- `results/smoke_test/` — CI smoke test output (1 epoch CPU run when demo data is present).

### Main codebase
- `spatial-cell-interactions/`
  - `configs/` — YAML configs (e.g., `default.yaml`).
  - `scripts/`
    - `01_prepare_data.py` — build `.h5ad` from 10x `outs/`.
    - `02_build_graph.py` — kNN/radius graph builder.
    - `03_train_ssl.py` — GATv2 self-supervised training (PyG load patch applied).
    - `04_make_figures.py`, `07_make_pretty_spatial_figures.py` — figure generation.
    - `05_run_end_to_end.py` — end-to-end runner.
    - `06_fix_spatial_alignment.py` — alignment utility for problematic Visium HD examples.
    - `ci_smoke_test.py` — 1-epoch CPU smoke test if demo data present.
  - `spatial_interactions/` — library package (preprocessing, graph, training, models, utils, visualization).
  - `tests/`, `notebooks/` — development utilities.

### Preprint package
- `reports/preprint_breast_ssl/`
  - `manuscript.tex`, `manuscript.pdf`, `manuscript.md`
  - `figures/` (copied demo images)
 - `references.bib`
 - `build_preprint.py` (assembles report; runs `pdflatex` if available)
  - `README.md` (rebuild instructions)

## Supervised interaction modeling (LR + cell-type proxies)
Labels and training are proxy tasks (expression/marker based), not ground-truth biology.

1) Build labels (LR + cell-type edges):
```bash
python spatial-cell-interactions/scripts/08_build_interaction_labels.py \
  --h5ad data/processed/breast_cytassist_ffpe.h5ad \
  --graph data/processed/breast_cytassist_ffpe_radius_graph.pt
```
Outputs to `data/processed/supervised_labels/`:
- `lr_edge_labels.parquet` (edge_id, src, dst, lr_score, y_lr, split)
- `celltype_node_labels.parquet` (obs_name, cell_type, per-type scores)
- `celltype_edge_labels_binary.parquet` (immune↔epithelial binary)
- `celltype_edge_labels_multiclass.parquet` (directed type pairs; rare classes → Other)

2) Train supervised edge classifiers (uses SSL embeddings if available, falls back to PCA):
```bash
python spatial-cell-interactions/scripts/09_train_supervised_edges.py \
  --h5ad data/processed/breast_cytassist_ffpe.h5ad \
  --graph data/processed/breast_cytassist_ffpe_radius_graph.pt \
  --labels_dir data/processed/supervised_labels \
  --out_dir results/run_breast_supervised \
  --device cpu
```
Outputs: `preds_lr.parquet`, `preds_immune_epi.parquet`, optional `preds_type_pair.parquet`, plus `metrics.json`.

3) Plot supervised predictions:
```bash
python spatial-cell-interactions/scripts/10_plot_supervised_interactions.py \
  --h5ad data/processed/breast_cytassist_ffpe.h5ad \
  --preds_lr results/run_breast_supervised/preds_lr.parquet \
  --preds_immune_epi results/run_breast_supervised/preds_immune_epi.parquet
```
Generates `results/figures_supervised/` (top-K edge maps, ROC/PR, LR score vs prob).

### SSL vs PCA ablation (test split)
| Task | Features | Metric(s) |
| --- | --- | --- |
| LR edge (binary) | SSL embeddings | AUROC 0.977, AP 0.920 |
| LR edge (binary) | PCA (expression) | AUROC 0.997, AP 0.987 |
| Immune–epithelial (regression) | SSL embeddings | Spearman 0.893, top-k overlap 0.778 |
| Immune–epithelial (regression) | PCA (expression) | Spearman 0.917, top-k overlap 0.824 |
PCA baseline matches/exceeds SSL on these proxy labels; improving SSL utility is future work. Binary immune–epithelial is highly imbalanced (~95% positives) and secondary; type_pair is exploratory.

## Notes / limitations
- The SSL objective measures link prediction / edge reconstruction (AUROC/AP) on the spatial graph; it is not a direct biological interaction ground truth.
- Some Visium HD “tiny” developer datasets are intentionally corner-cropped/edited and can produce misleading image-backed plots; the CytAssist breast dataset is recommended for clean figures.
