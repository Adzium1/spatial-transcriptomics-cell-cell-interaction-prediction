# Spatial cell–cell interaction prediction with graph SSL (CytAssist breast)

**Dataset:** 10x CytAssist FFPE Protein Expression Human Breast Cancer (spots=4169, genes=18085).
**Graph:** radius auto -> ~462 px (median NN ~308 px), edges ~24150.
**Training:** CPU, early stop epoch 19, val AUROC 0.914, val AP 0.890.

## Figures
- breast_total_counts_hires_cropped.png
- breast_in_tissue_lowres_cropped.png
- breast_radius_graph_spots_only.png
- breast_umap_leiden.png
- breast_spatial_leiden_lowres_cropped.png

## Supervised interaction metrics (test)
- LR (SSL features): AUROC 0.9769809411367852, AP 0.9202200856705721
- Immune–epithelial regression: Spearman 0.8929744092622207, top-k overlap 0.778

## Reproducibility (CPU)
```bash
mkdir -p data/external/breast_cytassist_ffpe/outs
cd data/external/breast_cytassist_ffpe/outs
curl -L -o filtered_feature_bc_matrix.h5 https://cf.10xgenomics.com/samples/spatial-exp/2.1.0/CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer/CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer_filtered_feature_bc_matrix.h5
curl -L -o spatial.tar.gz https://cf.10xgenomics.com/samples/spatial-exp/2.1.0/CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer/CytAssist_FFPE_Protein_Expression_Human_Breast_Cancer_spatial.tar.gz
tar -xzf spatial.tar.gz
cd ../../../..

python spatial-cell-interactions/scripts/01_prepare_data.py --visium_path data/external/breast_cytassist_ffpe/outs --count_file filtered_feature_bc_matrix.h5 --out_h5ad data/processed/breast_cytassist_ffpe.h5ad --filter_in_tissue 1 --min_spots_frac 0.001 --n_hvg 2000
python spatial-cell-interactions/scripts/02_build_graph.py --h5ad data/processed/breast_cytassist_ffpe.h5ad --out_graph data/processed/breast_cytassist_ffpe_radius_graph.pt --graph_type radius --distance_unit pixel --radius auto --rbf_dim 16
python spatial-cell-interactions/scripts/03_train_ssl.py --graph data/processed/breast_cytassist_ffpe_radius_graph.pt --out_dir results/run_breast_ssl --config spatial-cell-interactions/configs/default.yaml --device cpu
```

## Notes
- Metrics reflect adjacency reconstruction, not ligand–receptor validation.
- See requirements.txt for package versions.