# Integrating knowledge graph and LLM embedding for uncertainty-aware chemical-disease association prediction
## Overview
fusionEDA is developed for chemical-disease association prediction based on a large-scale heterogeneous knowledge graph (KG), which contains 1,084,706 nodes and 5,398,828 edges. 
Based on this architecture, we further propose fusionEDA_EDL by incorporating an evidential layer to enable chemical-disease association prediction and uncertainty quantification.
<p align="center">
<img src="fig/framework.png"
     alt="overview"
     width="600px" />
</p>

## Table of Contents
- [Environment Setup](#Environment-Setup) 
- [Data Download](#Data-Download)
- [Pretrain and Finetune](#Pretrain-and-Finetune)
- [Citation](#Citation)

## Environment Setup
### Clone the repository:
```bash
git clone https://github.com/lZhao-Git/fusionEDA.git
cd fusionEDA
```

### Using conda or pip
```bash
conda env create -f fusionEDA.yml
conda activate fusionEDA
pip install -r requirements.txt
```

## Data Download
- **KG**: Data for the different nodes and edges types useds to construct the KG can be downloaded from Zenodo: [**kg.rar**](https://zenodo.org/api/records/22637711/draft/files/kg.rar/content)

- **Feature extraction**: The code required to generate pretrained embeddings for different nodes can be found at feature/pretrained_feature.py. Alternatively, the precomputed node embedding can be downloaded directly from Zenodo: [**multimodal.rar**](https://zenodo.org/api/records/22637711/draft/files/multimodal.rar/content)
## Pretrain and Finetune
- **Pretrain**: The pretraining script is executed from the command line using ```bash python train.py```
- **Finetune**: The finetune script is executed from the command line using ```bash python finetune.py```
- **Finetune evidential deep learning model**: The finetune EDL script is executed from the command line using ```bash python finetune_edl.py```
## Citation




