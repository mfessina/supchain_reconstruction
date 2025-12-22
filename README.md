# Reconstruction models for firm-level supply chain networks
This repository contains implementations of several network configuration models for analyzing firm networks at different levels of aggregation. The models were developed and used in the research paper:

**Inferring firm-level supply chain networks with realistic systemic risk from industry sector-level data** - *Fessina, Massimiliano and Cimini, Giulio and Squartini, Tiziano and Astudillo-Est{\'e}vez, Pablo and Thurner, Stefan and Garlaschelli, Diego* - *arXiv preprint* - *2024*

**arXiv:2408.02467**

## Citation

If you use this code in your research, please cite our paper:
@article{fessina2024inferring,
  title={Inferring firm-level supply chain networks with realistic systemic risk from industry sector-level data},
  author={Fessina, Massimiliano and Cimini, Giulio and Squartini, Tiziano and Astudillo-Est{\'e}vez, Pablo and Thurner, Stefan and Garlaschelli, Diego},
  journal={arXiv preprint arXiv:2408.02467},
  year={2024},
  doi={10.48550/arXiv.2408.02467}
}

## Overview

This repository provides implementations of the following network models:

- **scGM (Sectoral Configuration Model)**: Implements the Sectoral Configuration Model for analyzing firm networks at different levels of ISIC aggregation, with support for both sector-specific and full network configurations.

- **IO_scGM (Input-Output Sectoral Configuration Model)**: An extension of scGM that incorporates sector-to-sector flows and scaling factors based on input-output relationships.

- **dcGM (Directed Configuration Model)**: A directed configuration model that uses a single global z parameter computed from the entire network, allowing connections across all sectors.

- **IO_maxent (Input-Output Maximum Entropy)**: Maximum entropy models for firm networks, available in both standard and Input-Output constrained variants.

## Repository Structure

All code implementations are stored in the `scripts/` folder:

- `scGM.py` - Sectoral Configuration Model implementation
- `IO_scGM.py` - Input-Output Sectoral Configuration Model implementation
- `dcGM.py` - Directed Configuration Model implementation
- `IO_maxent.py` - Maximum Entropy Model implementations- `SCGM_usage_example.py` - Example usage script demonstrating how to use the scGM module

- ## Requirements

The code requires the following Python packages:
- `pandas`
- `numpy`
- `scipy`
- `joblib`

## Quick Start
To get started with the models, see `scripts/SCGM_usage_example.py` which demonstrates a complete workflow for the SCGM model (same for the other models):

1. Loading network data (edgelist format)
2. Generating the ensemble and computing connection probabilities
3. Adding ensemble weights to probability edgelists
4. Sampling synthetic networks from the ensemble

