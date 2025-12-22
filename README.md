# Reconstruction models for firm-level supply chain networks
This repository contains implementations of several network configuration models for analyzing firm networks at different levels of aggregation. The models were developed and used in the research paper:

**Inferring firm-level supply chain networks with realistic systemic risk from industry sector-level data** - *Fessina, Massimiliano and Cimini, Giulio and Squartini, Tiziano and Astudillo-Estévez, Pablo and Thurner, Stefan and Garlaschelli, Diego* - *arXiv preprint* - *2024*


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{fessina2024inferring,
  title   = {Inferring firm-level supply chain networks with realistic systemic risk from industry sector-level data},
  author  = {Fessina, Massimiliano and Cimini, Giulio and Squartini, Tiziano and
             Astudillo-Est{\'e}vez, Pablo and Thurner, Stefan and Garlaschelli, Diego},
  journal = {arXiv preprint arXiv:2408.02467},
  year    = {2024},
  doi     = {10.48550/arXiv.2408.02467}
}
```

## Overview

This repository provides implementations of the following network models:

- **SCGM (Stripe-Corrected Gravity Model)**: reconstruction model for firm-level supply chain network, introduced in [Ialongo, L. N., de Valk, C., Marchese, E., Jansen, F., Zmarrou, H., Squartini, T., & Garlaschelli, D. (2022). Reconstructing firm-level interactions in the Dutch input–output network from production constraints. Scientific reports, 12(1), 11847].
  
- **IOGM (Input-Output Gravity Model)**: an extension of SCGM that makes use of sector-level aggregated information to reconstruct the network.

- **DCGM (Directed Configuration Model)**: reconstruction model for interbank networks, introduced in [Cimini, G., Squartini, T., Garlaschelli, D., & Gabrielli, A. (2015). Systemic risk analysis on reconstructed economic and financial networks. Scientific reports, 5(1), 15758]

- **SCMM (Stripe-Corrected Maxent Model)**: SCGM-based variation of the standard MaxEnt reconstruction model for networks.

## Repository Structure

All code implementations are stored in the `scripts/` folder:

- `SCGM.py` - SCGM implementation
- `IOGM.py` - IOGM implementation
- `DCGM.py` - DCGM implementation
- `SCMM.py` - SCMM implementation
- `SCGM_usage_example.py` - Example usage script demonstrating how to use the SCGM module

## Requirements

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

