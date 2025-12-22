# -*- coding: utf-8 -*-
"""
Usage example for the Sectoral Configuration Model (scGM).

This script demonstrates how to use the scGM module to:
1. Generate the scGM ensemble and compute connection probabilities
2. Add ensemble weights to the probability edgelists
3. Sample synthetic networks from the ensemble

Usage:
    - Modify the input file path and parameters as needed
    - Ensure the output directory 'scgm_prob_edgelist_4dig/' exists before running
    - The script can be run section by section or as a complete pipeline

Created on Tue Apr 18 15:45:20 2023

@author: massi
"""

import numpy as np
import pandas as pd
from scGM import scGM, network_sampling, scGM_weights

# ============================================================================
# DATA LOADING AND PARAMETER SETUP
# ============================================================================

# Load the network edgelist
# Expected columns: 'ISIC_prov', 'id_inf', 'id_prov', 'total_tax'
edgelist = pd.read_csv('4dig_z_thr1_files/edgelist_ecuador_2008_preprocessing_def_thr1_gcc.csv')

# Extract list of unique ISIC 4-digit codes (products/sectors) to process
products = list(sorted(set(edgelist['ISIC_prov'].tolist())))

# Number of iterations for z parameter convergence
n_iterations = 200

# Aggregation mode: 4 = ISIC 4-digit level (one z per sector), 0 = full network level
aggregation = 4

# ============================================================================
# STEP 1: GENERATE SCGM ENSEMBLE AND COMPUTE CONNECTION PROBABILITIES
# ============================================================================

# Generate the scGM ensemble and store link probabilities into product-specific edgelists
# For every product p in products, the function computes link probabilities between any 
# two firms and stores them in a single .csv file named 'scGM_prob_edgelist_{p}.csv' 
# inside the folder 'scgm_prob_edgelist_4dig/'.
#
# Note: The output folder 'scgm_prob_edgelist_4dig/' should be created before running.
#       To change the output directory, modify the filepath in the pij_4dig or pij_full 
#       function in scGM.py

scGM(edgelist, products, aggregation, n_iterations)

# ============================================================================
# STEP 2: ADD ENSEMBLE WEIGHTS TO THE PROBABILITY EDGELISTS
# ============================================================================

# Add ensemble weights w_ij to the probability edgelists
# For every product p in products, the function retrieves the corresponding probability 
# edgelist 'scgm_prob_edgelist_4dig/scGM_prob_edgelist_{p}.csv' and adds the column 'w_ij' 
# containing the ensemble weight for every link.
#
# The weights are computed as: w_ij = (s_in * s_out) / (w_tot * p_ij) when p_ij != 0

scGM_weights(edgelist, products)

# ============================================================================
# STEP 3: SAMPLE SYNTHETIC NETWORKS FROM THE ENSEMBLE
# ============================================================================

# Generate 1000 explicit samples from the ensemble for each product
# For every product p in products, the function generates 1000 synthetic network samples.
# Each sample is stored as a compressed sparse matrix in:
# 'scGM_prob_edgelist_4dig/scGM_prob_edgelist_{p}_samples/scGM_prob_edgelist_{p}_sample_{i}.npz'
# where i ranges from 0 to 999.
#
# The folder 'scGM_prob_edgelist_{p}_samples' is automatically created by the function.
#
# Each .npz file contains a binary array (1s and 0s) indicating which links are realized
# in the sample. The array has the same length as the corresponding probability edgelist.
# To interpret the samples, use the firm IDs and link weights from 'scGM_prob_edgelist_{p}.csv'

network_sampling(products)











