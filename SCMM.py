# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:18:11 2023

@author: massi
"""

import pandas as pd
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import os
import scipy.sparse as scp

# ============================================================================
# STRIPE CORRECTED MAXIMUM ENTROPY MODEL
# ============================================================================


def pij(p, edge_df):
    """
    Computes connection probabilities p_ij (weights w_ij) using maximum entropy principle.
    
    This function calculates the probability/weight of connection between all pairs of firms
    within a specific ISIC 4-digit sector using the maximum entropy principle. The weights
    are computed as the product of incoming and outgoing strengths normalized by total weight,
    representing the most unbiased distribution given the constraints.
    
    Args:
        p (str or int): ISIC 4-digit code identifying the sector.
        edge_df (pandas.DataFrame): DataFrame containing network data with columns:
                                   - 'ISIC_prov': ISIC 4-digit code of supplier firm
                                   - 'id_inf': ID of the client firm (target)
                                   - 'id_prov': ID of the supplier firm (source)
                                   - 'total_tax': weight of the link from 'id_prov' to 'id_inf'
    
    Returns:
        pandas.DataFrame: DataFrame containing the probability/weight edgelist with columns:
                         'id_prov', 'id_inf', 'w_ij' (connection weights)
    """
    
    temp = edge_df.loc[edge_df['ISIC_prov']==p]
    s_in = temp.groupby('id_inf')['total_tax'].sum()
    s_in.index = s_in.index.map(int)
    s_out = temp.groupby('id_prov')['total_tax'].sum()
    s_out.index = s_out.index.map(int)
    w_tot = np.sum(temp['total_tax'].values)
    
    a_in = temp['id_inf'].unique().astype(int)
    a_out = temp['id_prov'].unique().astype(int)
    mat = np.multiply.outer(s_out[a_out].values,s_in[a_in].values)/w_tot
    df = pd.DataFrame(mat, index=list(a_out), columns=list(a_in))
    edge = df.stack().reset_index()
    edge.columns = ['id_prov','id_inf','w_ij']
    return edge


# ============================================================================
# PARALLEL EXECUTION FUNCTIONS
# ============================================================================


def SCMaxEnt(edge_df, products):
    """
    Parallel execution of pij function for standard maximum entropy model.
    
    This function parallelizes the computation of connection weights using the maximum
    entropy principle across all sectors. It calls the pij function for each sector
    using all available CPU cores to accelerate the computation.
    
    Args:
        edge_df (pandas.DataFrame): DataFrame containing network data with columns:
                                   - 'ISIC_prov': ISIC 4-digit code of supplier firm
                                   - 'id_inf': ID of client firm (target)
                                   - 'id_prov': ID of supplier firm (source)
                                   - 'total_tax': weight of the link from 'id_prov' to 'id_inf'
        products (list): List of ISIC 4-digit codes of sectors to process.
    
    Returns:
        list: List of DataFrames (one per sector), where each DataFrame contains the
              weighted edgelist with columns: 'id_prov', 'id_inf', 'w_ij'
    """
    num_cores = multiprocessing.cpu_count()
    la = Parallel(n_jobs=num_cores)(delayed(pij)(p,edge_df) for p in products)
    
    return la















