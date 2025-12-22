# -*- coding: utf-8 -*-
"""
Input-Output Maximum Entropy Model (IO_maxent) implementation for network analysis.

This module implements Maximum Entropy models for analyzing firm networks. It provides
two variants: a standard maximum entropy model (IOMaxEnt) and an Input-Output constrained
maximum entropy model (IOGMMaxEnt) that incorporates sectoral scaling factors based on
input-output relationships between sectors.

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
# MAXIMUM ENTROPY MODEL: STANDARD VERSION
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
# MAXIMUM ENTROPY MODEL: INPUT-OUTPUT CONSTRAINED VERSION
# ============================================================================


def IO_pij(p, edge_df):
    """
    Computes connection probabilities p_ij (weights w_ij) using maximum entropy with IO constraints.
    
    This function calculates the probability/weight of connection between firms within a specific
    ISIC 4-digit sector using the maximum entropy principle, but incorporates sectoral scaling
    factors based on input-output relationships. The scaling factors account for sector-to-sector
    flow distributions (s_gigj/s_gin), ensuring the model respects input-output sector constraints
    while maximizing entropy.
    
    Args:
        p (str or int): ISIC 4-digit code identifying the sector.
        edge_df (pandas.DataFrame): DataFrame containing network data with columns:
                                   - 'ISIC_prov': ISIC 4-digit code of supplier firm
                                   - 'ISIC_inf': ISIC 4-digit code of client firm's sector
                                   - 'id_inf': ID of the client firm (target)
                                   - 'id_prov': ID of the supplier firm (source)
                                   - 'total_tax': weight of the link from 'id_prov' to 'id_inf'
    
    Returns:
        pandas.DataFrame: DataFrame containing the probability/weight edgelist with columns:
                         'id_prov', 'id_inf', 'w_ij' (connection weights with IO scaling)
    """
    
    
    temp = edge_df.loc[edge_df['ISIC_prov']==p].copy()
 
    # Computation of s_i^out and s_gi->gj can be done on temp (sector gi fixed)
    s_out = temp.groupby('id_prov')['total_tax'].sum()
    s_out.index = s_out.index.map(int)
    s_gigj = temp.groupby('ISIC_inf')['total_tax'].sum()
    w_tot = np.sum(temp['total_tax'].values)
    
    # Computation of s_j^in and s_gi^in has to be done on edge_df (involves all input sectors gi)
    
    s_in = edge_df.groupby(['id_inf','ISIC_inf'],as_index=False)['total_tax'].sum()
    s_in.set_index('id_inf',drop=True,inplace=True)
    s_in.index = s_in.index.map(int)
    s_gin = edge_df.groupby('ISIC_inf')['total_tax'].sum()
    
    a_in = s_in[s_in['ISIC_inf'].isin(s_gigj.index.tolist())].index.values.astype(int)
    a_out = temp['id_prov'].unique().astype(int)
    
    s_in = s_in.loc[a_in]
    s_out = s_out.loc[a_out]
    s_gigj = s_gigj.loc[s_in['ISIC_inf'].values]
    s_gin = s_gin.loc[s_in['ISIC_inf'].values]
    scale = s_gigj.values/s_gin.values

    mat = np.multiply.outer(s_out.values,(s_in['total_tax'].values*scale))/w_tot
    df = pd.DataFrame(mat, index=list(a_out), columns=list(a_in))
    edge = df.stack().reset_index()
    edge.columns = ['id_prov','id_inf','w_ij']

    return edge


# ============================================================================
# PARALLEL EXECUTION FUNCTIONS
# ============================================================================


def IOMaxEnt(edge_df, products):
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
              probability/weight edgelist with columns: 'id_prov', 'id_inf', 'w_ij'
    """
    num_cores = multiprocessing.cpu_count()
    la = Parallel(n_jobs=num_cores)(delayed(pij)(p,edge_df) for p in products)
    
    return la


def IOGMMaxEnt(edge_df, products):
    """
    Parallel execution of IO_pij function for Input-Output constrained maximum entropy model.
    
    This function parallelizes the computation of connection weights using the maximum
    entropy principle with Input-Output constraints across all sectors. It calls the
    IO_pij function for each sector using all available CPU cores to accelerate the
    computation.
    
    Args:
        edge_df (pandas.DataFrame): DataFrame containing network data with columns:
                                   - 'ISIC_prov': ISIC 4-digit code of supplier firm
                                   - 'ISIC_inf': ISIC 4-digit code of client firm's sector
                                   - 'id_inf': ID of client firm (target)
                                   - 'id_prov': ID of supplier firm (source)
                                   - 'total_tax': weight of the link from 'id_prov' to 'id_inf'
        products (list): List of ISIC 4-digit codes of sectors to process.
    
    Returns:
        list: List of DataFrames (one per sector), where each DataFrame contains the
              probability/weight edgelist with columns: 'id_prov', 'id_inf', 'w_ij'
              (with IO scaling factors applied)
    """
    num_cores = multiprocessing.cpu_count()
    la = Parallel(n_jobs=num_cores)(delayed(IO_pij)(p,edge_df) for p in products)
    
    return la











