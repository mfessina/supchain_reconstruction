# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 13:12:02 2023

@author: massi
"""

import pandas as pd
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import os
import scipy.sparse as scp

# ============================================================================
# MODEL SOLVED AT THE ISIC 4-DIGIT LEVEL: ONE Z PER LAYER (ISIC CODE) 
# FIXING ITS LINK DENSITY
# ============================================================================

def z_solver(z, s_in, s_out, scale):
    """
    Computes the sum of connection probabilities for the z-solver equation with sectoral scaling.
    
    This function evaluates the equation used to find the optimal z parameter that fixes
    the link density for a given sector. It computes the sum of all pairwise connection
    probabilities based on incoming and outgoing strengths, incorporating a sectoral scaling
    factor that accounts for input-output sector relationships.
    
    Args:
        z (float): Current value of the z parameter (arbitrary initialization value, set to 0.001 in IO_scGM function).
        s_in (pandas.DataFrame): DataFrame containing incoming strengths with columns:
                                - 'total_tax': Total tax value (strength/weight)
                                - 'ISIC_inf': ISIC code of the input sector
                                Indexed by firm IDs (id_inf).
        s_out (pandas.Series): Series containing outgoing strengths (sum of total_tax) 
                              indexed by firm IDs (id_prov).
        scale (numpy.ndarray): Array of sectoral scaling factors (s_gigj/s_gin) for each
                              firm in s_in, accounting for sector-to-sector flow distributions.
    
    Returns:
        float: Sum of all pairwise connection probabilities computed as 
               sum(s_in * scale * s_out / (1 + z * s_in * scale * s_out)).
    """
    d = sum(np.multiply.outer(s_in['total_tax'].to_numpy()*scale,s_out.to_numpy()).flatten()/(1+z*np.multiply.outer(s_in['total_tax'].to_numpy()*scale,s_out.to_numpy()).flatten()))
    return d


def z_solution(a, L, s_in, s_out, scale):
    """
    Solves iteratively for the optimal z parameter starting from an initial value with sectoral scaling.
    
    This function iteratively refines the z parameter value by solving the equation
    L = sum(s_in * scale * s_out / (1 + z * s_in * scale * s_out)) until convergence.
    The z parameter controls the overall connection probability density for a sector,
    with sectoral scaling factors incorporated. Uses a fixed 200 iterations.
    
    Args:
        a (float): Initial guess for the z parameter (typically 0.001).
        L (int): Total number of links (edges) in the network for the sector.
        s_in (pandas.DataFrame): DataFrame containing incoming strengths with columns:
                                - 'total_tax': Total tax value
                                - 'ISIC_inf': ISIC code of the input sector
                                Indexed by firm IDs (id_inf).
        s_out (pandas.Series): Series containing outgoing strengths indexed by firm IDs.
        scale (numpy.ndarray): Array of sectoral scaling factors for each firm in s_in.
    
    Returns:
        float: Converged value of the z parameter after 200 iterations (fixed number of iterations).
    """
    for i in range(200):
        x = L/z_solver(a, s_in, s_out, scale)
        a = x
    return a


def IO_pij_4dig(p, edge_df):
    """
    This function calculates the probability of connection between all pairs of firms within
    a specific ISIC 4-digit sector, incorporating sectoral scaling factors. It first solves
    for the optimal z parameter that fixes the link density for this sector, then computes
    all pairwise probabilities and saves them to a CSV file.
    
    The function computes sector-to-sector flows (s_gigj) and input sector totals (s_gin)
    to derive scaling factors that adjust connection probabilities based on input-output
    relationships between sectors.
    
    Args:
        p (str or int): ISIC 4-digit code identifying the sector.
        edge_df (pandas.DataFrame): DataFrame containing network data with columns:
                                   - 'ISIC_prov': ISIC 4-digit code of supplier firm
                                   - 'ISIC_inf': ISIC 4-digit code of client firm's sector
                                   - 'id_inf': ID of the client firm (target)
                                   - 'id_prov': ID of the supplier firm (source)
                                   - 'total_tax': weight of the link from 'id_prov' to 'id_inf'
    
    Returns:
        None: This function saves results to .csv files.
              Output files: 'IO_scgm_prob_edgelist_4dig/IO_scGM_prob_edgelist_{p}.csv'
              Contains columns: 'id_prov', 'id_inf', 'p_ij'
    """
    
    temp = edge_df.loc[edge_df['ISIC_prov']==p]
    a = temp.copy()
    L = np.size(temp, axis=0)
    
    # Computation of s_i^out and s_gi->gj can be done on temp (sector gi fixed)
    s_out = a.groupby('id_prov')['total_tax'].sum()
    s_gigj = a.groupby('ISIC_inf')['total_tax'].sum()
    
    # Computation of s_j^in and s_gi^in has to be done on edge_df (involves all input sectors gi)
    s_in = edge_df.groupby(['id_inf','ISIC_inf'],as_index=False)['total_tax'].sum()
    s_in.set_index('id_inf',drop=True,inplace=True)
    s_gin = edge_df.groupby('ISIC_inf')['total_tax'].sum()
    
    a_in = s_in[s_in['ISIC_inf'].isin(s_gigj.index.tolist())].index.values
    a_out = a['id_prov'].unique()
    
    s_in = s_in.loc[a_in]
    s_out = s_out.loc[a_out]
    s_gigj = s_gigj.loc[s_in['ISIC_inf'].values]
    s_gin = s_gin.loc[s_in['ISIC_inf'].values]
    scale = s_gigj.values/s_gin.values
    
    z = z_solution(0.001, L, s_in, s_out, scale)
    
    mat = np.multiply.outer(z*s_out.values,s_in['total_tax'].values*scale)/(1+np.multiply.outer(z*s_out.values,s_in['total_tax'].values*scale))
    df = pd.DataFrame(mat, index=list(a_out), columns=list(a_in))
    edge = df.stack().reset_index()
    edge.columns = ['id_prov','id_inf','p_ij']
    edge.to_csv('IO_scgm_prob_edgelist_4dig/IO_scGM_prob_edgelist_{}.csv'.format(p))


def IO_scGM(edge_df, products):
    """
    Main function to compute connection probabilities for all sectors using parallel processing.
    
    This function parallelizes the computation of connection probabilities (p_ij) between any two firms
    for every sector in the network, incorporating sectoral scaling factors based on input-output
    relationships. The model is solved at the ISIC 4-digit level with one z parameter per sector.
    
    The result is a list of sector-wise dataframes containing the link probability between any two
    firms in the ensemble, stored into separate .csv files.
    
    Args:
        edge_df (pandas.DataFrame): DataFrame containing network data with columns:
                                   - 'ISIC_prov': ISIC 4-digit code of supplier firm
                                   - 'ISIC_inf': ISIC 4-digit code of client firm's sector
                                   - 'id_inf': ID of client firm (target)
                                   - 'id_prov': ID of supplier firm (source)
                                   - 'total_tax': weight of the link from 'id_prov' to 'id_inf'
        products (list): List of ISIC 4-digit codes of sectors to process.
    
    Returns:
        list: List of dataframes (one per sector), where each dataframe contains the
              original edge data for that sector with computed p_ij values. The list
              needs to be concatenated in the calling code to reconstruct the full dataframe.
    """
    num_cores = multiprocessing.cpu_count() 
    la = Parallel(n_jobs=num_cores)(delayed(IO_pij_4dig)(p,edge_df) for p in products)
    
    return la


def IO_wij(p, edge_df):
    """
    Computes and assigns ensemble weights w_ij for the IO_scGM probability edgelist.
    
    This function calculates ensemble weights for edges based on the original tax data,
    computed connection probabilities, and sectoral scaling factors. The weights are
    computed as: w_ij = (s_in * s_out * scale) / (w_tot * p_ij) when p_ij != 0, otherwise 0.
    
    The function reads the probability edgelist from a CSV file, computes weights
    incorporating sectoral scaling, and saves the updated edgelist with the new w_ij
    column back to the same file.
    
    Args:
        p (str or int): ISIC 4-digit code identifying the sector to process.
        edge_df (pandas.DataFrame): DataFrame containing network data with columns:
                                   - 'ISIC_prov': ISIC 4-digit code of supplier firm
                                   - 'ISIC_inf': ISIC 4-digit code of client firm's sector
                                   - 'id_inf': ID of the client firm (target)
                                   - 'id_prov': ID of the supplier firm (source)
                                   - 'total_tax': weight of the link from 'id_prov' to 'id_inf'
    
    Returns:
        None: This function modifies and saves the .csv file but does not return a value.
              Updates file: 'IO_scGM_prob_edgelist_4dig/IO_scGM_prob_edgelist_{p}.csv'
              Adds column: 'w_ij' (ensemble weights)
    """
    temp = edge_df.loc[edge_df['ISIC_prov']==p]
    sample = pd.read_csv('IO_scGM_prob_edgelist_4dig/IO_scGM_prob_edgelist_{}.csv'.format(p), index_col=0)
    a = temp.copy()

    s_out = a.groupby('id_prov')['total_tax'].sum()
    s_out.index = s_out.index.map(int)
    w_tot = np.sum(a['total_tax'].values)
    
    s_in = edge_df.groupby(['id_inf','ISIC_inf'],as_index=False)['total_tax'].sum()
    s_in.set_index('id_inf',drop=True,inplace=True)
    
    s_gin = edge_df.groupby('ISIC_inf')['total_tax'].sum()
    s_gigj = a.groupby('ISIC_inf')['total_tax'].sum()
    
    f_in = sample['id_inf'].astype(int)
    f_out = sample['id_prov'].astype(int)
    sectors = s_in.loc[f_in,'ISIC_inf'].values
    s_gigj = s_gigj.loc[sectors]
    s_gin = s_gin[sectors]
    scale = s_gigj.values/s_gin.values
    pijs = sample['p_ij'].values 
    
    wijs = np.where(pijs!=0, (s_in.loc[f_in,'total_tax'].values*s_out[f_out].values*scale)/(w_tot*pijs), 0)
    sample['w_ij'] = wijs
    
    sample.to_csv('IO_scGM_prob_edgelist_4dig/IO_scGM_prob_edgelist_{}.csv'.format(p))


def IO_scGM_weights(edge_df, products):
    """
    Parallel execution of IO_wij function to assign weights to all sectors.
    
    This function parallelizes the computation of ensemble weights across all sectors.
    It calls the IO_wij function for each sector using all available CPU cores to
    accelerate the computation.
    
    Args:
        edge_df (pandas.DataFrame): DataFrame containing network data with columns:
                                   - 'ISIC_prov': ISIC 4-digit code of supplier firm
                                   - 'ISIC_inf': ISIC 4-digit code of client firm's sector
                                   - 'id_inf': ID of the client firm (target)
                                   - 'id_prov': ID of the supplier firm (source)
                                   - 'total_tax': weight of the link from 'id_prov' to 'id_inf'
        products (list): List of ISIC 4-digit codes of sectors to process.
    
    Returns:
        list: List of results from IO_wij function calls (None values, as IO_wij
              saves results to .csv files rather than returning them).
    """
    
    num_cores = multiprocessing.cpu_count()
    
    la = Parallel(n_jobs=num_cores)(delayed(IO_wij)(p,edge_df) for p in products)
    
    return la


def IO_sample(p):
    """
    Explicitly samples synthetic networks from the IO_scGM ensemble for a given sector.
    
    This function generates 1000 synthetic network samples from the IO_scGM ensemble for
    a specific sector. Sampling is done individually for each layer (sector), and due to
    memory constraints, each realized sample is stored in a separate .npz file as a
    compressed sparse matrix (CSC format).
    
    Samples are realized as binary adjacency matrices indicating which edges exist.
    The corresponding nodes can be traced back through the corresponding 
    IO_scGM_prob_edgelist_{p}_thr1.csv file which contains the node IDs.
    
    Args:
        p (str or int): ISIC 4-digit code identifying the sector to sample.
    
    Returns:
        None: This function saves samples to .npz files but does not return a value.
              Creates directory: 'IO_scGM_prob_edgelist_4dig_thr1/IO_scGM_prob_edgelist_{p}_samples_thr1/'
              Saves files: 'IO_scGM_prob_edgelist_{p}_sample_{i}_thr1.npz' (i from 0 to 999)
    """
    
    edge_df = pd.read_csv('IO_scGM_prob_edgelist_4dig/IO_scGM_prob_edgelist_{}.csv'.format(p), index_col=0)
    probs = edge_df['p_ij'].to_numpy() 
    dim = np.size(probs)

    
    newpath = r'C:\Users\massi\Desktop\Ecuador\IO_scGM_prob_edgelist_4dig\IO_scGM_prob_edgelist_{}_samples'.format(p) 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
   
    for i in range(1000):

        sample_mat = np.array(probs>np.random.sample(dim))
        samp_csc = scp.csc_matrix(sample_mat)
        
        scp.save_npz('IO_scGM_prob_edgelist_4dig/IO_scGM_prob_edgelist_{}_samples/IO_scGM_prob_edgelist_{}_sample_{}.npz'.format(p,p,i), samp_csc)   
    

def IO_network_sampling(products):
    """
    Parallel execution of IO_sample function to generate synthetic networks for all sectors.
    
    This function parallelizes the generation of synthetic network samples across all
    sectors. It calls the IO_sample function for each sector using all available
    CPU cores to accelerate the computation.
    
    Args:
        products (list): List of ISIC 4-digit codes of sectors to sample.
    
    Returns:
        list: List of results from IO_sample function calls (None values, as IO_sample
              saves results to .npz files rather than returning them).
    """
    num_cores = multiprocessing.cpu_count()
    
    la = Parallel(n_jobs=num_cores)(delayed(IO_sample)(p) for p in products)
    
    return la

















