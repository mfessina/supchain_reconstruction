# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:55:07 2023

@author: massi
"""

import pandas as pd
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import os
import scipy.sparse as scp

# ============================================================================
# MODEL SOLVED ON THE FULL NETWORK: ONE SINGLE Z FIXING THE FULL DENSITY
# ============================================================================


def z_full_solver(z, s_in, s_out):
    """
    Computes the sum of connection probabilities for the z-solver equation.
    
    This function evaluates the equation used to find the optimal z parameter that fixes
    the link density for the entire network. It computes the sum of all pairwise connection
    probabilities across all firms in the network.
    
    Args:
        z (float): Current value of the z parameter (arbitrary initialization value, set to 0.001 in dcGM function).
        s_in (pandas.Series): Series containing incoming strengths (sum of total_tax) 
                             indexed by firm IDs (id_inf) across all firms in the network.
        s_out (pandas.Series): Series containing outgoing strengths (sum of total_tax) 
                              indexed by firm IDs (id_prov) across all firms in the network.
    
    Returns:
        float: Sum of all pairwise connection probabilities computed as 
               sum(s_in * s_out / (1 + z * s_in * s_out)).
    """
    
    d = sum(np.multiply.outer(s_in.to_numpy(),s_out.to_numpy()).flatten()/(1+z*np.multiply.outer(s_in.to_numpy(),s_out.to_numpy()).flatten()))
    return d


def z_full_solution(a, edge_df, n_iterations):
    """
    Solves iteratively for the optimal z parameter for the full network.
    
    This function iteratively refines the z parameter value for the entire network
    by solving the equation L = sum(s_in * s_out / (1 + z * s_in * s_out)) across
    all firms until convergence. Uses a fixed n_iterations iterations.
    
    Args:
        a (float): Initial guess for the z parameter (typically 0.001).
        edge_df (pandas.DataFrame): DataFrame containing network data with columns:
                                   - 'ISIC_prov': ISIC 4-digit code of supplier firm
                                   - 'id_inf': ID of the client firm (target)
                                   - 'id_prov': ID of the supplier firm (source)
                                   - 'total_tax': weight of the link from 'id_prov' to 'id_inf'
    
    Returns:
        float: Converged value of the z parameter after n_iterations iterations.
    """
    
    L = np.size(edge_df,axis=0)
    s_in = edge_df.groupby('id_inf')['total_tax'].sum()
    s_out = edge_df.groupby('id_prov')['total_tax'].sum()
    
    for i in range(n_iterations):
        x = L/z_full_solver(a, s_in, s_out)
        a = x
    return a


def pij(p, edge_df, z):
    """
    Computes connection probabilities p_ij for a sector using the full network z parameter.
    
    This function calculates the probability of connection between firms in a specific
    sector and all firms in the network. Unlike scGM which may restrict connections
    within sectors, dcGM allows connections from firms in sector p to any firm in the
    network. It uses global strengths computed from the entire network and saves
    a complete probability edgelist.
    
    Args:
        p (str or int): ISIC 4-digit code identifying the sector to process.
        edge_df (pandas.DataFrame): DataFrame containing network data with columns:
                                   - 'ISIC_prov': ISIC 4-digit code of supplier firm
                                   - 'id_inf': ID of the client firm (target)
                                   - 'id_prov': ID of the supplier firm (source)
                                   - 'total_tax': weight of the link from 'id_prov' to 'id_inf'
        z (float): Global z parameter value computed for the full network.
    
    Returns:
        None: This function saves results to .csv files.
              Output files: 'dcGM_prob_edgelist/dcGM_prob_edgelist_{p}.csv'
              Contains columns: 'id_prov', 'id_inf', 'p_ij'
    """
    temp = edge_df.loc[edge_df['ISIC_prov']==p]
    s_in = edge_df.groupby('id_inf')['total_tax'].sum()
    s_in.index = s_in.index.map(int)
    s_out = edge_df.groupby('id_prov')['total_tax'].sum()
    s_out.index = s_out.index.map(int)
    
    a_in = edge_df['id_inf'].unique().astype(int)
    a_out = temp['id_prov'].unique().astype(int)
    mat = np.multiply.outer(z*s_out[a_out].values,s_in[a_in].values)/(1+np.multiply.outer(z*s_out[a_out].values,s_in[a_in].values))
    df = pd.DataFrame(mat, index=list(a_out), columns=list(a_in))
    edge = df.stack().reset_index()
    edge.columns = ['id_prov','id_inf','p_ij']
    edge.to_csv('dcGM_prob_edgelist/dcGM_prob_edgelist_{}.csv'.format(p))


def dcGM(edge_df, products):
    """
    Main function to compute connection probabilities for all sectors using parallel processing.
    
    This function parallelizes the computation of connection probabilities (p_ij) between any two firms
    for every sector in the network. It uses a single global z parameter computed from the entire
    network, allowing connections across all sectors based on global firm strengths.
    
    The result is a list of sector-wise dataframes containing the link probability between firms
    in each sector and all firms in the network, stored into separate .csv files.
    
    Args:
        edge_df (pandas.DataFrame): DataFrame containing network data with columns:
                                   - 'ISIC_prov': ISIC 4-digit code of supplier firm
                                   - 'id_inf': ID of client firm (target)
                                   - 'id_prov': ID of supplier firm (source)
                                   - 'total_tax': weight of the link from 'id_prov' to 'id_inf'
        products (list): List of ISIC 4-digit codes of sectors to process.
    
    Returns:
        list: List of dataframes (one per sector), where each dataframe contains the
              probability edgelist for that sector. The list needs to be concatenated
              in the calling code to reconstruct the full dataframe.
    """
    num_cores = multiprocessing.cpu_count()
    
    z = z_full_solution(0.001,edge_df)
        
    la = Parallel(n_jobs=num_cores)(delayed(pij)(p,edge_df,z) for p in products)
    
    return la


def wij(p, edge_df):
    """
    Computes and assigns ensemble weights w_ij for the dcGM probability edgelist.
    
    This function calculates ensemble weights for edges based on the original tax data
    and the computed connection probabilities. The weights are computed as:
    w_ij = (s_in * s_out) / (w_tot * p_ij) when p_ij != 0, otherwise 0.
    
    The function reads the probability edgelist from a CSV file, computes weights
    using global strengths from the entire network, and saves the updated edgelist
    with the new w_ij column back to the same file.
    
    Args:
        p (str or int): ISIC 4-digit code identifying the sector to process.
        edge_df (pandas.DataFrame): DataFrame containing network data with columns:
                                   - 'ISIC_prov': ISIC 4-digit code of supplier firm
                                   - 'id_inf': ID of the client firm (target)
                                   - 'id_prov': ID of the supplier firm (source)
                                   - 'total_tax': weight of the link from 'id_prov' to 'id_inf'
    
    Returns:
        None: This function modifies and saves the .csv file but does not return a value.
              Updates file: 'dcGM_prob_edgelist/dcGM_prob_edgelist_{p}.csv'
              Adds column: 'w_ij' (ensemble weights)
    """
    s_in = edge_df.groupby('id_inf')['total_tax'].sum()
    s_in.index = s_in.index.map(int)
    s_out = edge_df.groupby('id_prov')['total_tax'].sum()
    s_out.index = s_out.index.map(int)
    w_tot = np.sum(edge_df['total_tax'].values)
    sample = pd.read_csv('dcGM_prob_edgelist/dcGM_prob_edgelist_{}.csv'.format(p), index_col=0)
    
    f_in = sample['id_inf'].values.astype(int)
    f_out = sample['id_prov'].values.astype(int)
    pijs = sample['p_ij'].values 
    wijs = np.where(pijs!=0, (s_in[f_in].values*s_out[f_out].values)/(w_tot*pijs), 0)
    sample['w_ij'] = wijs
    
    sample.to_csv('dcGM_prob_edgelist/dcGM_prob_edgelist_{}.csv'.format(p))


def dcGM_weights(edge_df, products):
    """
    Parallel execution of wij function to assign weights to all sectors.
    
    This function parallelizes the computation of ensemble weights across all sectors.
    It calls the wij function for each sector using all available CPU cores to
    accelerate the computation.
    
    Args:
        edge_df (pandas.DataFrame): DataFrame containing network data with columns:
                                   - 'ISIC_prov': ISIC 4-digit code of supplier firm
                                   - 'id_inf': ID of the client firm (target)
                                   - 'id_prov': ID of the supplier firm (source)
                                   - 'total_tax': weight of the link from 'id_prov' to 'id_inf'
        products (list): List of ISIC 4-digit codes of sectors to process.
    
    Returns:
        list: List of results from wij function calls (None values, as wij
              saves results to .csv files rather than returning them).
    """
    
    num_cores = multiprocessing.cpu_count()
    
    la = Parallel(n_jobs=num_cores)(delayed(wij)(p,edge_df) for p in products)
    
    return la


def sample(p):
    """
    Explicitly samples synthetic networks from the dcGM ensemble for a given sector.
    
    This function generates 1000 synthetic network samples from the dcGM ensemble for
    a specific sector. Sampling is done individually for each layer (sector), and due to
    memory constraints, each realized sample is stored in a separate .npz file as a
    compressed sparse matrix (CSC format).
    
    Samples are realized as binary adjacency matrices indicating which edges exist.
    The corresponding nodes can be traced back through the corresponding 
    dcGM_prob_edgelist_{p}.csv file which contains the node IDs.
    
    Args:
        p (str or int): ISIC 4-digit code identifying the sector to sample.
    
    Returns:
        None: This function saves samples to .npz files but does not return a value.
              Creates directory: 'dcGM_prob_edgelist/dcGM_prob_edgelist_{p}_samples/'
              Saves files: 'dcGM_prob_edgelist_{p}_sample_{i}.npz' (i from 0 to 999)
    """
    
    edge_df = pd.read_csv('dcGM_prob_edgelist/dcGM_prob_edgelist_{}.csv'.format(p), index_col=0)
    probs = edge_df['p_ij'].to_numpy() 
    dim = np.size(probs)

    
    newpath = r'C:\Users\massi\Desktop\Ecuador\dcGM_prob_edgelist_thr1\dcGM_prob_edgelist_{}_samples_thr1'.format(p) 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
   
    for i in range(1000):

        sample_mat = np.array(probs>np.random.sample(dim))
        samp_csc = scp.csc_matrix(sample_mat)
        
        scp.save_npz('dcGM_prob_edgelist/dcGM_prob_edgelist_{}_samples/dcGM_prob_edgelist_{}_sample_{}.npz'.format(p,p,i), samp_csc)   
    

def network_sampling(products):
    """
    Parallel execution of sample function to generate synthetic networks for all sectors.
    
    This function parallelizes the generation of synthetic network samples across all
    sectors. It calls the sample function for each sector using all available
    CPU cores to accelerate the computation.
    
    Args:
        products (list): List of ISIC 4-digit codes of sectors to sample.
    
    Returns:
        list: List of results from sample function calls (None values, as sample
              saves results to .npz files rather than returning them).
    """
    num_cores = multiprocessing.cpu_count()
    
    la = Parallel(n_jobs=num_cores)(delayed(sample)(p) for p in products)
    
    return la


























