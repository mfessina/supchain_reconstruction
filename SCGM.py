# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:09:16 2023

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

def z_solver(z, s_in, s_out):
    """
    Computes the sum of connection probabilities for the z-solver equation.
    
    This function evaluates the equation used to find the optimal z parameter
    that fixes the link density for a given sector. It computes the sum of all
    pairwise connection probabilities based on incoming and outgoing strengths.
    
    Args:
        z (float): Current value of the z parameter (arbitrary initialization value, set to 0.001 in scGM function).
        s_in (pandas.Series): Series containing incoming strengths (sum of total_tax) 
                             indexed by firm IDs (id_inf).
        s_out (pandas.Series): Series containing outgoing strengths (sum of total_tax) 
                              indexed by firm IDs (id_prov).
    
    Returns:
        float: Sum of all pairwise connection probabilities computed as 
               sum(s_in * s_out / (1 + z * s_in * s_out)).
    """
    d = sum(np.multiply.outer(s_in.to_numpy(),s_out.to_numpy()).flatten()/(1+z*np.multiply.outer(s_in.to_numpy(),s_out.to_numpy()).flatten()))
    return d


def z_solution(a, L, s_in, s_out, n_iterations):
    """
    Solves iteratively for the optimal z parameter starting from an initial value.
    
    This function iteratively refines the z parameter value by solving the equation
    L = sum(s_in * s_out / (1 + z * s_in * s_out)) until convergence. The z parameter
    controls the overall connection probability density for a sector.
    
    Args:
        a (float): Initial guess for the z parameter.
        L (int): Total number of links (edges) in the network for the sector.
        s_in (pandas.Series): Series containing incoming strengths indexed by firm IDs.
        s_out (pandas.Series): Series containing outgoing strengths indexed by firm IDs.
        n_iterations (int): Number of iterations to perform for convergence (10^2 were found to be enough on Ecuadorian network).
    
    Returns:
        float: Converged value of the z parameter after n_iterations.
    """
    for i in range(n_iterations):
        x = L/z_solver(a, s_in, s_out)
        a = x
    return a


def pij_4dig(p, edge_df, n_iterations):
    """
    This function calculates the probability of connection between all pairs of firms within
    a specific ISIC 4-digit sector. It first solves for the optimal z parameter that fixes
    the link density for this sector, then computes all pairwise probabilities and saves
    them to a CSV file.
    
    Args:
        p (str or int): ISIC 4-digit code identifying the sector.
        edge_df (pandas.DataFrame): DataFrame containing network data with columns:
                                   - 'ISIC_prov': ISIC code of the providing sector
                                   - 'id_inf': ID of the firm receiving the connection
                                   - 'id_prov': ID of the firm providing the connection
                                   - 'total_tax': Total tax value (used as strength/weight)
        n_iterations (int): Number of iterations for z parameter convergence.
    
    Returns:
        None: This function saves results to .csv files.
              Output files: 'scgm_prob_edgelist_4dig/scGM_prob_edgelist_{p}.csv'
              Contains columns: 'id_prov', 'id_inf', 'p_ij'
    """
    temp = edge_df.loc[edge_df['ISIC_prov']==p]
    a = temp.copy()
    L = np.size(temp, axis=0)
    s_in = a.groupby('id_inf')['total_tax'].sum()
    s_in.index = s_in.index.map(int)
    s_out = a.groupby('id_prov')['total_tax'].sum()
    s_out.index = s_out.index.map(int)
    z = z_solution(0.001, L, s_in, s_out, n_iterations)
    
    a_in = temp['id_inf'].unique().astype(int)
    a_out = temp['id_prov'].unique().astype(int)
    mat = np.multiply.outer(z*s_out[a_out].values,s_in[a_in].values)/(1+np.multiply.outer(z*s_out[a_out].values,s_in[a_in].values))
    df = pd.DataFrame(mat, index=list(a_out), columns=list(a_in))
    edge = df.stack().reset_index()
    edge.columns = ['id_prov','id_inf','p_ij']
    edge.to_csv('scgm_prob_edgelist_4dig/scGM_prob_edgelist_{}.csv'.format(p))


# ============================================================================
# MODEL SOLVED ON THE FULL NETWORK: ONE SINGLE Z FIXING THE FULL DENSITY
# ============================================================================


def z_full_solver(z, edge_df, products):
    """
    Computes the sum of connection probabilities across all sectors for full network model.
    
    This function evaluates the equation used to find the optimal z parameter for the
    entire network (all sectors combined). It aggregates connection probabilities across
    all products/sectors to compute a single global z parameter.
    
    Args:
        z (float): Current value of the z parameter (connection probability scaling factor).
        edge_df (pandas.DataFrame): DataFrame containing network data with columns:
                                   - 'ISIC_prov': ISIC 4-digit code of supplier firm
                                   - 'id_inf': ID of the client firm (target)
                                   - 'id_prov': ID of the supplier firm (source)
                                   - 'total_tax': weight of the link from 'id_prov' to 'id_inf'
        products (list): List of ISIC 4-digit codes (products/sectors) to include.
    
    Returns:
        float: Sum of all pairwise connection probabilities across all sectors.
    """
    
    d = 0
    
    for p in products:
        a = edge_df.loc[edge_df['ISIC_prov']==p]
        s_in = a.groupby('id_inf')['total_tax'].sum()
        s_out = a.groupby('id_prov')['total_tax'].sum()  
        d += sum(np.multiply.outer(s_in.to_numpy(),s_out.to_numpy()).flatten()/(1+z*np.multiply.outer(s_in.to_numpy(),s_out.to_numpy()).flatten()))

    return d


def z_full_solution(a, edge_df, products, n_iterations):
    """
    Solves iteratively for the optimal z parameter for the full network.
    
    This function iteratively refines the z parameter value for the entire network
    by solving the equation L = sum(s_in * s_out / (1 + z * s_in * s_out)) across
    all sectors until convergence. Uses a fixed 200 iterations.
    
    Args:
        a (float): Initial guess for the z parameter (typically 0.001).
        edge_df (pandas.DataFrame): DataFrame containing network data with columns:
                                   - 'ISIC_prov': ISIC 4-digit code of supplier firm
                                   - 'id_inf': ID of the client firm (target)
                                   - 'id_prov': ID of the supplier firm (source)
                                   - 'total_tax': weight of the link from 'id_prov' to 'id_inf'
        products (list): List of ISIC 4-digit codes of sectors to include in the calculation.
        n_iterations (int): Number of iterations to perform for convergence (10^2 were found to be enough on Ecuadorian network).
    
    Returns:
        float: Converged value of the z parameter after 200 iterations (fixed number of iterations).
    """
    
    L = np.size(edge_df,axis=0)
    
    for i in range(200):
        x = L/z_full_solver(a, edge_df, products)
        a = x
    return a


def pij_full(p, edge_df, z):
    """
    Computes connection probabilities p_ij for a sector using the full network z parameter.
    
    This function calculates the probability of connection between firms in a specific
    sector using a global z parameter computed from the entire network. It computes
    probabilities for the original edgelist and saves a complete probability edgelist
    for all possible firm pairs in the sector.
    
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
              Output files: 'scgm_full_prob_edgelists_2008/scGM_full_prob_edgelist_{p}.csv'
              Contains columns: 'id_prov', 'id_inf', 'p_ij'
    """
    
    temp = edge_df.loc[edge_df['ISIC_prov']==p]
    a = temp.copy()
    s_in = a.groupby('id_inf')['total_tax'].sum()
    s_in.index = s_in.index.map(int)
    s_out = a.groupby('id_prov')['total_tax'].sum()
    s_out.index = s_out.index.map(int)
    f_in = temp['id_inf'].values.astype(int)
    f_out = temp['id_prov'].values.astype(int)
    
    a_in = temp['id_inf'].unique().astype(int)
    a_out = temp['id_prov'].unique().astype(int)
    mat = np.multiply.outer(z*s_out[a_out].values,s_in[a_in].values)/(1+np.multiply.outer(z*s_out[a_out].values,s_in[a_in].values))
    df = pd.DataFrame(mat, index=list(a_out), columns=list(a_in))
    edge = df.stack().reset_index()
    edge.columns = ['id_prov','id_inf','p_ij']
    edge.to_csv('scgm_full_prob_edgelists_2008/scGM_full_prob_edgelist_{}.csv'.format(p))
    
    print(p)


# ============================================================================
# FUNCTIONS DEFINED FOR BOTH MODEL CASES
# ============================================================================


def scGM(edge_df, products, aggregation, n_iterations):
    """
    Main function to compute connection probabilities for all products using parallel processing.
    
    This function parallelizes the computation of connection probabilities (p_ij) between any two firms for every sector in the network:
    - aggregation=4: ISIC 4-digit level (one z per sector)
    - aggregation=0: Full network level (single z for entire network)
    
    The result is a list of product-wise dataframes that containing the link probability between any two firms in the ensemble, stored into separate .csv files.
    
    Args:
        edge_df (pandas.DataFrame): DataFrame containing firm-level network
                                   - 'ISIC_prov': ISIC 4-digit code of supplier firm
                                   - 'id_inf': ID of client firm (target)
                                   - 'id_prov': ID of supplier firm (source)
                                   - 'total_tax': weight of the link from 'id_prov' to 'id_inf'
        products (list): List of ISIC 4-digit codes of sectors to process.
        aggregation (int): Aggregation mode:
                          - 4: ISIC 4-digit level (one z per sector)
                          - 0: Full network level (single z for entire network)
        n_iterations (int): Number of iterations for z parameter convergence 
                           (used only when aggregation=4).
    
    Returns:
        list: List of dataframes (one per sector), where each dataframe contains the
              original edge data for that sector with computed p_ij values. The list
              needs to be concatenated in the calling code to reconstruct the full dataframe.
    """    
    num_cores = multiprocessing.cpu_count()
    
    if aggregation == 4:
        la = Parallel(n_jobs=num_cores)(delayed(pij_4dig)(p,edge_df,n_iterations) for p in products)
        
    elif aggregation == 0:
        # When solving the model with a single z for the whole network, fit z value first
        z = z_full_solution(0.001,edge_df,products)
        
        la = Parallel(n_jobs=num_cores)(delayed(pij_full)(p,edge_df,z) for p in products)
    
    return la


def wij(p, edge_df):
    """
    Computes and assigns sample weights w_ij for the scGM probability edgelist.
    
    This function calculates ensemble weights for edges based on the original data
    and the computed connection probabilities. The weights are computed as:
    w_ij = (s_in * s_out) / (w_tot * p_ij) when p_ij != 0, otherwise 0.
    
    The function reads the probability edgelist from a CSV file, computes weights,
    and saves the updated edgelist with the new w_ij column back to the same file.
    
    Args:
        p (str or int): ISIC 4-digit code identifying the sector to process.
        edge_df (pandas.DataFrame): DataFrame containing network data with columns:
                                   - 'ISIC_prov': ISIC 4-digit code of supplier firm
                                   - 'id_inf': ID of the client firm (target)
                                   - 'id_prov': ID of the supplier firm (source)
                                   - 'total_tax': weight of the link from 'id_prov' to 'id_inf'
    
    Returns:
        None: This function modifies and saves the .csv file but does not return a value.
              Updates file: 'scgm_prob_edgelist_4dig/scGM_prob_edgelist_{p}.csv'
              Adds column: 'w_ij' (ensemble weights)
    """
    temp = edge_df.loc[edge_df['ISIC_prov']==p]
    s_in = temp.groupby('id_inf')['total_tax'].sum()
    s_in.index = s_in.index.map(int)
    s_out = temp.groupby('id_prov')['total_tax'].sum()
    s_out.index = s_out.index.map(int)
    w_tot = np.sum(temp['total_tax'].values)
    sample = pd.read_csv('scgm_prob_edgelist_4dig/scGM_prob_edgelist_{}.csv'.format(p), index_col=0)
    
    f_in = sample['id_inf'].values.astype(int)
    f_out = sample['id_prov'].values.astype(int)
    pijs = sample['p_ij'].values 
    wijs = np.where(pijs!=0, (s_in[f_in].values*s_out[f_out].values)/(w_tot*pijs), 0)
    sample['w_ij'] = wijs
    
    sample.to_csv('scgm_prob_edgelist_4dig/scGM_prob_edgelist_{}.csv'.format(p))


def scGM_weights(edge_df, products):
    """
    Parallel execution of wij function to assign weights to all sectors.
    
    This function parallelizes the computation of ensemble weights across all sectors.
    It calls the wij function for each product using all available CPU cores to accelerate
    the computation.
    
    Args:
        edge_df (pandas.DataFrame): DataFrame containing network data with columns:
                                   - 'ISIC_prov': ISIC 4-digit code of supplier firm
                                   - 'id_inf': ID of the client firm (target)
                                   - 'id_prov': ID of the supplier firm (source)
                                   - 'total_tax': weight of the link from 'id_prov' to 'id_inf'
        products (list): List of ISIC 4-digit codes of sectors to process.
    
    Returns:
        list: List of results from wij function calls (typically None values, as wij
              saves results to .csv files rather than returning them).
    """
    num_cores = multiprocessing.cpu_count()
    
    la = Parallel(n_jobs=num_cores)(delayed(wij)(p,edge_df) for p in products)
    
    return la


def sample(p):
    """
    Explicitly samples synthetic networks from the scGM ensemble for a given product.
    
    This function generates 1000 synthetic network samples from the scGM ensemble for
    a specific product/sector. Sampling is done individually for each layer (product),
    and due to memory constraints, each realized sample is stored in a separate .npz
    file as a compressed sparse matrix (CSC format).
    
    Samples are realized as binary adjacency matrices indicating which edges exist.
    The corresponding nodes can be traced back through the corresponding 
    scGM_prob_edgelist_{p}.csv file which contains the node IDs.
    
    Args:
        p (str or int): ISIC 4-digit code identifying the sector to sample.
    
    Returns:
        None: This function saves samples to .npz files but does not return a value.
              Creates directory: 'scGM_prob_edgelist_4dig/scGM_prob_edgelist_{p}_samples/'
              Saves files: 'scGM_prob_edgelist_{p}_sample_{i}.npz' (i from 0 to 999)
    """
    
    edge_df = pd.read_csv('scGM_prob_edgelist_4dig/scGM_prob_edgelist_{}.csv'.format(p), index_col=0)
    probs = edge_df['p_ij'].to_numpy() 
    dim = np.size(probs)

    
    newpath = r'C:\Users\massi\Desktop\Ecuador\scGM_prob_edgelist_4dig\scGM_prob_edgelist_{}_samples'.format(p) 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
   
    for i in range(1000):

        sample_mat = np.array(probs>np.random.sample(dim))
        samp_csc = scp.csc_matrix(sample_mat)
        
        scp.save_npz('scGM_prob_edgelist_4dig/scGM_prob_edgelist_{}_samples/scGM_prob_edgelist_{}_sample_{}.npz'.format(p,p,i), samp_csc)   
    

def network_sampling(products):
    """
    Parallel execution of sample function to generate synthetic networks for all products.
    
    This function parallelizes the generation of synthetic network samples across all
    products/sectors. It calls the sample function for each product using all available
    CPU cores to accelerate the computation.
    
    Args:
        products (list): List of ISIC codes (products/sectors) to sample.
    
    Returns:
        list: List of results from sample function calls (None values, as sample
              saves results to .npz files rather than returning them).
    """
    
    num_cores = multiprocessing.cpu_count()
    
    la = Parallel(n_jobs=num_cores)(delayed(sample)(p) for p in products)
    
    return la





