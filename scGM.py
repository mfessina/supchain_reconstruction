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

'MODEL SOLVED AT THE ISIC 4-DIGIT LEVEL: ONE Z PER LAYER (ISIC CODE) FIXING ITS LINK DENSITY'

#Writes down the equation for z
def z_solver(z, s_in, s_out):
    d = sum(np.multiply.outer(s_in.to_numpy(),s_out.to_numpy()).flatten()/(1+z*np.multiply.outer(s_in.to_numpy(),s_out.to_numpy()).flatten()))
    return d

#Solves iteratively the former equation for z starting from an initial value
def z_solution(a, L, s_in, s_out, n_iterations):
    for i in range(n_iterations):
        x = L/z_solver(a, s_in, s_out)
        a = x
    return a


#Using the two former functions gets z for every sector (ISIC 4-DIGIT CODE) a and computes all the corresponding p_ija, storing them in a csv file

def pij_4dig(p, edge_df, n_iterations):
    temp = edge_df.loc[edge_df['ISIC_prov']==p]
    a = temp.copy()
    L = np.size(temp, axis=0)
    s_in = a.groupby('id_inf')['total_tax'].sum()
    s_in.index = s_in.index.map(int)
    s_out = a.groupby('id_prov')['total_tax'].sum()
    s_out.index = s_out.index.map(int)
    z = z_solution(0.001, L, s_in, s_out, n_iterations)
    f_in = temp['id_inf'].values.astype(int) #all firms in the target nodes column of temp
    f_out = temp['id_prov'].values.astype(int) #all firms in the source nodes column of temp
    
    
    #Saving the probability edgelist for all possible couples of firms to csv file
    
    a_in = temp['id_inf'].unique().astype(int) #unique firms with incoming links
    a_out = temp['id_prov'].unique().astype(int) #unique firms with outgoing links
    mat = np.multiply.outer(z*s_out[a_out].values,s_in[a_in].values)/(1+np.multiply.outer(z*s_out[a_out].values,s_in[a_in].values))
    df = pd.DataFrame(mat, index=list(a_out), columns=list(a_in))
    edge = df.stack().reset_index()
    edge.columns = ['id_prov','id_inf','p_ij']
    edge.to_csv('scgm_prob_edgelist_4dig/scGM_prob_edgelist_{}.csv'.format(p))
    

'MODEL SOLVED ON THE FULL NETWORK: ONE SINGLE Z FIXING THE FULL DENSITY'


def z_full_solver(z, edge_df, products):
    
    d = 0
    
    for p in products:
        a = edge_df.loc[edge_df['ISIC_4']==p]
        s_in = a.groupby('id_inf')['total_tax'].sum()
        s_out = a.groupby('id_prov')['total_tax'].sum()  
        d += sum(np.multiply.outer(s_in.to_numpy(),s_out.to_numpy()).flatten()/(1+z*np.multiply.outer(s_in.to_numpy(),s_out.to_numpy()).flatten()))

    return d

#Solves iteratively the former equation for z starting from an initial value
def z_full_solution(a, edge_df, products):
    
    L = np.size(edge_df,axis=0)
    
    for i in range(200):
        x = L/z_full_solver(a, edge_df, products)
        a = x
    return a

    

def pij_full(p, edge_df, z):
    
    temp = edge_df.loc[edge_df['ISIC_4']==p]
    a = temp.copy()
    s_in = a.groupby('id_inf')['total_tax'].sum()
    s_in.index = s_in.index.map(int)
    s_out = a.groupby('id_prov')['total_tax'].sum()
    s_out.index = s_out.index.map(int)
    f_in = temp['id_inf'].values.astype(int) #all firms in the incoming links column of temp
    f_out = temp['id_prov'].values.astype(int) #all firms in the outgoing links column of temp
    
    #Saving pij on the original edgelist for couples of firms appearing there
    
    a.loc[:,'p_ij_full'] = s_out[f_out].values*s_in[f_in].values*z/(1+s_out[f_out].values*s_in[f_in].values*z)  
    
    #Saving the probability edgelist for all possible couples of firms to csv file
    
    a_in = temp['id_inf'].unique().astype(int) #unique firms with incoming links
    a_out = temp['id_prov'].unique().astype(int) #unique firms with outgoing links
    mat = np.multiply.outer(z*s_out[a_out].values,s_in[a_in].values)/(1+np.multiply.outer(z*s_out[a_out].values,s_in[a_in].values))
    df = pd.DataFrame(mat, index=list(a_out), columns=list(a_in))
    edge = df.stack().reset_index()
    edge.columns = ['id_prov','id_inf','p_ij']
    edge.to_csv('scgm_full_prob_edgelists_2008/scGM_full_prob_edgelist_{}.csv'.format(p))
    
    print(p)
    return a

'FOLLOWING FUNCTIONS ARE DEFINED FOR BOTH CASES'
    
#Parallelizes the computation of the p_ija for every product: the result is a list of product-wise dataframes, 
#which have to be concatenated in the main (see usage in scGM_computation.py) to get the complete original dataframes with the p_ija as last column
def scGM(edge_df,products,aggregation,n_iterations):    
    
    #Launching the pij function in parallel
    num_cores = multiprocessing.cpu_count()
    
    if aggregation == 4:
    
        la = Parallel(n_jobs=num_cores)(delayed(pij_4dig)(p,edge_df,n_iterations) for p in products)
        
    elif aggregation == 0:
        
        #When solving the model with a single z for the whole network, it is better to fit its value
        #before the generation of the ensemble
        
        z = z_full_solution(0.001,edge_df,products)
        
        la = Parallel(n_jobs=num_cores)(delayed(pij_full)(p,edge_df,z) for p in products)
    
    return la

#Function to compute and assign sample weights for the scGM: basically copied by the pij, but in this case it only
#works on the probabilities edgelist already stored in .csv files

def wij(p, edge_df):
    
    temp = edge_df.loc[edge_df['ISIC_prov']==p]
    s_in = temp.groupby('id_inf')['total_tax'].sum()
    s_in.index = s_in.index.map(int)
    s_out = temp.groupby('id_prov')['total_tax'].sum()
    s_out.index = s_out.index.map(int)
    w_tot = np.sum(temp['total_tax'].values)
    sample = pd.read_csv('scgm_prob_edgelist_4dig/scGM_prob_edgelist_{}.csv'.format(p), index_col=0)
    
    #Computing the assigned weight and putting it into the new column w_ij of the edgelist file
    
    f_in = sample['id_inf'].values.astype(int)
    f_out = sample['id_prov'].values.astype(int)
    pijs = sample['p_ij'].values 
    wijs = np.where(pijs!=0, (s_in[f_in].values*s_out[f_out].values)/(w_tot*pijs), 0)
    sample['w_ij'] = wijs
    
    sample.to_csv('scgm_prob_edgelist_4dig/scGM_prob_edgelist_{}.csv'.format(p))    

#Parallel running of the wij function to assign weights to samples
def scGM_weights(edge_df,products):
    
    #Launching the wij function in parallel
    
    num_cores = multiprocessing.cpu_count()
    
    la = Parallel(n_jobs=num_cores)(delayed(wij)(p,edge_df) for p in products)
    
    return la


def sample(p):
    
    #Explicit sampling of a synthetic network from the scGM ensemble.
    #The sampling is done individually for each layer (i.e., product), and, due to memory issues, each of the realized samples
    #is stored in a single .npz file
    #Samples are realized as edgelists, stored in an array: the corresponding nodes have to be traced back
    #through the corresponding scGM_prob_edgelist_{}.csv file
    
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
    
    #Launching the sample function in parallel
    
    num_cores = multiprocessing.cpu_count()
    
    la = Parallel(n_jobs=num_cores)(delayed(sample)(p) for p in products)
    
    return la




