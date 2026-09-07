import scipy.io
import urllib.request
import dgl
from dgl.ops import edge_softmax
import math
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from torch.utils import data
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve, average_precision_score, recall_score, confusion_matrix, classification_report, roc_auc_score, f1_score, auc, precision_recall_curve, precision_score
from sklearn.model_selection import KFold
import copy
import pickle
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from random import choice
from collections import Counter
import requests
from zipfile import ZipFile 
import shutil
import warnings
warnings.filterwarnings("ignore")

#device = torch.device("cuda:0")

from .data_splits.datasplit import DataSplitter

        
def preprocess_kg(path, split, test_size = 0.05, one_hop = False, mask_ratio = 0.1):
    if split in ['nervous_system_diseases','endocrine_system_diseases']:        
        name2id = { 
                    'nervous_system_diseases': 'MONDO:0005071',
                    'endocrine_system_diseases':'MONDO:0005151'
                  }
        ds = DataSplitter(kg_path = path)
        test_kg = ds.get_test_kg_for_disease(name2id[split], test_size = test_size, one_hop = one_hop, mask_ratio = mask_ratio)
        all_kg = ds.kg
        all_kg['split'] = 'train'
        test_kg['split'] = 'test'
        df = pd.concat([all_kg, test_kg]).drop_duplicates(subset = ['x_index', 'y_index'], keep = 'last').reset_index(drop = True)
        if test_size != 0.05:
            folder_name = split + '_kg_frac' + str(test_size)
        elif one_hop:
            folder_name = split + '_kg' + '_one_hop_ratio' + str(mask_ratio)
        else:
            folder_name = split + '_kg'        
        path = os.path.join(path, folder_name)        
        os.makedirs(path,exist_ok=True)
        print('save kg to ', os.path.join(path, 'kg.tsv'))
        df.to_csv(os.path.join(path, 'kg.tsv'),sep='\t', index = False)
        df = df[['x_type', 'x_id', 'relation', 'y_type', 'y_id', 'split']]
    else:
        df = pd.read_csv(os.path.join(path, 'kg.tsv'),sep='\t') 
        df = df[['x_type', 'x_id', 'relation', 'y_type', 'y_id']]
    unique_relation = np.unique(df.relation.values) 
    undirected_index = []   
    for i in tqdm(unique_relation):
        if ('_' in i) and (i.split('_')[0] == i.split('_')[1]): 
            # homogeneous graph
            df_temp = df[df.relation == i] 
            df_temp['check_string'] = df_temp.apply(lambda row: '_'.join(sorted([str(row['x_id']), str(row['y_id'])])), axis=1)
            undirected_index.append(df_temp.drop_duplicates('check_string').index.values.tolist()) 
        else:
            d_off = df[df.relation == i] 
            undirected_index.append(d_off[d_off.x_type == d_off.x_type.iloc[0]].index.values.tolist())
    flat_list = [item for sublist in undirected_index for item in sublist] 
    df = df[df.index.isin(flat_list)] 
    unique_node_types = np.unique(np.append(np.unique(df.x_type.values), np.unique(df.y_type.values))) 

    df['x_idx'] = np.nan
    df['y_idx'] = np.nan
    df['x_id'] = df.x_id.apply(lambda x: convert2str(x))
    df['y_id'] = df.y_id.apply(lambda x: convert2str(x))
    for i in tqdm(unique_node_types): 
        names = np.unique(np.append(df[df.x_type == i]['x_id'].values, df[df.y_type == i]['y_id'].values)) 
        names2idx = dict(zip(names, list(range(len(names))))) 
        df.loc[df.x_type == i, 'x_idx'] = df[df.x_type == i]['x_id'].apply(lambda x: names2idx[x])
        df.loc[df.y_type == i, 'y_idx'] = df[df.y_type == i]['y_id'].apply(lambda x: names2idx[x])        
    print('save kg_directed.csv...')
    df.to_csv(os.path.join(path, 'kg_directed.csv'), index = False)

def random_fold(df, fold_seed, frac): 
    train_frac, val_frac, test_frac = frac
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()
    for i in df.relation.unique(): 
        df_temp = df[df.relation == i]
        test = df_temp.sample(frac = test_frac, replace = False, random_state = fold_seed) 
        train_val = df_temp[~df_temp.index.isin(test.index)]

        val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1) 
        train = train_val[~train_val.index.isin(val.index)]

        df_train = pd.concat([df_train,train])
        df_valid = pd.concat([df_valid,val])
        df_test = pd.concat([df_test,test])        
    return {'train': df_train.reset_index(drop = True), 
            'valid': df_valid.reset_index(drop = True),  
            'test': df_test.reset_index(drop = True)}

def disease_eval_fold(df, fold_seed, disease_idx):
    if not isinstance(disease_idx, list):  
        disease_idx = np.array([disease_idx])
    else:
        disease_idx = np.array(disease_idx) 
        
    dd_rel_types = ['CPD_DIS_ass']
    df_not_dd = df[~df.relation.isin(dd_rel_types)]
    df_dd = df[df.relation.isin(dd_rel_types)]
    
    unique_diseases = df_dd.y_idx.unique()
   
    # remove the unique disease out of training
    train_diseases = np.setdiff1d(unique_diseases, disease_idx)
    df_dd_train_val = df_dd[df_dd.y_idx.isin(train_diseases)]                               
    df_dd_test = df_dd[df_dd.y_idx.isin(disease_idx)]
    
    # randomly get 5% disease-compound pairs for validation 
    df_dd_val = df_dd_train_val.sample(frac = 0.05, replace = False, random_state = fold_seed)
    df_dd_train = df_dd_train_val[~df_dd_train_val.index.isin(df_dd_val.index)]
                                       
    df_train = pd.concat([df_not_dd, df_dd_train])
    df_valid = df_dd_val
    df_test = df_dd_test                           
                                   
    return {'train': df_train.reset_index(drop = True), 
            'valid': df_valid.reset_index(drop = True), 
            'test': df_test.reset_index(drop = True)}                      

def complex_disease_fold(df, fold_seed, frac): 
    dd_rel_types = ['CPD_DIS_ass']
    df_not_dd = df[~df.relation.isin(dd_rel_types)] 
    df_dd = df[df.relation.isin(dd_rel_types)]  
    
    unique_diseases = df_dd.y_idx.unique() 
    np.random.seed(fold_seed)
    np.random.shuffle(unique_diseases)
    train, valid, test = np.split(unique_diseases, [int(frac[0]*len(unique_diseases)), int((frac[0] + frac[1])*len(unique_diseases))]) 
    
    df_dd_train = df_dd[df_dd.y_idx.isin(train)] 
    df_dd_valid = df_dd[df_dd.y_idx.isin(valid)] 
    df_dd_test = df_dd[df_dd.y_idx.isin(test)] 
    
    df = df_not_dd 
    train_frac, val_frac, test_frac = frac
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()

    for i in df.relation.unique(): 
        df_temp = df[df.relation == i] 
        test = df_temp.sample(frac = test_frac, replace = False, random_state = fold_seed) 
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1) 
        train = train_val[~train_val.index.isin(val.index)] 
        df_train = pd.concat([df_train,train])
        df_valid = pd.concat([df_valid,val])
        df_test = pd.concat([df_test,test])
    
    df_train = pd.concat([df_train, df_dd_train]) 
    df_valid = pd.concat([df_valid, df_dd_valid]) 
    df_test = pd.concat([df_test, df_dd_test]) 
    return {'train': df_train.reset_index(drop = True), 
            'valid': df_valid.reset_index(drop = True), 
            'test': df_test.reset_index(drop = True)}
        
def few_edeges_to_kg_fold(df, fold_seed, frac):
    
    dd_rel_types = ['CPD_DIS_ass']
    df_not_dd = df[~df.relation.isin(dd_rel_types)]
    df_dd = df[df.relation.isin(dd_rel_types)]
    
    disease2num_neighbors_1 = dict(df_not_dd[df_not_dd.x_type == 'disease'].groupby('x_idx').y_id.agg(len))
    disease2num_neighbors_2 = dict(df_not_dd[df_not_dd.y_type == 'disease'].groupby('y_idx').x_id.agg(len))

    disease2num_neighbors = {}

    # Iterating through keys in both dictionaries
    for key in set(disease2num_neighbors_1).union(disease2num_neighbors_2):
        disease2num_neighbors[key] = disease2num_neighbors_1.get(key, 0) + disease2num_neighbors_2.get(key, 0)
    
    disease_with_less_than_3_connections_in_kg = np.array([i for i,j in disease2num_neighbors.items() if j <= 3])
    unique_diseases = df_dd.y_idx.unique()
    train_val_diseases = np.setdiff1d(unique_diseases, disease_with_less_than_3_connections_in_kg)
    test = np.intersect1d(unique_diseases, disease_with_less_than_3_connections_in_kg) 
    print('Number of testing diseases: ', len(test))
    np.random.seed(fold_seed)
    np.random.shuffle(train_val_diseases)
    train, valid = np.split(train_val_diseases, [int(frac[0]*len(unique_diseases))])
    print('Number of train diseases: ', len(train))
    print('Number of valid diseases: ', len(valid))
    
    df_dd_train = df_dd[df_dd.y_idx.isin(train)]
    df_dd_valid = df_dd[df_dd.y_idx.isin(valid)]
    df_dd_test = df_dd[df_dd.y_idx.isin(test)]
    
    df = df_not_dd
    train_frac, val_frac, test_frac = frac
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()

    for i in df.relation.unique():
        df_temp = df[df.relation == i]
        test = df_temp.sample(frac = test_frac, replace = False, random_state = fold_seed)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = pd.concat([df_train,train])
        df_valid = pd.concat([df_valid,val])
        df_test = pd.concat([df_test,test])
    
    df_train = pd.concat([df_train, df_dd_train])
    df_valid = pd.concat([df_valid, df_dd_valid])
    df_test = pd.concat([df_test, df_dd_test])
    
    return {'train': df_train.reset_index(drop = True), 
            'valid': df_valid.reset_index(drop = True), 
            'test': df_test.reset_index(drop = True)}
    
    
def few_edeges_to_CPD_DIS_ass_fold(df, fold_seed, frac):
    
    dd_rel_types = ['CPD_DIS_ass']
    df_not_dd = df[~df.relation.isin(dd_rel_types)]
    df_dd = df[df.relation.isin(dd_rel_types)]
    
    disease2num_CPD_DIS_asss = dict(df_dd[(df_dd.y_type == 'disease') & (df_dd.relation == 'CPD_DIS_ass')].groupby('x_idx').y_id.agg(len))
    
    disease_with_less_than_3_CPD_DIS_asss_in_kg = np.array([i for i,j in disease2num_CPD_DIS_asss.items() if j <= 3])
    unique_diseases = df_dd.y_idx.unique()
    train_val_diseases = np.setdiff1d(unique_diseases, disease_with_less_than_3_CPD_DIS_asss_in_kg)
    test = np.intersect1d(unique_diseases, disease_with_less_than_3_CPD_DIS_asss_in_kg) 
    print('Number of testing diseases: ', len(test))
    np.random.seed(fold_seed)
    np.random.shuffle(train_val_diseases)
    train, valid = np.split(train_val_diseases, [int(frac[0]*len(unique_diseases))])
    print('Number of train diseases: ', len(train))
    print('Number of valid diseases: ', len(valid))
    
    df_dd_train = df_dd[df_dd.y_idx.isin(train)]
    df_dd_valid = df_dd[df_dd.y_idx.isin(valid)]
    df_dd_test = df_dd[df_dd.y_idx.isin(test)]
    
    df = df_not_dd
    train_frac, val_frac, test_frac = frac
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()

    for i in df.relation.unique():
        df_temp = df[df.relation == i]
        test = df_temp.sample(frac = test_frac, replace = False, random_state = fold_seed)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = pd.concat([df_train,train])
        df_valid = pd.concat([df_valid,val])
        df_test = pd.concat([df_test,test])
    
    df_train = pd.concat([df_train, df_dd_train])
    df_valid = pd.concat([df_valid, df_dd_valid])
    df_test = pd.concat([df_test, df_dd_test])
    
    return {'train': df_train.reset_index(drop = True), 
            'valid': df_valid.reset_index(drop = True), 
            'test': df_test.reset_index(drop = True)}
    
def create_fold_cv(df, split_num, num_splits): 
    dd_rel_types = ['CPD_DIS_ass']
    df_not_dd = df[~df.relation.isin(dd_rel_types)]
    df_dd = df[df.relation.isin(dd_rel_types)] 
    
    unique_diseases = df_dd.y_idx.unique() 
    np.random.seed(42)
    np.random.shuffle(unique_diseases)
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=num_splits)
    split_num_idx = {}
    for i, (train_index, test_index) in enumerate(kf.split(unique_diseases)):  
        train_index, valid_index, _ = np.split(train_index, [int(0.9*len(train_index)), int(len(train_index))])
        split_num_idx[i+1] = {'train': unique_diseases[train_index],
                              'valid': unique_diseases[valid_index],
                              'test': unique_diseases[test_index]}
        
    train, valid, test = split_num_idx[split_num]['train'], split_num_idx[split_num]['valid'], split_num_idx[split_num]['test']
    df_dd_train = df_dd[df_dd.y_idx.isin(train)]
    df_dd_valid = df_dd[df_dd.y_idx.isin(valid)]
    df_dd_test = df_dd[df_dd.y_idx.isin(test)]
    
    df = df_not_dd
    train_frac, val_frac, test_frac = [0.83125, 0.11875, 0.05]
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()
    for i in df.relation.unique():
        df_temp = df[df.relation == i]
        test = df_temp.sample(frac = test_frac, replace = False, random_state = split_num)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = pd.concat([df_train,train])
        df_valid = pd.concat([df_valid,val])
        df_test = pd.concat([df_test,test])
    
    df_train = pd.concat([df_train, df_dd_train])
    df_valid = pd.concat([df_valid, df_dd_valid])
    df_test = pd.concat([df_test, df_dd_test])
    return df_train.reset_index(drop = True), df_valid.reset_index(drop = True), df_test.reset_index(drop = True)  

def create_fold_cv5(datafolder, df, num_splits, foldid): 
    if foldid < 1 or foldid > num_splits:
        raise ValueError(f"foldid must be in [1, {num_splits}], got {foldid}")
    dd_rel_types = ['CPD_DIS_ass']
    df_not_dd_dp = df[~((df.relation.isin(dd_rel_types)))]
    df_dd = df[df.relation.isin(dd_rel_types)] 
    np.random.seed(42)
    df_dd = df_dd.sample(frac=1, random_state=42).reset_index(drop=True)
    df_not_dd_dp = df_not_dd_dp.reset_index(drop=True)

    kf = KFold(n_splits = num_splits)
    split_num_idx_dd = {}
    for i, (train_index, test_index) in enumerate(kf.split(df_dd)): 
        train_index, valid_index, _ = np.split(train_index, [int(0.9*len(train_index)), int(len(train_index))])
        split_num_idx_dd[i+1] = {'train': df_dd.iloc[train_index],'valid': df_dd.iloc[valid_index],'test': df_dd.iloc[test_index]}  

    train_frac, val_frac, test_frac = [0.85, 0.1, 0.05]
    df_train_rest = pd.DataFrame()
    df_valid_rest = pd.DataFrame()
    df_test_rest = pd.DataFrame()
    for i in df_not_dd_dp.relation.unique():
        df_temp = df_not_dd_dp[df_not_dd_dp.relation == i]
        test = df_temp.sample(frac = test_frac, replace = False, random_state = 1)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train_rest = pd.concat([df_train_rest,train],axis=0)
        df_valid_rest = pd.concat([df_valid_rest,val],axis=0)
        df_test_rest = pd.concat([df_test_rest,test],axis=0)
    
    for split_num in range(1, num_splits + 1):
        df_dd_train, df_dd_valid, df_dd_test = split_num_idx_dd[split_num]['train'], split_num_idx_dd[split_num]['valid'], split_num_idx_dd[split_num]['test']
        df_train = pd.concat([df_dd_train, df_train_rest],ignore_index=True,axis=0).reset_index(drop = True)
        df_valid = pd.concat([df_dd_valid,df_valid_rest],ignore_index=True,axis=0).reset_index(drop = True)
        df_test = pd.concat([df_dd_test,df_test_rest],ignore_index=True,axis=0).reset_index(drop = True)

        unique_rel = df[['x_type', 'relation', 'y_type']].drop_duplicates() 
        df_train = reverse_rel_generation(df, df_train, unique_rel)  
        df_valid = reverse_rel_generation(df, df_valid, unique_rel) 
        df_test = reverse_rel_generation(df, df_test, unique_rel)
        df_train.to_csv(os.path.join(datafolder,'cv'+str(split_num),'train.csv'), index = False)
        df_valid.to_csv(os.path.join(datafolder,'cv'+str(split_num), 'valid.csv'), index = False)
        df_test.to_csv(os.path.join(datafolder,'cv'+str(split_num), 'test.csv'), index = False)    
        if split_num == foldid:
            selected_train = df_train
            selected_valid = df_valid
            selected_test = df_test 
    return selected_train, selected_valid, selected_test 

def _split_entities(entity_ids, fold_seed=42, num_splits=5, valid_frac=0.1):
    """Return dict: fold_id -> {'train': ids, 'valid': ids, 'test': ids}."""
    from sklearn.model_selection import KFold
    entity_ids = np.array(sorted(pd.Series(entity_ids).dropna().unique()))
    if len(entity_ids) < num_splits:
        raise ValueError(f'Not enough entities for {num_splits}-fold cold-start split: {len(entity_ids)}')

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=fold_seed)
    out = {}
    for fold_id, (train_valid_idx, test_idx) in enumerate(kf.split(entity_ids), start=1):
        train_valid_ids = entity_ids[train_valid_idx].copy()
        test_ids = entity_ids[test_idx].copy()
        rng = np.random.RandomState(fold_seed + fold_id)
        rng.shuffle(train_valid_ids)
        n_valid = max(1, int(round(valid_frac * len(train_valid_ids))))
        valid_ids = train_valid_ids[:n_valid]
        train_ids = train_valid_ids[n_valid:]
        out[fold_id] = {'train': train_ids,'valid': valid_ids,'test': test_ids}
    return out

def _filter_edges_touching_entities(df_in, node_type, entity_ids):
    entity_ids = set(pd.Series(list(entity_ids)).dropna().astype(float).tolist())
    return df_in[
        ((df_in.x_type == node_type) & (df_in.x_idx.astype(float).isin(entity_ids))) |
        ((df_in.y_type == node_type) & (df_in.y_idx.astype(float).isin(entity_ids)))
    ]

def _assert_cold_split(df_train, df_valid, df_test, split, target_rel='CPD_DIS_ass'):
    if split == 'cold_cpd':
        train_ids = set(df_train[(df_train.relation == target_rel) & (df_train.x_type == 'compound')].x_idx.dropna().astype(float))
        valid_ids = set(df_valid[(df_valid.relation == target_rel) & (df_valid.x_type == 'compound')].x_idx.dropna().astype(float))
        test_ids = set(df_test[(df_test.relation == target_rel) & (df_test.x_type == 'compound')].x_idx.dropna().astype(float))
        name = 'compound'
    elif split == 'cold_dis':
        train_ids = set(df_train[(df_train.relation == target_rel) & (df_train.y_type == 'disease')].y_idx.dropna().astype(float))
        valid_ids = set(df_valid[(df_valid.relation == target_rel) & (df_valid.y_type == 'disease')].y_idx.dropna().astype(float))
        test_ids = set(df_test[(df_test.relation == target_rel) & (df_test.y_type == 'disease')].y_idx.dropna().astype(float))
        name = 'disease'
    else:
        return
    assert len(train_ids & valid_ids) == 0, f'{split}: train/valid {name} leakage in {target_rel}'
    assert len(train_ids & test_ids) == 0, f'{split}: train/test {name} leakage in {target_rel}'
    assert len(valid_ids & test_ids) == 0, f'{split}: valid/test {name} leakage in {target_rel}'

def create_cold_start_cv5(datafolder, df, split, foldid, fold_seed=42, target_rel='CPD_DIS_ass', strict_cold=False, valid_frac=0.1):
    if foldid < 1 or foldid > 5:
        raise ValueError(f'foldid must be in [1, 5], got {foldid}')
    if split not in ['cold_cpd', 'cold_dis']:
        raise ValueError(f'Unsupported cold-start split: {split}')

    df_target = df[df.relation == target_rel].copy().reset_index(drop=True)
    df_rest = df[df.relation != target_rel].copy().reset_index(drop=True)
    if df_target.empty:
        raise ValueError(f'No target relation {target_rel} found in df.')

    if split == 'cold_cpd':
        node_type = 'compound'
        entity_ids = df_target.loc[df_target.x_type == 'compound', 'x_idx'].dropna().astype(float).unique()
        id_col = 'x_idx'
    else:
        node_type = 'disease'
        entity_ids = df_target.loc[df_target.y_type == 'disease', 'y_idx'].dropna().astype(float).unique()
        id_col = 'y_idx'

    fold_entities = _split_entities(entity_ids, fold_seed=fold_seed, num_splits=5, valid_frac=valid_frac)
    unique_rel = df[['x_type', 'relation', 'y_type']].drop_duplicates()
    selected_train = selected_valid = selected_test = None
    for split_num in range(1, 6):
        train_ids = set(fold_entities[split_num]['train'].astype(float))
        valid_ids = set(fold_entities[split_num]['valid'].astype(float))
        test_ids = set(fold_entities[split_num]['test'].astype(float))
        df_target_train = df_target[df_target[id_col].astype(float).isin(train_ids)]
        df_target_valid = df_target[df_target[id_col].astype(float).isin(valid_ids)]
        df_target_test = df_target[df_target[id_col].astype(float).isin(test_ids)]

        if strict_cold:
            heldout_ids = valid_ids | test_ids
            df_rest_train = df_rest.drop(_filter_edges_touching_entities(df_rest, node_type, heldout_ids).index)
            df_rest_valid = df_rest.iloc[0:0].copy()
            df_rest_test = df_rest.iloc[0:0].copy()
        else:
            df_rest_train = df_rest
            df_rest_valid = df_rest.iloc[0:0].copy()
            df_rest_test = df_rest.iloc[0:0].copy()
        df_train = pd.concat([df_rest_train, df_target_train], ignore_index=True).reset_index(drop=True)
        df_valid = pd.concat([df_rest_valid, df_target_valid], ignore_index=True).reset_index(drop=True)
        df_test = pd.concat([df_rest_test, df_target_test], ignore_index=True).reset_index(drop=True)
        _assert_cold_split(df_train, df_valid, df_test, split, target_rel=target_rel)
        df_train = reverse_rel_generation(df, df_train, unique_rel)
        df_valid = reverse_rel_generation(df, df_valid, unique_rel)
        df_test = reverse_rel_generation(df, df_test, unique_rel)
        df_train.to_csv(os.path.join(datafolder,'cv'+str(split_num), 'train.csv'), index=False)
        df_valid.to_csv(os.path.join(datafolder,'cv'+str(split_num), 'valid.csv'), index=False)
        df_test.to_csv(os.path.join(datafolder,'cv'+str(split_num), 'test.csv'), index=False)

        if split_num == foldid:
            selected_train, selected_valid, selected_test = df_train, df_valid, df_test
    return selected_train, selected_valid, selected_test

def create_fold(df, fold_seed = 100, frac = [0.7, 0.1, 0.2], method = 'random', disease_idx = 0.0):
    if method == 'random':
        out = random_fold(df, fold_seed, frac) 
    elif method == 'complex_disease':
        out = complex_disease_fold(df, fold_seed, frac) 
        out = few_edeges_to_kg_fold(df, fold_seed, [0.7, 0.1, 0.2])
    elif method == 'few_edeges_to_CPD_DIS_ass':
        out = few_edeges_to_CPD_DIS_ass_fold(df, fold_seed, [0.7, 0.1, 0.2])
    elif method == 'downstream_pred':
        out = disease_eval_fold(df, fold_seed, disease_idx)        
    elif method == 'disease_eval':
        out = disease_eval_fold(df, fold_seed, disease_idx)
    elif method == 'full_graph':
        out = random_fold(df, fold_seed, [0.95, 0.05, 0.0])
        out['test'] = out['valid'] 
    else:
        # disease split
        train_val = df[df.split == 'train'].reset_index(drop = True)
        test = df[df.split == 'test'].reset_index(drop = True)
        out = random_fold(train_val, fold_seed, [0.8, 0.2, 0.0])
        out['test'] = test
    return out['train'], out['valid'], out['test'] 


def create_split(data_folder, df, split, disease_eval_index, split_data_path, seed, fold):
    if split == 'complex_disease_cv':
        if seed < 1 or seed > 20:
            raise ValueError('Complex disease cross validation 20 folds, select seed from 1-20.')
        df_train, df_valid, df_test = create_fold_cv(df, split_num = seed, num_splits = 20)
    elif split == 'cv5':
        df_train, df_valid, df_test = create_fold_cv5(data_folder, df, num_splits = 5, foldid=fold)
    elif split in ['cold_cpd', 'cold_dis']:
        df_train, df_valid, df_test = create_cold_start_cv5(data_folder, df, split=split, foldid=fold, fold_seed=seed,
                                                            target_rel='CPD_DIS_ass', strict_cold=False, valid_frac=0.1)
    else:
        df_train, df_valid, df_test = create_fold(df, fold_seed = seed, frac = [0.8, 0.1, 0.1], method = split, disease_idx = disease_eval_index) 
   
    if split not in ['cv5', 'cold_cpd', 'cold_dis']:
        unique_rel = df[['x_type', 'relation', 'y_type']].drop_duplicates() 
        df_train = reverse_rel_generation(df, df_train, unique_rel) 
        df_valid = reverse_rel_generation(df, df_valid, unique_rel) 
        df_test = reverse_rel_generation(df, df_test, unique_rel)
        df_train.to_csv(os.path.join(split_data_path, 'train.csv'), index = False)
        df_valid.to_csv(os.path.join(split_data_path, 'valid.csv'), index = False)
        df_test.to_csv(os.path.join(split_data_path, 'test.csv'), index = False)
    return df_train, df_valid, df_test
    
def construct_negative_graph_each_etype(graph, k, etype, method, weights, device):
    utype, _, vtype = etype 
    src, dst = graph.edges(etype=etype)
    
    if method == 'corrupt_dst':
        neg_src = src.repeat_interleave(k)
        neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,))
    elif method == 'corrupt_src':
        neg_dst = dst.repeat_interleave(k)
        neg_src = torch.randint(0, graph.number_of_nodes(utype), (len(dst) * k,))
    elif method == 'corrupt_both':
        neg_src = torch.randint(0, graph.number_of_nodes(utype), (len(dst) * k,))
        neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,))
    elif (method == 'multinomial_src') or (method == 'inverse_src') or (method == 'fix_src'):
        neg_dst = dst.repeat_interleave(k)
        try:
            neg_src = weights[etype].multinomial(len(neg_dst), replacement=True)
        except:
            neg_src = torch.Tensor([])
    elif (method == 'multinomial_dst') or (method == 'inverse_dst') or (method == 'fix_dst'):
        neg_src = src.repeat_interleave(k) 
        try:
            neg_dst = weights[etype].multinomial(len(neg_src), replacement=True) 
        except:
            neg_dst = torch.Tensor([])
    return {etype: (neg_src.to(device), neg_dst.to(device))}

def construct_negative_graph(graph, k, device):
    out = {}   
    for etype in graph.canonical_etypes:
        out.update(construct_negative_graph_each_etype(graph, k, etype, device))
    return dgl.heterograph(out, num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})

class Minibatch_NegSampler(object):
    def __init__(self, g, k, method): 
        if method == 'multinomial_dst':
            self.weights = {
                etype: g.in_degrees(etype=etype).float() ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == 'fix_dst':  
            self.weights = {
                etype: (g.in_degrees(etype=etype) > 0).float()
                for etype in g.canonical_etypes
            } 
        self.k = k

    def __call__(self, g, eids_dict):  
        result_dict = {}
        for etype, eids in eids_dict.items():
            src, _ = g.find_edges(eids, etype=etype) 
            src = src.repeat_interleave(self.k) 
            dst = self.weights[etype].multinomial(len(src), replacement=True)  
            result_dict[etype] = (src, dst) 
        return result_dict


def edge_sets_build(df_list):
    required_cols = ["x_type", "relation", "y_type", "x_idx", "y_idx"]
    t_edge_sets = {}
    for df_id, df in enumerate(df_list):
        if df is None or len(df) == 0:
            continue
        missing_cols = [c for c in required_cols if c not in df.columns]
        if len(missing_cols) > 0:
            raise ValueError(f"df_list[{df_id}] is missing required columns: {missing_cols}. ")
        df_use = (df[required_cols].dropna(subset=["x_type", "relation", "y_type", "x_idx", "y_idx"]).copy())
        df_use["x_type"] = df_use["x_type"].astype(str)
        df_use["relation"] = df_use["relation"].astype(str)
        df_use["y_type"] = df_use["y_type"].astype(str)
        df_use["x_idx"] = df_use["x_idx"].astype(np.int64)
        df_use["y_idx"] = df_use["y_idx"].astype(np.int64)

        for (x_type, rel, y_type), df_rel in df_use.groupby(["x_type", "relation", "y_type"],sort=False):
            etype = (x_type, rel, y_type)
            src = df_rel["x_idx"].to_numpy(dtype=np.int64)
            dst = df_rel["y_idx"].to_numpy(dtype=np.int64)
            edge_set = t_edge_sets.setdefault(etype, set())
            edge_set.update(zip(src.tolist(), dst.tolist()))
    total_num = sum(len(edge_set) for edge_set in t_edge_sets.values())
    return t_edge_sets


class Full_Graph_NegSamples:
    def __init__(self,g,k,method,device,t_edge_sets=None,max_tries=100,verbose=True):
        self.k = int(k)
        self.method = method
        self.device = device
        self.t_edge_sets = t_edge_sets if t_edge_sets is not None else {}
        self.max_tries = int(max_tries)
        self.verbose = verbose

        if method == "multinomial_src":
            self.weights = {
                etype: g.out_degrees(etype=etype).float() ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == "multinomial_dst":
            self.weights = {
                etype: g.in_degrees(etype=etype).float() ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == "inverse_dst":
            self.weights = {
                etype: -g.in_degrees(etype=etype).float() ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == "inverse_src":
            self.weights = {
                etype: -g.out_degrees(etype=etype).float() ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == "fix_dst":
            self.weights = {
                etype: (g.in_degrees(etype=etype) > 0).float()
                for etype in g.canonical_etypes
            }
        elif method == "fix_src":
            self.weights = {
                etype: (g.out_degrees(etype=etype) > 0).float()
                for etype in g.canonical_etypes
            }
        else:
            raise ValueError(f"Unsupported negative sampling method: {method}")

        self.weights = {k: v.detach().cpu() for k, v in self.weights.items()}

    def _sample_fix_dst_each_etype(self, graph, etype):
        src, _ = graph.edges(etype=etype)
        src = src.detach().cpu().long()
        if src.numel() == 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        neg_src = src.repeat_interleave(self.k)
        n_need = neg_src.numel()
        weights = self.weights[etype].clone().float()
        if weights.sum().item() <= 0:
            if self.verbose:
                print(f"[Warning] {etype}: no dst candidates with in_degree > 0.")
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        true_set = self.t_edge_sets.get(etype, set())
        neg_dst = torch.empty(n_need, dtype=torch.long)
        unresolved = torch.ones(n_need, dtype=torch.bool)

        for _ in range(self.max_tries):
            idx = torch.where(unresolved)[0]
            if idx.numel() == 0:
                break
            sampled_dst = weights.multinomial(idx.numel(), replacement=True).long()
            sampled_src = neg_src[idx]
            valid_mask = []
            for s, d in zip(sampled_src.tolist(), sampled_dst.tolist()):
                valid_mask.append((int(s), int(d)) not in true_set)
            valid_mask = torch.tensor(valid_mask, dtype=torch.bool)

            if valid_mask.any():
                valid_idx = idx[valid_mask]
                neg_dst[valid_idx] = sampled_dst[valid_mask]
                unresolved[valid_idx] = False

        if unresolved.any():
            candidate_dst = torch.where(weights > 0)[0].long().tolist()
            unresolved_idx = torch.where(unresolved)[0].tolist()
            for pos in unresolved_idx:
                s = int(neg_src[pos].item())
                valid_candidates = [d for d in candidate_dst if (s, int(d)) not in true_set]
                if len(valid_candidates) == 0:
                    raise RuntimeError(
                                        f"{etype}: src={s} has no valid negative dst. "
                                        f"All candidate dst nodes are true positive edges. "
                                        f"Cannot generate same number of negatives."
                                    )

                j = torch.randint(low=0,high=len(valid_candidates),size=(1,)).item()
                neg_dst[pos] = int(valid_candidates[j])
                unresolved[pos] = False
        assert neg_src.numel() == neg_dst.numel()
        assert neg_src.numel() == graph.num_edges(etype=etype) * self.k
        return neg_src.to(self.device), neg_dst.to(self.device)

    def __call__(self, graph):
        out = {}
        for etype in graph.canonical_etypes:
            if graph.num_edges(etype=etype) == 0:
                continue
            if self.method == "fix_dst":
                neg_src, neg_dst = self._sample_fix_dst_each_etype(graph, etype)
                if neg_src.numel() > 0:
                    out[etype] = (neg_src, neg_dst)
            else:
                temp = construct_negative_graph_each_etype(graph,self.k,etype,self.method,self.weights,self.device)
                if len(temp[etype][0]) != 0:
                    out.update(temp)
        return dgl.heterograph(out,num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})

class Full_Graph_NegSampler:
    def __init__(self, g, k, method, device): 
        if method == 'multinomial_src':
            self.weights = {
                etype: g.out_degrees(etype=etype).float() ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == 'multinomial_dst':
            self.weights = {
                etype: g.in_degrees(etype=etype).float() ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == 'inverse_dst':
            self.weights = {
                etype: -g.in_degrees(etype=etype).float() ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == 'inverse_src':
            self.weights = {
                etype: -g.out_degrees(etype=etype).float() ** 0.75
                for etype in g.canonical_etypes
            }
        elif method == 'fix_dst':
            self.weights = {
                etype: (g.in_degrees(etype=etype) > 0).float()
                for etype in g.canonical_etypes
            } 
        elif method == 'fix_src':
            self.weights = {
                etype: (g.out_degrees(etype=etype) > 0).float()
                for etype in g.canonical_etypes
            }
        else:
            self.weights = {}
            
        self.k = k
        self.method = method
        self.device = device
    def __call__(self, graph): 
        out = {}   
        for etype in graph.canonical_etypes: 
            temp = construct_negative_graph_each_etype(graph, self.k, etype, self.method, self.weights, self.device) 
            if len(temp[etype][0]) != 0:
                out.update(temp)            
        return dgl.heterograph(out, num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})

def eval_graph_construct(df_eval,g,neg_sampler, k,device,t_edge_sets=None):
    out = {}
    df_in = df_eval[["x_idx", "relation", "y_idx"]].dropna(subset=["x_idx", "y_idx"]).copy()
    for etype in g.canonical_etypes:
        try:
            df_temp = df_in[df_in.relation == etype[1]] 
            src = torch.Tensor(df_temp.x_idx.values).to(device).to(dtype = torch.int64) 
            dst = torch.Tensor(df_temp.y_idx.values).to(device).to(dtype = torch.int64)
            out.update({etype: (src, dst)}) 
        except:
            print(etype[1])  
    g_pos = dgl.heterograph(out,num_nodes_dict={ntype: g.number_of_nodes(ntype) for ntype in g.ntypes},).to(device)

    ng = Full_Graph_NegSamples(g=g_pos,k=k,method=neg_sampler,device=device,t_edge_sets=t_edge_sets,max_tries=1000000,verbose=True)
    g_neg = ng(g_pos)
    return g_pos, g_neg
        
def evaluate_graph_construct(df_valid, g, neg_sampler, k, device): 
    out = {}
    df_in = df_valid[['x_idx', 'relation', 'y_idx']] 
    for etype in g.canonical_etypes: 
        try:
            df_temp = df_in[df_in.relation == etype[1]] 
            src = torch.Tensor(df_temp.x_idx.values).to(device).to(dtype = torch.int64) 
            dst = torch.Tensor(df_temp.y_idx.values).to(device).to(dtype = torch.int64) 
            out.update({etype: (src, dst)})
        except:
            print(etype[1])       
    g_valid = dgl.heterograph(out, num_nodes_dict={ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}) 
    
    ng = Full_Graph_NegSampler(g_valid, k, neg_sampler, device) 
    g_neg_valid = ng(g_valid) 
    return g_valid, g_neg_valid

def graphs_construct(data_path,device):
    df_kg = pd.read_csv(os.path.join(data_path,'kg.csv'))
    df_kg['x_idx'] = df_kg['x_idx'].astype(int)
    df_kg['y_idx'] = df_kg['y_idx'].astype(int)
    device = device
    data_path = data_path
    edge_sets = edge_sets_build([df_kg])
    g = create_graph(df_kg)

    df_train = pd.read_csv(os.path.join(data_path,'train.csv'))
    df_train['x_idx'] = df_train['x_idx'].astype(int)
    df_train['y_idx'] = df_train['y_idx'].astype(int)
    g_train_pos, g_train_neg = eval_graph_construct(df_train,g,neg_sampler="fix_dst",k=1,device=device,t_edge_sets=edge_sets)
    rows = []
    for etype in g_train_neg.canonical_etypes:
        src, dst = g_train_neg.edges(etype=etype)   
        for s, d in zip(src.cpu().tolist(), dst.cpu().tolist()):
            rows.append([etype[0],etype[1], etype[2],s, d])
    df_train_neg = pd.DataFrame(rows, columns=['x_type','relation','y_type', 'x_idx', 'y_idx'])
    df_train_neg.drop_duplicates(inplace=True)
    df_train_neg = df_train_neg.reset_index(drop=True)
    false_edge_sets = edge_sets_build([df_train_neg])
    for etype, edge_set in false_edge_sets.items():
        if etype in edge_sets:
            edge_sets[etype].update(edge_set)  
        else:
            edge_sets[etype] = edge_set.copy()  

    df_valid = pd.read_csv(os.path.join(data_path,'valid.csv'))
    df_valid['x_idx'] = df_valid['x_idx'].astype(int)
    df_valid['y_idx'] = df_valid['y_idx'].astype(int)
    g_valid_pos, g_valid_neg = eval_graph_construct(df_valid,g,neg_sampler="fix_dst",k=1,device=device,t_edge_sets=edge_sets)
    rows = []
    for etype in g_valid_neg.canonical_etypes:
        src, dst = g_valid_neg.edges(etype=etype)   
        for s, d in zip(src.cpu().tolist(), dst.cpu().tolist()):
            rows.append([etype[0],etype[1], etype[2],s, d])
    df_valid_neg = pd.DataFrame(rows, columns=['x_type','relation','y_type', 'x_idx', 'y_idx'])
    df_valid_neg.drop_duplicates(inplace=True)
    df_valid_neg = df_valid_neg.reset_index(drop=True)
    false_edge_sets = edge_sets_build([df_valid_neg])
    for etype, edge_set in false_edge_sets.items():
        if etype in edge_sets:
            edge_sets[etype].update(edge_set)   
        else:
            edge_sets[etype] = edge_set.copy() 

    df_test = pd.read_csv(os.path.join(data_path,'test.csv'))
    df_test['x_idx'] = df_test['x_idx'].astype(int)
    df_test['y_idx'] = df_test['y_idx'].astype(int)
    g_test_pos, g_test_neg = eval_graph_construct(df_test,g,neg_sampler="fix_dst",k=1,device=device,t_edge_sets=edge_sets)
    return g_train_pos, g_train_neg, g_valid_pos, g_valid_neg, g_test_pos, g_test_neg

def predict_from_alpha(alpha, eps=1e-8):
    S = torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / (S)
    evidence = alpha - 1.0
    uncertainty = alpha.shape[1] / (S.squeeze(1))
    belief = evidence / (S)

    prob_pos = prob[:, 1]
    pred = (prob_pos >= 0.5).long()
    # pred = torch.argmax(prob, dim=1)
    return prob_pos, pred, uncertainty, evidence, belief

def get_all_metrics_fb_evi(pred_score_pos, pred_score_neg, scores, labels, G, full_mode = False):
    auroc_rel = {}
    auprc_rel = {}
    if full_mode: 
        etypes = G.canonical_etypes
    else:
        etypes = [ ('compound', 'CPD_DIS_ass', 'disease'),    
                  ('disease', 'rev_CPD_DIS_ass', 'compound')]
             
    for etype in etypes:    
        try:
            alpha_pos = pred_score_pos[etype] 
            alpha_neg = pred_score_neg[etype] 
            alphas = torch.cat([alpha_pos, alpha_neg],dim=0)
            prob_pos, _, _, _, _ = predict_from_alpha(alphas)
            prob_np = prob_pos.detach().cpu().numpy()
            y_np = np.concatenate([np.ones(alpha_pos.shape[0], dtype=int),np.zeros(alpha_neg.shape[0], dtype=int)])     
            auroc_rel[etype] = roc_auc_score(y_np, prob_np) 
            auprc_rel[etype] = average_precision_score(y_np, prob_np) 
        except:
            pass

    micro_auroc = roc_auc_score(labels, scores)
    micro_auprc = average_precision_score(labels, scores)
    macro_auroc = np.mean(list(auroc_rel.values()))
    macro_auprc = np.mean(list(auprc_rel.values()))
    return auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc

def get_all_metrics(y, pred, rels):
    edge_dict_ = {v:k for k,v in edge_dict.items()}

    auroc_rel = {}
    auprc_rel = {}
    for rel in np.unique(rels):
        index = np.where(rels == rel)
        y_ = y[index]
        pred_ = pred[index]
        try:
            auroc_rel[edge_dict_[rel]] = roc_auc_score(y_, pred_)
            auprc_rel[edge_dict_[rel]] = average_precision_score(y_, pred_)
        except:
            pass
    micro_auroc = roc_auc_score(y, pred)
    micro_auprc = average_precision_score(y, pred)
    macro_auroc = np.mean(list(auroc_rel.values()))
    macro_auprc = np.mean(list(auprc_rel.values()))
    
    return auroc_rel, auprc_rel, micro_auroc, \
            micro_auprc, macro_auroc, macro_auprc

def get_all_metrics_fb(pred_score_pos, pred_score_neg, scores, labels, G, full_mode = False):
    auroc_rel = {}
    auprc_rel = {}
    if full_mode: 
        etypes = G.canonical_etypes
    else:
        etypes = [('compound', 'CPD_DIS_ass', 'disease'), 
                  ('disease', 'rev_CPD_DIS_ass', 'compound')]      
    for etype in etypes: 
        try:
            out_pos = pred_score_pos[etype].reshape(-1,).detach().cpu().numpy() 
            out_neg = pred_score_neg[etype].reshape(-1,).detach().cpu().numpy() 
            pred_ = np.concatenate((out_pos, out_neg)) 
            y_ = [1]*len(out_pos) + [0]*len(out_neg) 
        
            auroc_rel[etype] = roc_auc_score(y_, pred_) 
            auprc_rel[etype] = average_precision_score(y_, pred_) 
        except:
            pass
    
    micro_auroc = roc_auc_score(labels, scores)
    micro_auprc = average_precision_score(labels, scores)
    macro_auroc = np.mean(list(auroc_rel.values()))
    macro_auprc = np.mean(list(auprc_rel.values()))
    return auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc

def get_all_metrics_test(pred_score_pos, pred_score_neg, scores, labels, G, full_mode = False):
    auroc_rel = {}
    auprc_rel = {}
    acc_rel = {}
    se_rel = {}
    spe_rel = {}
    pre_rel = {}
    f1_rel = {}
    
    if full_mode: 
        etypes = G.canonical_etypes
    else:
        etypes = [ ('compound', 'CPD_DIS_ass', 'disease'), 
                  ('disease', 'rev_CPD_DIS_ass', 'compound')]
       
    for etype in etypes:   
        try:
            out_pos = pred_score_pos[etype].reshape(-1,).detach().cpu().numpy() 
            out_neg = pred_score_neg[etype].reshape(-1,).detach().cpu().numpy() 
            pred_ = np.concatenate((out_pos, out_neg)) 
            y_ = [1]*len(out_pos) + [0]*len(out_neg) 
        
            auroc_rel[etype] = roc_auc_score(y_, pred_) 
            auprc_rel[etype] = average_precision_score(y_, pred_) 

            pred_sigmoid = torch.sigmoid(torch.cat((pred_score_pos[etype], pred_score_neg[etype])).reshape(-1,)).detach().cpu().numpy()           
            y_pred_label = [1 if i else 0 for i in (pred_sigmoid >= 0.5)] 
            y_true_label = y_ 
            TN, FP, FN, TP = confusion_matrix(y_true_label, y_pred_label).ravel()
            acc_rel[etype] = accuracy_score(y_true_label, y_pred_label)
            se_rel[etype] = recall_score(y_true_label, y_pred_label)
            spe_rel[etype] = TN / float(TN + FP)
            pre_rel[etype] = precision_score(y_true_label, y_pred_label)
            f1_rel[etype] = f1_score(y_true_label, y_pred_label)
        except:
            pass
    
    micro_auroc = roc_auc_score(labels, scores)
    micro_auprc = average_precision_score(labels, scores)
    macro_auroc = np.mean(list(auroc_rel.values()))
    macro_auprc = np.mean(list(auprc_rel.values()))
    return auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc,acc_rel,se_rel,spe_rel,pre_rel,f1_rel

def evaluate(model, valid_data, G):
    model.eval()    
    logits_valid, rels = model(G, valid_data) 
    scores = torch.sigmoid(logits_valid) 
    return get_all_metrics(valid_data.label.values, scores.cpu().detach().numpy(), rels)

def evaluate_fb(model, g_pos, g_neg, G, dd_etypes, device, return_embed = False, mode = 'valid'):
    model.eval()
    with torch.no_grad():
        pred_score_pos, pred_score_neg, pos_score, neg_score = model(G, g_neg, g_pos, pretrain_mode = False, mode = mode)
    pos_score = torch.cat([pred_score_pos[i] for i in dd_etypes]) 
    neg_score = torch.cat([pred_score_neg[i] for i in dd_etypes]) 

    logits = torch.cat((pos_score, neg_score)).reshape(-1)
    labels_t = torch.cat([torch.ones_like(pos_score),torch.zeros_like(neg_score)]).float().to(device)
    scores = torch.sigmoid(logits)
    loss = F.binary_cross_entropy_with_logits(logits, labels_t)
    labels = labels_t.reshape(-1,).detach().cpu().numpy()       
    if return_embed:
        return get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, G, True), loss.item(), pred_score_pos, pred_score_neg
    else:
        return get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, G, True), loss.item()

def evaluate_test(model, g_pos, g_neg, G, dd_etypes, device, return_embed = False, mode = 'valid'):
    model.eval()
    with torch.no_grad():
        pred_score_pos, pred_score_neg, pos_score, neg_score = model(G, g_neg, g_pos, pretrain_mode = False, mode = mode)
    pos_score = torch.cat([pred_score_pos[i] for i in dd_etypes]) 
    neg_score = torch.cat([pred_score_neg[i] for i in dd_etypes]) 

    logits = torch.cat((pos_score, neg_score)).reshape(-1)
    labels_t = torch.cat([torch.ones_like(pos_score),torch.zeros_like(neg_score)]).float().to(device)
    scores = torch.sigmoid(logits)
    loss = F.binary_cross_entropy_with_logits(logits, labels_t)
    labels = labels_t.reshape(-1,).detach().cpu().numpy()           
    if return_embed:
        return get_all_metrics_test(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, G, True), loss.item(), pred_score_pos, pred_score_neg
    else:
        return get_all_metrics_test(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, G, True), loss.item()

def evaluate_evi(model, g_pos, g_neg, G, eval_dd_etypes, device, mode="valid", return_raw=False):
    model.eval()
    G = G.to(device)
    g_pos = g_pos.to(device)
    g_neg = g_neg.to(device)
    with torch.no_grad():
        pred_score_pos, pred_score_neg, pos_score, neg_score = model(G, g_neg, g_pos, pretrain_mode = False, mode = mode)

    auroc_rel = {}
    auprc_rel = {}
    acc_rel = {}
    se_rel = {}
    spe_rel = {}
    pre_rel = {}
    f1_rel = {}
    result_rows = []
    losses = []
    for etype in eval_dd_etypes:
        try:
            alpha_pos = pred_score_pos[etype]  
            alpha_neg = pred_score_neg[etype]  
            alphas = torch.cat([alpha_pos, alpha_neg], dim=0)
            labels = torch.cat([torch.ones(alpha_pos.shape[0], device=device), torch.zeros(alpha_neg.shape[0], device=device)]).float()
            loss = dirichlet_loss(labels, alphas=alphas,device=device, lam=0.05)
            losses.append(loss.item())

            prob_pos, pred, uncertainty, evidence, belief = predict_from_alpha(alphas)
            labels_np = labels.detach().cpu().numpy().astype(int)
            prob_np = prob_pos.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy().astype(int)
            uncertainty_np = uncertainty.detach().cpu().numpy()
            evidence_np = evidence.detach().cpu().numpy()
            belief_np = belief.detach().cpu().numpy()
        
            auroc_rel[etype] = roc_auc_score(labels_np, prob_np) 
            auprc_rel[etype] = average_precision_score(labels_np, prob_np) 
            TN, FP, FN, TP = confusion_matrix(labels_np, pred_np).ravel()
            acc_rel[etype] = accuracy_score(labels_np, pred_np)
            se_rel[etype] = recall_score(labels_np, pred_np)
            spe_rel[etype] = TN / float(TN + FP)
            pre_rel[etype] = precision_score(labels_np, pred_np)
            f1_rel[etype] = f1_score(labels_np, pred_np)

            temp = pd.DataFrame({
                "relation": [str(etype)] * len(labels_np),
                "label": labels_np,
                "pred": pred_np,
                "prob": prob_np,
                "uncertainty": uncertainty_np,
                "evidence_0": evidence_np[:, 0],
                "evidence_1": evidence_np[:, 1],
                "belief_0": belief_np[:, 0],
                "belief_1": belief_np[:, 1],
            })
            result_rows.append(temp)
        except:
            pass    
    pred_df = pd.concat(result_rows, ignore_index=True)

    micro_auroc = roc_auc_score(pred_df["label"].values, pred_df["prob"].values)
    micro_auprc = average_precision_score(pred_df["label"].values, pred_df["prob"].values)
    macro_auroc = np.mean(list(auroc_rel.values()))
    macro_auprc = np.mean(list(auprc_rel.values()))
    loss = float(np.mean(losses))
    if return_raw:
        return auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc,acc_rel,se_rel,spe_rel,pre_rel,f1_rel, loss,pred_df
    return auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc,acc_rel,se_rel,spe_rel,pre_rel,f1_rel,loss

def dirichlet_loss(y, alphas,device, lam=1):
    """
    Use Evidential Learning Dirichlet loss from Sensoy et al
    :y: labels to predict
    :alphas: predicted parameters for Dirichlet
    :lambda: coefficient to weight KL term

    :return: Loss
    """
    def KL(alpha):
        """
        Compute KL for Dirichlet defined by alpha to uniform dirichlet
        :alpha: parameters for Dirichlet

        :return: KL
        """
        beta = torch.ones_like(alpha)
        S_alpha = torch.sum(alpha, dim=-1, keepdim=True)
        S_beta = torch.sum(beta, dim=-1, keepdim=True)

        ln_alpha = torch.lgamma(S_alpha)-torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
        ln_beta = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(S_beta)

        # digamma terms
        dg_alpha = torch.digamma(alpha)
        dg_S_alpha = torch.digamma(S_alpha)

        # KL
        kl = ln_alpha + ln_beta + torch.sum((alpha - beta)*(dg_alpha - dg_S_alpha), dim=-1, keepdim=True)
        return kl

    # Hard code to 2 classes per task, since this assumption is already made
    # for the existing chemprop classification tasks
    num_classes = 2
    num_tasks = 1

    y = y.long()
    y_one_hot = F.one_hot(y, num_classes=num_classes).float().to(device)

    alphas = torch.reshape(alphas, (alphas.shape[0], num_classes))
    # SOS term
    S = torch.sum(alphas, dim=-1, keepdim=True)
    p = alphas / S
    A = torch.sum(torch.pow((y_one_hot - p), 2), dim=-1, keepdim=True)
    B = torch.sum((p*(1 - p)) / (S+1), dim=-1, keepdim=True)
    SOS = A + B

    # KL
    alpha_hat = y_one_hot + (1-y_one_hot)*alphas
    kl = lam * KL(alpha_hat)

    loss = SOS + kl
    return loss.mean()

    
def evaluate_gnnexplainer(model, G, g_valid_pos, g_valid_neg, only_relation, epoch, etypes_train, penalty_scaling, device, mode = 'validation', weight_bias_track = False, wandb = None):
    loss_fct = nn.MSELoss()
    model.eval()
    G = G.to(device)
    with torch.no_grad():
        original_predictions_pos, original_predictions_neg, _, _ = model.gnnexplainer_forward(G, g_valid_pos, g_valid_neg, graphmask_mode = False, only_relation = only_relation)
        pos_score = torch.cat([original_predictions_pos[i] for i in etypes_train])
        neg_score = torch.cat([original_predictions_neg[i] for i in etypes_train])
        original_predictions = torch.sigmoid(torch.cat((pos_score, neg_score))).to('cpu')

        updated_predictions_pos, updated_predictions_neg, penalty, num_masked = model.gnnexplainer_forward(G, g_valid_pos, g_valid_neg, graphmask_mode = True, only_relation = only_relation)
        pos_score = torch.cat([updated_predictions_pos[i] for i in etypes_train])
        neg_score = torch.cat([updated_predictions_neg[i] for i in etypes_train])
        updated_predictions = torch.sigmoid(torch.cat((pos_score, neg_score)))

        labels = [1] * len(pos_score) + [0] * len(neg_score)
        loss_pred = F.binary_cross_entropy(updated_predictions, torch.Tensor(labels).float().to(device)).item()

        original_predictions = original_predictions.to(device)
        loss_pred_ori = F.binary_cross_entropy(original_predictions, torch.Tensor(labels).float().to(device)).item()

        loss = loss_fct(original_predictions, updated_predictions)
        total_loss = loss + penalty * penalty_scaling        
        
        print("----- " + mode + " Result -----")
        print("Running epoch {0:n} of GNNExplainer training. Mean divergence={1:.4f}, mean penalty={2:.4f}, bce_update={3:.4f}, bce_original={4:.4f}, num_masked_l1={5:.4f}, num_masked_l2={6:.4f}".format(
            epoch,
            float(loss.item()),
            float((penalty * penalty_scaling).item()),
            loss_pred,
            loss_pred_ori,
            num_masked[0]/G.number_of_edges(),
            num_masked[1]/G.number_of_edges())
        )
        return float(loss.item()) + float((penalty * penalty_scaling).item())
    
    
def evaluate_graphmask(model, G, g_valid_pos, g_valid_neg, only_relation, epoch, etypes_train, allowance, penalty_scaling, device, mode = 'validation', weight_bias_track = False, wandb = None, no_base = False):
    model.eval()
    G = G.to(device)
    with torch.no_grad():
        loss_fct = nn.MSELoss()

        g_valid_pos = g_valid_pos.to(device)
        g_valid_neg = g_valid_neg.to(device)

        original_predictions_pos, original_predictions_neg, _, _ = model.graphmask_forward(G, g_valid_pos, g_valid_neg, graphmask_mode = False, only_relation = only_relation, no_base = no_base)

        pos_score = torch.cat([original_predictions_pos[i] for i in etypes_train])
        neg_score = torch.cat([original_predictions_neg[i] for i in etypes_train])
        original_predictions = torch.sigmoid(torch.cat((pos_score, neg_score)))

        original_predictions = original_predictions.to('cpu')

        updated_predictions_pos, updated_predictions_neg, penalty, num_masked = model.graphmask_forward(G, g_valid_pos, g_valid_neg, graphmask_mode = True, only_relation = only_relation, no_base = no_base)
        pos_score = torch.cat([updated_predictions_pos[i] for i in etypes_train])
        neg_score = torch.cat([updated_predictions_neg[i] for i in etypes_train])
        updated_predictions = torch.sigmoid(torch.cat((pos_score, neg_score)))

        labels = [1] * len(pos_score) + [0] * len(neg_score)
        loss_pred = F.binary_cross_entropy(updated_predictions, torch.Tensor(labels).float().to(device)).item()

        # loss is the divergence with original predictions
        G = G.to('cpu')
        original_predictions = original_predictions.to(device)
        loss_pred_ori = F.binary_cross_entropy(original_predictions, torch.Tensor(labels).float().to(device)).item()

        loss = loss_fct(original_predictions, updated_predictions)

        g = torch.relu(loss - allowance).mean() 
        f = penalty * penalty_scaling 

        g_valid_pos = g_valid_pos.to('cpu')
        g_valid_neg = g_valid_neg.to('cpu')
        
        print("----- " + mode + " Result -----")
        print("Epoch {0:n}, Mean divergence={1:.4f}, mean penalty={2:.4f}, bce_update={3:.4f}, bce_original={4:.4f}, num_masked_l1={5:.4f}, num_masked_l2={6:.4f}".format(
            epoch,
            float(loss.mean().item()),
            float(f),
            loss_pred,
            loss_pred_ori,
            num_masked[0]/G.number_of_edges(),
            num_masked[1]/G.number_of_edges())
        )
        print("-------------------------------")
        
        if mode == 'testing':
            test_metrics = {}
            pred_update = updated_predictions.detach().cpu().numpy()
            pred_ori = original_predictions.detach().cpu().numpy()
            y_ = np.array(labels)
            
            test_metrics['test auroc original'] = roc_auc_score(y_, pred_ori)
            test_metrics['test auprc original'] = average_precision_score(y_, pred_ori)
            test_metrics['test auroc update'] = roc_auc_score(y_, pred_update)
            test_metrics['test auprc update'] = average_precision_score(y_, pred_update)
            test_metrics['test %masked_L1'] = num_masked[0]/G.number_of_edges()
            test_metrics['test %masked_L2'] = num_masked[1]/G.number_of_edges()
            
        if weight_bias_track:
            wandb.log({mode + ' divergence': float(loss.mean().item()),
                       mode + ' penalty': float(f),
                       mode + ' bce_masked': loss_pred,
                       mode + ' bce_original': loss_pred_ori,
                       mode + ' %masked_L1': num_masked[0]/G.number_of_edges(),
                       mode + ' %masked_L2': num_masked[1]/G.number_of_edges()})

        g_, f_ = float(loss.mean().item()), float(f)
        del original_predictions, updated_predictions, f, g, loss, pos_score, neg_score, original_predictions_pos,original_predictions_neg,updated_predictions_pos, updated_predictions_neg
    if mode == 'testing':
        return g_ + f_ , test_metrics
    else:
        return g_ + f_
    
    
def evaluate_ib(model, G, g_valid_pos, g_valid_neg, only_relation, epoch, etypes_train, allowance, penalty_scaling, device, mode = 'validation', weight_bias_track = False, wandb = None, no_base = False):
    model.eval()
    G = G.to(device)
    with torch.no_grad():
        loss_fct = nn.MSELoss()

        g_valid_pos = g_valid_pos.to(device)
        g_valid_neg = g_valid_neg.to(device)

        original_predictions_pos, original_predictions_neg, _, _ = model.ib_forward(G, g_valid_pos, g_valid_neg, graphmask_mode = False, only_relation = only_relation, no_base = no_base)

        pos_score = torch.cat([original_predictions_pos[i] for i in etypes_train])
        neg_score = torch.cat([original_predictions_neg[i] for i in etypes_train])
        original_predictions = torch.sigmoid(torch.cat((pos_score, neg_score)))

        original_predictions = original_predictions.to('cpu')

        updated_predictions_pos, updated_predictions_neg, penalty, num_masked = model.ib_forward(G, g_valid_pos, g_valid_neg, graphmask_mode = True, only_relation = only_relation, no_base = no_base)
        pos_score = torch.cat([updated_predictions_pos[i] for i in etypes_train])
        neg_score = torch.cat([updated_predictions_neg[i] for i in etypes_train])
        updated_predictions = torch.sigmoid(torch.cat((pos_score, neg_score)))

        labels = [1] * len(pos_score) + [0] * len(neg_score)
        loss_pred = F.binary_cross_entropy(updated_predictions, torch.Tensor(labels).float().to(device)).item()

        # loss is the divergence with original predictions
        G = G.to('cpu')
        original_predictions = original_predictions.to(device)
        loss_pred_ori = F.binary_cross_entropy(original_predictions, torch.Tensor(labels).float().to(device)).item()

        loss = loss_fct(original_predictions, updated_predictions)

        g = torch.relu(loss - allowance).mean()
        f = penalty * penalty_scaling

        g_valid_pos = g_valid_pos.to('cpu')
        g_valid_neg = g_valid_neg.to('cpu')
        
        print("----- " + mode + " Result -----")
        print("Epoch {0:n}, Mean divergence={1:.4f}, mean penalty={2:.4f}, bce_update={3:.4f}, bce_original={4:.4f}, num_masked_l1={5:.4f}, num_masked_l2={6:.4f}".format(
            epoch,
            float(loss.mean().item()),
            float(f),
            loss_pred,
            loss_pred_ori,
            num_masked[0]/G.number_of_edges(),
            num_masked[1]/G.number_of_edges())
        )
        print("-------------------------------")
        
        if mode == 'testing':
            test_metrics = {}
            pred_update = updated_predictions.detach().cpu().numpy()
            pred_ori = original_predictions.detach().cpu().numpy()
            y_ = np.array(labels)
            
            test_metrics['test auroc original'] = roc_auc_score(y_, pred_ori)
            test_metrics['test auprc original'] = average_precision_score(y_, pred_ori)
            test_metrics['test auroc update'] = roc_auc_score(y_, pred_update)
            test_metrics['test auprc update'] = average_precision_score(y_, pred_update)
            test_metrics['test %masked_L1'] = num_masked[0]/G.number_of_edges()
            test_metrics['test %masked_L2'] = num_masked[1]/G.number_of_edges()
            
        if weight_bias_track:
            wandb.log({mode + ' divergence': float(loss.mean().item()),
                       mode + ' penalty': float(f),
                       mode + ' bce_masked': loss_pred,
                       mode + ' bce_original': loss_pred_ori,
                       mode + ' %masked_L1': num_masked[0]/G.number_of_edges(),
                       mode + ' %masked_L2': num_masked[1]/G.number_of_edges()})

        g_, f_ = float(loss.mean().item()), float(f)
        del original_predictions, updated_predictions, f, g, loss, pos_score, neg_score
    if mode == 'testing':
        return g_ + f_ , test_metrics
    else:
        return g_ + f_
    
def evaluate_mb(model, g_pos, g_neg, G, dd_etypes, device, return_embed = False, mode = 'valid'):
    model.eval()
    #model = model.to('cpu')
    pred_score_pos, pred_score_neg, pos_score, neg_score = model.forward_minibatch(g_pos.to(device), g_neg.to(device), [G.to(device), G.to(device)], G.to(device), mode = mode, pretrain_mode = False)
    
    pos_score = torch.cat([pred_score_pos[i] for i in dd_etypes])
    neg_score = torch.cat([pred_score_neg[i] for i in dd_etypes])
    
    scores = torch.sigmoid(torch.cat((pos_score, neg_score)).reshape(-1,))
    labels = [1] * len(pos_score) + [0] * len(neg_score)
    loss = F.binary_cross_entropy(scores, torch.Tensor(labels).float().to(device))
    
    model = model.to(device)
    if return_embed:
        return get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, G, True), loss.item(), pred_score_pos, pred_score_neg
    else:
        return get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, G, True), loss.item()

# disable all gradient
def disable_all_gradients(module):
    for param in module.parameters():
        param.requires_grad = False

def print_dict(x, dd_only = False): 
    if dd_only:
        etypes = [('compound', 'CPD_DIS_ass', 'disease'), 
                  ('disease', 'rev_CPD_DIS_ass', 'compound')]
        
        for i in etypes:
            print(str(i) + ': ' + str(x[i])) 
    else:
        for i, j in x.items(): 
            print(str(i) + ': ' + str(j))  
        
def to_wandb_table(auroc, auprc):
    return [[idx, i[1], j, auprc[i]] for idx, (i, j) in enumerate(auroc.items())]

def get_n_params(model):
    pp=0
    for p in list(model.parameters()): 
        nn=1
        for s in list(p.size()):
            nn = nn*s 
        pp += nn
    return pp

def process_df(df_train, edge_dict):
    df_train['relation_idx'] = [edge_dict[i] for i in df_train['relation']]
    df_train = df_train[['x_type', 'x_idx', 'relation_idx', 'y_type', 'y_idx', 'degree', 'label']].rename(columns = {'x_type': 'head_type', 
                                                                                    'x_idx': 'head', 
                                                                                    'relation_idx': 'relation',
                                                                                    'y_type': 'tail_type',
                                                                                    'y_idx': 'tail'})
    df_train['head'] = df_train['head'].astype(int)
    df_train['tail'] = df_train['tail'].astype(int)
    return df_train


def reverse_rel_generation(df, df_valid, unique_rel): 
    for i in unique_rel.values: 
        temp = df_valid[df_valid.relation == i[1]] 
        temp = temp.rename(columns={"x_type": "y_type", 
                                    "x_id": "y_id", 
                                    "x_idx": "y_idx",
                                    "y_type": "x_type", 
                                    "y_id": "x_id", 
                                    "y_idx": "x_idx"})
        if i[0] != i[2]: 
            # bi identity
            temp["relation"] = 'rev_' + i[1] 
        df_valid = pd.concat([df_valid,temp])
    return df_valid.reset_index(drop = True) 


def get_wandb_log_dict(auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc, mode):
    
    results = {
              mode + " Micro AUROC": micro_auroc,
              mode + " Micro AUPRC": micro_auprc,
              mode + " Macro AUROC": macro_auroc,
              mode + " Macro AUPRC": macro_auprc
    } 
    
    relations = [('compound', 'CPD_DIS_ass', 'disease'), 
                ('disease', 'rev_CPD_DIS_ass', 'compound')]
    
    name_mapping = {
                    ('compound', 'CPD_DIS_ass', 'disease'): ' CPD_DIS_ass ',
                    ('disease', 'rev_CPD_DIS_ass', 'compound'): ' rev_CPD_DIS_ass ',
                   }
    
    for i in relations: 
        if i in auroc_rel:
            results.update({mode + name_mapping[i] + "AUROC": auroc_rel[i]}) 
        if i in auprc_rel:
            results.update({mode + name_mapping[i] + "AUPRC": auprc_rel[i]}) 
    return results

def sim_matrix(a, b, eps=1e-8): 
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None] 
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1)) 
    return sim_mt


def obtain_protein_random_walk_profile(disease, num_walks, path_len, g, disease_etypes, disease_nodes, walk_mode):
    random_walks = []
    num_nodes = len(g.nodes('gene/protein'))
    for _ in range(num_walks):
        successor = g.successors(disease, etype = 'rev_GEN_DIS_ass')
        if len(successor) > 0:
            current = choice(successor)
        else:
            continue
        path = [current.item()]
        for path_idx in range(path_len):
            successor = g.successors(current, etype = 'GEN_GEN_int')
            if len(successor) > 0:
                current = choice(successor)
                path.append(current.item())
            else:
                break

        random_walks = random_walks + path
        
    if walk_mode == 'bit':
        visted_nodes = np.unique(np.array(random_walks))
        node_profile = torch.zeros((num_nodes,))
        node_profile[visted_nodes] = 1.
    elif walk_mode == 'prob':
        visted_nodes = Counter(random_walks)
        node_profile = torch.zeros((num_nodes,))
        for x, y in visted_nodes.items():
            node_profile[x] = y/len(random_walks)
    return node_profile

def obtain_disease_profile(G, disease, disease_etypes, disease_nodes): 
    profiles_for_each_disease_types = []
    for idx, disease_etype in enumerate(disease_etypes): 
        nodes = G.successors(disease, etype=disease_etype) 
        num_nodes = len(G.nodes(disease_nodes[idx])) 
        node_profile = torch.zeros((num_nodes,)) 
        node_profile[nodes] = 1.  
        profiles_for_each_disease_types.append(node_profile) 
    return torch.cat(profiles_for_each_disease_types)

def exponential(x, lamb):
    return lamb * torch.exp(-lamb * x) + 0.2 

def convert2str(x):
    try:
        if '_' in str(x): 
            pass
        else:
            x = float(x) 
    except:
        pass
    return str(x)

def map_node_id_2_idx(x, id2idx):
        id_ = convert2str(x)
        if id_ in id2idx:
            return id2idx[id_]
        else:
            return 'null'
        
def process_disease_area_split(data_folder, df, df_test, split,split_data_path):
    disease_list = pd.read_csv(os.path.join(data_folder, 'disease_files', split + '.csv'))
    temp = df_test[df_test.relation.isin(['CPD_DIS_ass'])]
    df_test = df_test.drop(temp[~temp.y_idx.isin(disease_list.node_idx.unique())].index)
    temp = df_test[df_test.relation.isin(['rev_CPD_DIS_ass'])]
    df_test = df_test.drop(temp[~temp.x_idx.isin(disease_list.node_idx.unique())].index)
    df_test.to_csv(os.path.join(split_data_path, 'test.csv'), index = False)
    return df_test

def sparsify_train_edges(df_train,
                        keep_ratio=1.0,
                        seed=42,
                        mode="all",
                        target_rel=("CPD_DIS_ass", "rev_CPD_DIS_ass"),
                        min_edges_per_rel=1,
                        preserve_reverse_pair=True,
                        ):
    if keep_ratio <= 0 or keep_ratio >= 1:
        raise ValueError(f"keep_ratio must be in (0, 1), got {keep_ratio}")
    df_train = df_train.copy().reset_index(drop=True)

    required_cols = {"x_type", "x_idx", "relation", "y_type", "y_idx"}
    missing = required_cols - set(df_train.columns)
    if missing:
        raise ValueError(f"df_train missing columns: {missing}")

    if mode == "all":
        df_keep_fixed = pd.DataFrame(columns=df_train.columns)
        df_sample_pool = df_train
    elif mode == "context_only":
        target_rel = set(target_rel)
        df_keep_fixed = df_train[df_train["relation"].isin(target_rel)]
        df_sample_pool = df_train[~df_train["relation"].isin(target_rel)]
    else:
        raise ValueError("mode must be 'all' or 'context_only'")

    sampled_list = []
    if preserve_reverse_pair:
        pool_sample = df_sample_pool.copy()

        def canonical_rel(rel):
            rel = str(rel)
            return rel[4:] if rel.startswith("rev_") else rel
        pool_sample["base_rel"] = pool_sample["relation"].apply(canonical_rel)

        left = (
            pool_sample["x_type"].astype(str)
            + ":"
            + pool_sample["x_idx"].astype(int).astype(str)
        )
        right = (
            pool_sample["y_type"].astype(str)
            + ":"
            + pool_sample["y_idx"].astype(int).astype(str)
        )
        pair_min = left.where(left <= right, right)
        pair_max = right.where(left <= right, left)
        pool_sample["pair_key"] = (
            pool_sample["base_rel"].astype(str)
            + "|"
            + pair_min.astype(str)
            + "|"
            + pair_max.astype(str)
        )

        unique_pairs = pool_sample[["pair_key", "base_rel"]].drop_duplicates()

        keep_unique_pairs = []
        for rel, sub_pairs in unique_pairs.groupby("base_rel"):
            n = len(sub_pairs)
            k = max(min_edges_per_rel, int(round(n * keep_ratio)))
            k = min(k, n)
            keep_unique_pairs.append(sub_pairs.sample(n=k, replace=False, random_state=seed))
        keep_pairs = pd.concat(keep_unique_pairs, ignore_index=True)["pair_key"]
        sampled = pool_sample[pool_sample["pair_key"].isin(set(keep_pairs))]
        sampled = sampled.drop(columns=["base_rel", "pair_key"])
        sampled_list.append(sampled)
    else:
        for rel, sub_df in df_sample_pool.groupby("relation"):
            n = len(sub_df)
            k = max(min_edges_per_rel, int(round(n * keep_ratio)))
            k = min(k, n)
            sampled_list.append(sub_df.sample(n=k, replace=False, random_state=seed))

    df_sparse = pd.concat([df_keep_fixed] + sampled_list, ignore_index=True)
    df_sparse = df_sparse.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(
        f"[sparsify_train_edges] mode={mode}, keep_ratio={keep_ratio}, "
        f"before={len(df_train)}, after={len(df_sparse)}, "
        f"actual_ratio={len(df_sparse) / max(len(df_train), 1):.4f}"
    )
    return df_sparse

def graph_to_edge_df(g):    
    rows = []
    for etype in g.canonical_etypes:
        src_type, rel, dst_type = etype
        src, dst = g.edges(etype=etype)
        src = src.detach().cpu().numpy()
        dst = dst.detach().cpu().numpy()
        if len(src) == 0:
            continue
        temp = pd.DataFrame({"x_type": src_type, "x_idx": src.astype(int),
                             "relation": rel,"y_type": dst_type, "y_idx": dst.astype(int)})
        rows.append(temp)
    if len(rows) == 0:
        raise ValueError("No edges found in graph.")
    return pd.concat(rows, ignore_index=True)

def edge_df_to_graph(df_edges, g_ref):
    edges_dict = {}
    for (src_type, rel, dst_type), sub_df in df_edges.groupby(["x_type", "relation", "y_type"]):
        src = sub_df["x_idx"].astype(int).to_numpy()
        dst = sub_df["y_idx"].astype(int).to_numpy()
        edges_dict[(src_type, rel, dst_type)] = (src, dst)
    num_nodes_dict = {ntype: g_ref.number_of_nodes(ntype) for ntype in g_ref.ntypes}
    g_new = dgl.heterograph(edges_dict, num_nodes_dict=num_nodes_dict)

    for ntype in g_ref.ntypes:
        for key, val in g_ref.nodes[ntype].data.items():
            g_new.nodes[ntype].data[key] = val.detach().cpu().clone()
    for etype in g_new.canonical_etypes:
        if etype in g_ref.canonical_etypes and "id" in g_ref.edges[etype].data:
            if g_ref.number_of_edges(etype) > 0:
                rel_id = g_ref.edges[etype].data["id"][0].detach().cpu().item()
                g_new.edges[etype].data["id"] = (torch.ones(g_new.number_of_edges(etype), dtype=torch.long) * rel_id)
    return g_new

def sample_neg_edges_match_pos(df_neg_full,df_pos_sparse,seed=42):
    rng_seed = seed
    sampled_list = []
    group_cols = ["x_type", "relation", "y_type"]
    pos_counts = (df_pos_sparse.groupby(group_cols).size().reset_index(name="n_pos"))

    for _, row in pos_counts.iterrows():
        src_type = row["x_type"]
        rel = row["relation"]
        dst_type = row["y_type"]
        n_pos = int(row["n_pos"])

        sub_neg = df_neg_full[(df_neg_full["x_type"] == src_type)
                                & (df_neg_full["relation"] == rel)
                                & (df_neg_full["y_type"] == dst_type)]
        if len(sub_neg) == 0:
            print(f"[Warning] No negative edges found for "
                f"({src_type}, {rel}, {dst_type}). Skip this etype.")
            continue
        n_sample = min(n_pos, len(sub_neg))
        sampled = sub_neg.sample(n=n_sample,replace=False,random_state=rng_seed)
        sampled_list.append(sampled)
    if len(sampled_list) == 0:
        raise ValueError("No negative edges sampled. Please check g_train_neg.pkl.")

    df_neg_sparse = pd.concat(sampled_list, ignore_index=True)
    df_neg_sparse = df_neg_sparse.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df_neg_sparse

def make_sparse_train_from_saved_pkl(base_pkl_dir,
                                    out_pkl_dir,
                                    keep_ratio,
                                    seed,
                                    mode,
                                    target_rel,
                                    ):
    train_pos = os.path.join(base_pkl_dir, "g_train_pos.pkl")
    train_neg = os.path.join(base_pkl_dir, "g_train_neg.pkl")
    if not os.path.exists(train_pos):
        raise FileNotFoundError(f"Missing {train_pos}")
    if not os.path.exists(train_neg):
        raise FileNotFoundError(f"Missing {train_neg}")

    with open(train_pos, "rb") as f:
        g_train_pos_full = pickle.load(f).to("cpu")
    with open(train_neg, "rb") as f:
        g_train_neg_full = pickle.load(f).to("cpu")
    # graph -> edge df
    df_train_pos_full = graph_to_edge_df(g_train_pos_full)
    df_train_neg_full = graph_to_edge_df(g_train_neg_full)
    print('train:',df_train_pos_full.shape,df_train_neg_full.shape)

    # sparse positive edges
    if keep_ratio < 1.0:
        df_train_pos_sparse = sparsify_train_edges(df_train=df_train_pos_full,
                                                    keep_ratio=keep_ratio,
                                                    seed=seed,
                                                    mode=mode,
                                                    target_rel=target_rel,
                                                    min_edges_per_rel=1,
                                                    preserve_reverse_pair=True,
                                                    )
    else:
        df_train_pos_sparse = df_train_pos_full.reset_index(drop=True)

    # sparse negative edges from full g_train_neg.pkl
    if keep_ratio < 1.0:
        df_train_neg_sparse = sample_neg_edges_match_pos(
                                                        df_neg_full=df_train_neg_full,
                                                        df_pos_sparse=df_train_pos_sparse,
                                                        seed=seed)
    else:
        df_train_neg_sparse = df_train_neg_full.reset_index(drop=True)

    # edge df -> graph
    g_train_pos_sparse = edge_df_to_graph(df_train_pos_sparse, g_train_pos_full)
    g_train_neg_sparse = edge_df_to_graph(df_train_neg_sparse, g_train_neg_full)
    os.makedirs(out_pkl_dir,exist_ok=True)

    for name in ["g_valid_pos.pkl", "g_valid_neg.pkl","g_test_pos.pkl","g_test_neg.pkl"]:
        src = os.path.join(base_pkl_dir, name)
        dst = os.path.join(out_pkl_dir, name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing fixed eval pkl: {src}")
        shutil.copyfile(src, dst)
    return g_train_pos_sparse, g_train_neg_sparse

def create_graph(df_train):
    unique_graph = df_train[['x_type', 'relation', 'y_type']].drop_duplicates() 
    DGL_input = {}
    for i in unique_graph.values: 
        o = df_train[df_train.relation == i[1]][['x_idx', 'y_idx']].values.T 
        DGL_input[tuple(i)] = (o[0].astype(int), o[1].astype(int)) 

    temp = dict(df_train.groupby('x_type')['x_idx'].max()) 
    temp2 = dict(df_train.groupby('y_type')['y_idx'].max()) 
    output = {} 
    for d in (temp, temp2):
        for k, v in d.items(): 
            output.setdefault(k, float('-inf')) 
            output[k] = max(output[k], v) 

    g = dgl.heterograph(DGL_input, num_nodes_dict={i: int(output[i])+1 for i in output.keys()})
    # get node, edge dictionary mapping relation sent to index
    node_dict = {}
    edge_dict = {}
    for ntype in g.ntypes: 
        node_dict[ntype] = len(node_dict) 
    for etype in g.etypes:
        edge_dict[etype] = len(edge_dict) 
        g.edges[etype].data['id'] = torch.ones(g.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] 
    return g

def create_dgl_graph(df_train, df):
    unique_graph = df_train[['x_type', 'relation', 'y_type']].drop_duplicates() 
    DGL_input = {}
    for i in unique_graph.values: 
        o = df_train[df_train.relation == i[1]][['x_idx', 'y_idx']].values.T 
        DGL_input[tuple(i)] = (o[0].astype(int), o[1].astype(int)) 

    temp = dict(df.groupby('x_type')['x_idx'].max()) 
    temp2 = dict(df.groupby('y_type')['y_idx'].max()) 
    output = {} 
    for d in (temp, temp2):
        for k, v in d.items(): 
            output.setdefault(k, float('-inf')) 
            output[k] = max(output[k], v)

    g = dgl.heterograph(DGL_input, num_nodes_dict={i: int(output[i])+1 for i in output.keys()})
    # get node, edge dictionary mapping relation sent to index
    node_dict = {}
    edge_dict = {}
    for ntype in g.ntypes: 
        node_dict[ntype] = len(node_dict) 
    for etype in g.etypes: 
        edge_dict[etype] = len(edge_dict) 
        g.edges[etype].data['id'] = torch.ones(g.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] 
    return g

def initialize_node_embedding(g,n_inp,
                                cpd_path=None,
                                dis_path=None,
                                gen_path=None,
                                pwy_path=None,
                                bp_path=None,
                                mf_path=None,
                                cc_path=None,
                                ):
    init_config = {
                    "compound": (cpd_path, 768),
                    "disease": (dis_path, 768),
                    "gene/protein": (gen_path, 1280),
                    "pathway": (pwy_path, 768),
                    "biological_process": (bp_path, 768),
                    "molecular_function": (mf_path, 768),
                    "cellular_component": (cc_path, 768),
                }

    def _load_emb(path, num_nodes, raw_dim):
        raw_emb = torch.empty(num_nodes, raw_dim, dtype=torch.float32)
        nn.init.xavier_uniform_(raw_emb)
        with open(path, "rb") as f:
            emb_dict = pickle.load(f)
        for idx, vec in emb_dict.items():
            raw_emb[idx] = torch.from_numpy(vec).float().view(-1)
        return raw_emb

    for ntype in g.ntypes:
        num_nodes = g.number_of_nodes(ntype)
        if ntype in init_config:
            emb_path, raw_dim = init_config[ntype]
            if os.path.exists(emb_path):
                raw_emb = _load_emb(path=emb_path,num_nodes=num_nodes,raw_dim=raw_dim)
                g.nodes[ntype].data["inp"] = raw_emb
            else:
                raise FileNotFoundError(f"{ntype}: pretrained embedding path not found: {emb_path}")
        else:
            raise FileNotFoundError(f"{ntype}: not in init_config")
    return g

def disease_centric_evaluation(df, df_train, df_valid, df_test, data_path, G, model, device, disease_ids = None, relation = None, weight_bias_track = False, wandb = None, show_plot = False, verbose = False, return_raw = False, simulate_random = True, only_prediction = False):
    G = G.to(device) 
    model = model.eval()
    dd_etypes = [
                ('compound', 'CPD_DIS_ass', 'disease'), 
                ]
    dd_rel_types = ['CPD_DIS_ass']

    disease_etypes = [('disease', 'rev_CPD_DIS_ass', 'compound')]

    disease_rel_types = ['rev_CPD_DIS_ass']

    df['x_id'] = df.x_id.apply(lambda x: convert2str(x)) 
    df['y_id'] = df.y_id.apply(lambda x: convert2str(x))

    idx2id_compound = dict(df[df.x_type == 'compound'][['x_idx', 'x_id']].drop_duplicates().values) 
    idx2id_compound.update(dict(df[df.y_type == 'compound'][['y_idx', 'y_id']].drop_duplicates().values))

    idx2id_disease = dict(df[df.x_type == 'disease'][['x_idx', 'x_id']].drop_duplicates().values)
    idx2id_disease.update(dict(df[df.y_type == 'disease'][['y_idx', 'y_id']].drop_duplicates().values))

    df_ = pd.read_csv(os.path.join(data_path, 'kg.tsv'),sep='\t') 
    df_['x_id'] = df_.x_id.apply(lambda x: convert2str(x))
    df_['y_id'] = df_.y_id.apply(lambda x: convert2str(x))

    id2name_disease = dict(df_[df_.x_type == 'disease'][['x_id', 'x_name']].drop_duplicates().values) 
    id2name_disease.update(dict(df_[df_.y_type == 'disease'][['y_id', 'y_name']].drop_duplicates().values)) 

    id2name_compound = dict(df_[df_.x_type == 'compound'][['x_id', 'x_name']].drop_duplicates().values) 
    id2name_compound.update(dict(df_[df_.y_type == 'compound'][['y_id', 'y_name']].drop_duplicates().values)) 

    compound_ids_rels = {}
    disease_ids_rels = {}

    for i in ['CPD_DIS_ass']:
        compound_ids_rels['rev_' + i] = df[df.relation == i].x_id.unique() 
        disease_ids_rels[i] = df[df.relation == i].y_id.unique()

    num_of_compounds_rels = {}
    num_of_diseases_rels = {}
    for i in ['CPD_DIS_ass']:
        num_of_compounds_rels['rev_' + i] = len(compound_ids_rels['rev_' + i]) 
        num_of_diseases_rels[i] = len(disease_ids_rels[i]) 


    def mean_reciprocal_rank(rs): 
        rs = (np.asarray(r).nonzero()[0] for r in rs)
        return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs]) 

    def precision_at_k(r, k): 
        assert k >= 1
        r = np.asarray(r)[:k] != 0 
        if r.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(r)

    def average_precision(r): 
        r = np.asarray(r) != 0 
        out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]] 
        if not out:
            return 0.
        return np.mean(out)

    def calculate_metrics(rel, preds_all, labels_all, mode = 'compound', subset_mode = True):
        if mode == 'compound':
            etype = dd_rel_types 
            ids_rels = disease_ids_rels 
            if subset_mode:
                k10 = int(num_of_diseases_rels[rel] * 0.1)
                k5 = int(num_of_diseases_rels[rel] * 0.05)
                k1 = int(num_of_diseases_rels[rel] * 0.01)
                num_items = num_of_diseases_rels

            else:
                k10 = 2229
                k5 = 1114
                k1 = 222
                num_items = {'CPD_DIS_ass': 22293}
        else:
            ids_rels = compound_ids_rels 
            if subset_mode: 
                k10 = int(num_of_compounds_rels[rel] * 0.1) 
                k5 = int(num_of_compounds_rels[rel] * 0.05) 
                k1 = int(num_of_compounds_rels[rel] * 0.01) 
                num_items = num_of_compounds_rels 
            else:
                k10 = 792
                k5 = 396
                k1 = 79
                num_items = {'rev_CPD_DIS_ass': 7926}

        if mode == 'compound':
            id2name = id2name_disease
            id2name_rev = id2name_compound
        if mode == 'disease':
            id2name = id2name_compound 
            id2name_rev = id2name_disease 

        ids_all = list(preds_all[rel].keys()) 

        name, auroc, auprc =  {}, {}, {}
        acc, sens, spec, f1, ppv, npv, fpr, fnr, fdr, pos_len, ids, ranked_list = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

        AP, MRR, Recall, Recall_Random, Enrichment, not_in_ranked_list, in_ranked_list = {}, {}, {}, {}, {}, {}, {}

        disease_not_intersecting_list = []

        k_num = {'1%': k1, '5%': k5, '10%': k10, '10': 10, '50': 50, '100': 100} 

        for i, j in k_num.items(): 
            AP[i], MRR[i], Recall[i], Recall_Random[i], Enrichment[i], not_in_ranked_list[i], in_ranked_list[i] = {}, {}, {}, {}, {}, {}, {}
        for entity_id in ids_all: 
            pred = preds_all[rel][entity_id] 
            lab = labels_all[rel][entity_id] 
            # remove training set compounds/diseases, which are labelled -1
            # retrieving only the compounds/diseases that belong to the rel types
            fixed_keys = np.intersect1d(ids_rels[rel], [i for i,j in lab.items() if j != -1]) 
            pred_array = np.array([pred[i] for i in fixed_keys]) 
            lab_array = np.array([lab[i] for i in fixed_keys]) 

            idx2id = {idx: i for idx, i in enumerate(fixed_keys)} 

            pos_idx = np.where(np.array(lab_array) == 1)[0] 
            pos_len[entity_id] = len(pos_idx) 
            
            if len(pos_idx) == 0:
                auroc[entity_id] = -1
                auprc[entity_id] = -1
            else:
                try:
                    auroc[entity_id] = roc_auc_score(lab_array, pred_array) 
                except:
                    auroc[entity_id] = -1
                try:
                    auprc[entity_id] = average_precision_score(lab_array, pred_array) 
                except:    
                    auprc[entity_id] = -1
            
            ranked_list_entity = np.argsort(pred_array)[::-1] 
            ranked_list[entity_id] = [id2name[idx2id[i]] for i in ranked_list_entity] 
            
            if simulate_random: 
                ranked_list_random = []
                for i in range(500):
                    non_guided_compound_list = list(range(len(ranked_list_entity))) 
                    np.random.shuffle(non_guided_compound_list) 
                    ranked_list_random.append(non_guided_compound_list)  
            
            ranked_list_k = {i: ranked_list_entity[:j] for i,j in k_num.items()}       

            for i, j in ranked_list_k.items(): 
                recalled_list = np.intersect1d(ranked_list_k[i], pos_idx) 
                in_ranked_list[i][entity_id] = [id2name[idx2id[x]] for x in recalled_list] 
                not_in_ranked_list[i][entity_id] = [id2name[idx2id[x]] for x in pos_idx if x not in recalled_list] 
                if len(pos_idx) == 0:
                    Recall[i][entity_id] = -1
                    Recall_Random[i][entity_id] = -1
                    Enrichment[i][entity_id] = -1
                    AP[i][entity_id] = -1
                    MRR[i][entity_id] = -1
                else:
                    Recall[i][entity_id] = len(recalled_list)/len(pos_idx) 
                
                    if simulate_random: 
                        Recall_Random[i][entity_id] = np.mean([len(np.intersect1d(sim_trial[:k_num[i]], pos_idx))/len(pos_idx) for sim_trial in ranked_list_random]) 
                    else:
                        Recall_Random[i][entity_id] = k_num[i]/num_items[rel]
                        
                    Enrichment[i][entity_id] = len(recalled_list) / (Recall_Random[i][entity_id] * len(pos_idx)) 

                    rs = [1 if x in pos_idx else 0 for x in ranked_list_k[i]] 
                    AP[i][entity_id] = average_precision(rs) 
                    MRR[i][entity_id] = mean_reciprocal_rank([rs]) 

            y_pred_s = [1 if i else 0 for i in (pred_array >= 0.5)] 
            y = lab_array 
            cm1 = confusion_matrix(y, y_pred_s) 
            if len(cm1) == 1:
                cm1 = np.array([[cm1[0,0], 0], [0, 0]])
            total1=sum(sum(cm1)) 
            accuracy1=(cm1[0,0]+cm1[1,1])/total1 
            acc[entity_id] = accuracy1 

            sensitivity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
            sens[entity_id] = sensitivity1 

            specificity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
            spec[entity_id] = specificity1 

            f1[entity_id] = f1_score(y, y_pred_s) 

            TN = cm1[0][0]
            FN = cm1[1][0]
            TP = cm1[1][1]
            FP = cm1[0][1]

            # Precision or positive predictive value
            ppv[entity_id] = TP/(TP+FP)
            # Negative predictive value
            npv[entity_id] = TN/(TN+FN)
            # Fall out or false positive rate
            fpr[entity_id] = FP/(FP+TN)
            # False negative rate
            fnr[entity_id] = FN/(TP+FN)
            # False discovery rate
            fdr[entity_id] = FP/(TP+FP)
            name[entity_id] = id2name_rev[entity_id] 
            ids[entity_id] = entity_id 

        out_dict = {'ID': ids, 
                'Name': name, 
                'Ranked List': ranked_list,           
                'AUROC': auroc,  
                'AUPRC': auprc,  
                'Accuracy': acc, 
                'Sensitivity': sens, 
                'Specificity': spec,
                'F1': f1,
                'PPV': ppv, 
                'NPV': npv,
                'FPR': fpr,
                'FNR': fnr,
                'FDR': fdr,
                '# of Pos': pos_len,  
                'Prediction': preds_all[rel], 
                'Labels': labels_all[rel] 
               }

        for i in list(k_num.keys()):
            out_dict.update({'Recall@' + i: Recall[i]}) 
            out_dict.update({'Recall_Random@' + i: Recall_Random[i]}) 
            out_dict.update({'Enrichment@' + i: Enrichment[i]})
            out_dict.update({'MRR@' + i: MRR[i]})
            out_dict.update({'AP@' + i: AP[i]})
            out_dict.update({'Hits@' + i: in_ranked_list[i]}) 
            out_dict.update({'Missed@' + i: not_in_ranked_list[i]}) 

        return out_dict, disease_not_intersecting_list

    def summary(result, rel_type, mode = 'compound', show_plot = True, verbose = True):
        out_dict_mean = {}
        out_dict_std = {}
        for i in list(result.keys()): 
            if isinstance(list(result[i].values())[0], (int, float)):
                if verbose: 
                    print('---------')
                    print(i + ' mean: ', np.mean(list(result[i].values()))) 
                    print(i + ' std: ', np.std(list(result[i].values()))) 
                    print('---------')
                out_dict_mean[i] = np.mean(list(result[i].values()))
                out_dict_std[i] = np.std(list(result[i].values()))

        if show_plot:
            import seaborn as sns
            import matplotlib.pyplot as plt
            sns.scatterplot(list(range(len(result['Recall@5%']))), list(result['# of Pos'].values())).set_title("#pos scatter plot")
            plt.show()

            for i in ['Recall@1%', 'Recall@5%', 'Recall@10%', 'Recall@10', 'Recall@50', 'Recall@100', 'AUROC', 'AUPRC', 'MRR@10', 'MRR@50', 'MRR@100', 'AP@10', 'AP@50', 'AP@100']:
                sns.distplot(list(result[i].values())).set_title(i + " distribution")
                plt.show()


            preds_ = np.concatenate([np.array(list(j.values())) for i, j in results['Prediction'].items()]).reshape(-1,)
            labels_ = np.concatenate([np.array(list(j.values())) for i, j in results['Labels'].items()]).reshape(-1,)

            preds_pos = preds_[np.where(labels_ == 1)]
            preds_neg = preds_[np.where(labels_ == 0)]

            sns.distplot(preds_neg).set_title("prediction score distribution")
            sns.distplot(preds_pos)
            plt.show()

        return out_dict_mean, out_dict_std 

    def get_scores_disease(rel, disease_ids): 
        df_train_valid = pd.concat([df_train, df_valid]) 
        df_dd = df_test[df_test.relation.isin(disease_rel_types)] 
        df_dd_train = df_train_valid[df_train_valid.relation.isin(disease_rel_types)] 

        df_rel_dd = df_dd[df_dd.relation == rel] 
        df_rel_dd_train = df_dd_train[df_dd_train.relation == rel] 
        compound_nodes = G.nodes('compound').cpu().numpy() 
        
        if disease_ids is None: 
            disease_ids = df_rel_dd.x_idx.unique() 
        preds_contra = {}
        labels_contra = {}
        ids_contra = {}

        for disease_id in tqdm(disease_ids): 
            candidate_pos = df_rel_dd[df_rel_dd.x_idx == disease_id][['x_idx', 'y_idx']] 
            candidate_pos_train = df_rel_dd_train[df_rel_dd_train.x_idx == disease_id][['x_idx', 'y_idx']] 
            compound_pos = candidate_pos.y_idx.values 
            compound_pos_train_val = candidate_pos_train.y_idx.values 

            labels = {} 
            for i in compound_nodes: 
                if i in compound_pos:
                    labels[i] = 1
                elif i in compound_pos_train_val:
                    labels[i] = -1
                    # in the training set
                else:
                    labels[i] = 0

            # construct eval graph
            out = {}
            src = torch.Tensor([disease_id] * len(labels)).to(device).to(dtype = torch.int64) 
            dst = torch.Tensor(list(labels.keys())).to(device).to(dtype = torch.int64) 
            out.update({('disease', rel, 'compound'): (src, dst)}) 

            g_eval = dgl.heterograph(out, num_nodes_dict={ntype: G.number_of_nodes(ntype) for ntype in G.ntypes}).to(device)
            model.eval()
            with torch.no_grad():
                _, pred_score_rel, _, pred_score = model(G, g_eval) 
            pred = pred_score_rel[('disease', rel, 'compound')].reshape(-1,)
            pred = torch.sigmoid(pred).detach().cpu().numpy() 
            lab = {idx2id_compound[i]: labels[i] for i in g_eval.edges()[1].detach().cpu().numpy()} 
            preds_contra[idx2id_disease[disease_id]] = {idx2id_compound[i]: pred[idx] for idx, i in enumerate(g_eval.edges()[1].detach().cpu().numpy())} 
            labels_contra[idx2id_disease[disease_id]] = lab 
            ids_contra[idx2id_disease[disease_id]] = g_eval.edges()[1].detach().cpu().numpy() 

            del pred_score_rel, pred_score
        return preds_contra, labels_contra, compound_nodes, [id2name_compound[idx2id_compound[i]] for i in compound_nodes]
    
    if disease_ids is None: 
        # downstream evaluate all test set diseases
        
        temp_d, preds_all, labels_all, org_out_all, metrics_all = {}, {}, {}, {}, {}

        for rel_type in disease_rel_types: 
            print('Evaluating relation: ' + rel_type[4:])
            preds_, labels_, compound_idxs, compound_names = get_scores_disease(rel_type, disease_ids)
            preds_all[rel_type], labels_all[rel_type] = preds_, labels_ 
            results, _ = calculate_metrics(rel_type, preds_all, labels_all, mode = 'disease') 
            out_dict_mean, out_dict_std = summary(results, rel_type, mode = 'disease', show_plot = show_plot, verbose = verbose) 
            org_out = [[idx, i, out_dict_mean[i], out_dict_std[i]] for idx, i in enumerate(out_dict_mean.keys())] 
            org_out_all[rel_type] = org_out 
            metrics_all[rel_type] = results 
            
            if weight_bias_track:
                temp_d.update({"disease_centric_evaluation_" + rel_type: wandb.Table(data=org_out,
                                    columns = ["metric_id", "metric", "mean", "std"])
                              })

        if weight_bias_track:
            wandb.log(temp_d)
            
        if return_raw:
            out = {'prediction': preds_all, 'label': labels_all, 'summary': org_out_all, 'result': metrics_all}
            return out
        else:
            return {rel_type: pd.DataFrame.from_dict(metrics_all[rel_type]) for rel_type in disease_rel_types} 
    
    else:
        # downstream evaluate a specified list of diseases
        temp_d, preds_all, labels_all, metrics_all = {}, {}, {}, {}
        
        rel_type = 'rev_' + relation 
        
        preds_, labels_, compound_idxs, compound_names = get_scores_disease(rel_type, disease_ids)
        preds_all[rel_type], labels_all[rel_type] = preds_, labels_
        if only_prediction:
            for i, j in labels_all[rel_type].items():
                for k, l in j.items():
                    if l == -1:
                        labels_all[rel_type][i][k] = 1
            
        results, _ = calculate_metrics(rel_type, preds_all, labels_all, mode = 'disease')
        metrics_all[rel_type] = results
    
        if return_raw:
            out = {'prediction': preds_all[rel_type], 'label': labels_all[rel_type], 'result': metrics_all[rel_type]}
            return out
        else:
            return pd.DataFrame.from_dict(results)

def disease_centric_prediction(df, df_train, df_valid, df_test, data_path, G, model, device, disease_ids = None, relation = None, weight_bias_track = False, wandb = None, show_plot = False, verbose = False, return_raw = False, simulate_random = True, only_prediction = False):
    G = G.to(device)  
    model = model.eval()

    dd_etypes = [
                ('compound', 'CPD_DIS_ass', 'disease'), 
                ]
    dd_rel_types = ['CPD_DIS_ass']

    disease_etypes = [('disease', 'rev_CPD_DIS_ass', 'compound')]
    disease_rel_types = ['rev_CPD_DIS_ass']

    df['x_id'] = df.x_id.apply(lambda x: convert2str(x)) 
    df['y_id'] = df.y_id.apply(lambda x: convert2str(x))
    idx2id_compound = dict(df[df.x_type == 'compound'][['x_idx', 'x_id']].drop_duplicates().values) 
    idx2id_compound.update(dict(df[df.y_type == 'compound'][['y_idx', 'y_id']].drop_duplicates().values))
    idx2id_disease = dict(df[df.x_type == 'disease'][['x_idx', 'x_id']].drop_duplicates().values)
    idx2id_disease.update(dict(df[df.y_type == 'disease'][['y_idx', 'y_id']].drop_duplicates().values))

    df_ = pd.read_csv(os.path.join(data_path, 'kg.tsv'),sep='\t') 
    df_['x_id'] = df_.x_id.apply(lambda x: convert2str(x))
    df_['y_id'] = df_.y_id.apply(lambda x: convert2str(x))
    id2name_disease = dict(df_[df_.x_type == 'disease'][['x_id', 'x_name']].drop_duplicates().values) 
    id2name_disease.update(dict(df_[df_.y_type == 'disease'][['y_id', 'y_name']].drop_duplicates().values)) 
    id2name_compound = dict(df_[df_.x_type == 'compound'][['x_id', 'x_name']].drop_duplicates().values) 
    id2name_compound.update(dict(df_[df_.y_type == 'compound'][['y_id', 'y_name']].drop_duplicates().values)) 

    def get_scores_disease(rel, disease_ids): 
        df_train_valid = pd.concat([df_train, df_valid]) 
        df_dd = df_test[df_test.relation.isin(disease_rel_types)] 
        df_dd_train = df_train_valid[df_train_valid.relation.isin(disease_rel_types)] 

        df_rel_dd = df_dd[df_dd.relation == rel] 
        df_rel_dd_train = df_dd_train[df_dd_train.relation == rel] 
        compound_nodes = G.nodes('compound').cpu().numpy() 
        if disease_ids is None: 
            disease_ids = df_rel_dd.x_idx.unique() 
        preds_contra = {}
        labels_contra = {}
        ids_contra = {}

        for disease_id in tqdm(disease_ids): 
            # construct eval graph
            out = {}
            src = torch.Tensor([disease_id] * compound_nodes.shape[0]).to(device).to(dtype = torch.int64) 
            dst = torch.Tensor(list(compound_nodes)).to(device).to(dtype = torch.int64) 
            out.update({('disease', rel, 'compound'): (src, dst)}) 

            g_eval = dgl.heterograph(out, num_nodes_dict={ntype: G.number_of_nodes(ntype) for ntype in G.ntypes}).to(device)
            model.eval()
            with torch.no_grad():
                pred_score_rel,pred_score = model(G, g_eval, mode = 'pred') 
            pred = pred_score_rel[('disease', rel, 'compound')].reshape(-1,)
            pred = torch.sigmoid(pred).detach().cpu().numpy() 
            preds_contra[idx2id_disease[disease_id]] = {idx2id_compound[i]: pred[idx] for idx, i in enumerate(g_eval.edges()[1].detach().cpu().numpy())} 
            labels_contra[idx2id_disease[disease_id]] = {idx2id_compound[i]: (1 if pred[idx]>= 0.5 else 0) for idx, i in enumerate(g_eval.edges()[1].detach().cpu().numpy())}
            ids_contra[idx2id_disease[disease_id]] = g_eval.edges()[1].detach().cpu().numpy() 

            del pred_score_rel, pred_score
        return preds_contra, labels_contra, ids_contra

    preds_all, labels_all, ids_all = {}, {}, {}
    rel_type = 'rev_' + relation 
    preds_, labels_, ids_ = get_scores_disease(rel_type, disease_ids)
    preds_all[rel_type], labels_all[rel_type], ids_all[rel_type] = preds_, labels_, ids_
    if only_prediction:
        for i, j in labels_all[rel_type].items():
            for k, l in j.items():
                if l == -1:
                    labels_all[rel_type][i][k] = 1
        
    if return_raw:
        out = {'prediction': preds_all[rel_type], 'label': labels_all[rel_type], 'idx': ids_all[rel_type]}
        return out


        
def find_two_hops(x_idx_value, x_type_value, df):

    # Identify 1-hop neighbors
    one_hop = df[((df['x_idx'] == x_idx_value) & (df['x_type'] == x_type_value)) | 
                 ((df['y_idx'] == x_idx_value) & (df['y_type'] == x_type_value))]

    # Collect all unique neighbor pairs
    neighbors = set(zip(one_hop['x_idx'], one_hop['x_type'])) | set(zip(one_hop['y_idx'], one_hop['y_type']))

    # Create a DataFrame from neighbors
    neighbors_df = pd.DataFrame(list(neighbors), columns=['idx', 'type'])

    # Find 2-hop neighbors by joining
    two_hop = df.merge(neighbors_df, left_on=['x_idx', 'x_type'], right_on=['idx', 'type'])
    two_hop = two_hop.append(df.merge(neighbors_df, left_on=['y_idx', 'y_type'], right_on=['idx', 'type']))

    # Combine 1-hop and 2-hop neighbors and remove duplicates
    return pd.concat([one_hop, two_hop]).drop_duplicates()

import torch
import numpy as np
import dgl

def remove_random_edges(hetero_graph, K):
    removed_edges = {}
    for etype in hetero_graph.canonical_etypes:
        num_existing_edges = hetero_graph.number_of_edges(etype)
        num_edges_to_remove = int(K / 100 * num_existing_edges)

        np.random.seed(42)
        edges_to_remove = np.random.choice(num_existing_edges, num_edges_to_remove, replace=False)
        edge_ids = torch.tensor(edges_to_remove, dtype=torch.int64).to(hetero_graph.device)

        hetero_graph.remove_edges(edge_ids, etype=etype)
        removed_edges[etype] = edge_ids
        
    return hetero_graph, removed_edges


def remove_relation_type(hetero_graph, relation_type):
    # Check if the relation type is valid
    if relation_type not in hetero_graph.etypes:
        print(f"Relation type {relation_type} not found in the graph.")
        return hetero_graph

    # Find all canonical edge types that correspond to the relation type
    etypes_to_remove = [etype for etype in hetero_graph.canonical_etypes if etype[1] == relation_type]
    etypes_to_remove += [etype for etype in hetero_graph.canonical_etypes if etype[1] == 'rev_' + relation_type]

    # Check if any etypes were found
    if not etypes_to_remove:
        print(f"No edge types found for relation type {relation_type}.")
        return hetero_graph

    # Remove edges of these types
    for etype in etypes_to_remove:
        num_edges = hetero_graph.number_of_edges(etype)
        edge_ids = torch.arange(num_edges, dtype=torch.int64).to(hetero_graph.device)
        hetero_graph.remove_edges(edge_ids, etype=etype)
    
    return hetero_graph


def add_random_edges(hetero_graph, K):
    added_edges = {}
    for etype in hetero_graph.canonical_etypes:
        src_type, rel_type, dst_type = etype

        num_src_nodes = hetero_graph.number_of_nodes(src_type)
        num_dst_nodes = hetero_graph.number_of_nodes(dst_type)
        num_existing_edges = hetero_graph.number_of_edges(etype)
        
        num_edges_to_add = int(K / 100 * num_existing_edges)

        np.random.seed(42)
        src_random_nodes = np.random.randint(0, num_src_nodes, num_edges_to_add)
        dst_random_nodes = np.random.randint(0, num_dst_nodes, num_edges_to_add)

        src_tensor = torch.tensor(src_random_nodes, dtype=torch.int64).to(hetero_graph.device)
        dst_tensor = torch.tensor(dst_random_nodes, dtype=torch.int64).to(hetero_graph.device)

        hetero_graph.add_edges(src_tensor, dst_tensor, etype=etype)
        added_edges[etype] = (src_tensor, dst_tensor)
        
    return hetero_graph, added_edges

import torch
import numpy as np
import dgl

import torch
import numpy as np
import dgl

def randomize_edges(hetero_graph):
    randomized_edges = {}
    for etype in hetero_graph.canonical_etypes:
        src_type, rel_type, dst_type = etype

        # Number of nodes for each node type
        num_src_nodes = hetero_graph.number_of_nodes(src_type)
        num_dst_nodes = hetero_graph.number_of_nodes(dst_type)

        # Number of existing edges for this relation
        num_edges = hetero_graph.number_of_edges(etype)

        # Generate random edges
        np.random.seed(42)
        src_random_nodes = np.random.randint(0, num_src_nodes, num_edges)
        dst_random_nodes = np.random.randint(0, num_dst_nodes, num_edges)

        # Convert to tensors and move to the same device as the graph
        src_tensor = torch.tensor(src_random_nodes, dtype=torch.int64).to(hetero_graph.device)
        dst_tensor = torch.tensor(dst_random_nodes, dtype=torch.int64).to(hetero_graph.device)
        edge_ids = torch.arange(num_edges, dtype=torch.int64).to(hetero_graph.device)

        # Remove existing edges
        hetero_graph.remove_edges(edge_ids, etype=etype)

        # Add new random edges
        hetero_graph.add_edges(src_tensor, dst_tensor, etype=etype)

        randomized_edges[etype] = (src_tensor, dst_tensor)

    return hetero_graph, randomized_edges
