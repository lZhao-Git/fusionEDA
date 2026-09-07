import os
import math
import copy
import pickle

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import dgl

from .utils import preprocess_kg, create_split, process_disease_area_split, create_dgl_graph, evaluate_graph_construct, convert2str
from .utils import  sparsify_train_edges, make_sparse_train_from_saved_pkl

import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import Dataset, DataLoader

class EDAData:
    def __init__(self, data_folder):
        os.makedirs(data_folder,exist_ok=True)    
        self.data_folder = data_folder 
               
    def prepare_split(self, split = 'random',
                    fold = 1,
                    disease_eval_idx = None,
                    seed = 42,
                    no_kg = False,
                    test_size = 0.05,
                    mask_ratio = 0.1, 
                    one_hop = False,
                    edge_keep_ratio=1.0,
                    sparsify_mode='none',
                    sparsify_seed=42):
        
        if split not in ['random', 'complex_disease', 'complex_disease_cv', 'disease_eval', 'nervous_system_diseases','endocrine_system_diseases', 'full_graph', 'downstream_pred', 'few_edeges_to_kg', 'few_edeges_to_CPD_DIS_ass','cv5','cold_cpd','cold_dis']:
            raise ValueError('Please select supported splits')            
        if disease_eval_idx is not None:
            split = 'disease_eval'
            print('disease eval index is not none, use the individual disease split...')
        self.split = split
        
        if split in ['nervous_system_diseases','endocrine_system_diseases']:            
            if test_size != 0.05:
                folder_name = split + '_kg' + '_frac' + str(test_size)
            elif one_hop:
                folder_name = split + '_kg' + '_one_hop_ratio' + str(mask_ratio)
            else:
                folder_name = split + '_kg'
            os.makedirs(os.path.join(self.data_folder, folder_name),exist_ok=True)
            kg_path = os.path.join(self.data_folder, folder_name, 'kg_directed.csv')
        else:
            kg_path = os.path.join(self.data_folder, 'kg_directed.csv')
            
        if os.path.exists(kg_path):
            print('Found saved processed KG... Loading...')
            df = pd.read_csv(kg_path) 
        else:
            if os.path.exists(os.path.join(self.data_folder, 'kg.tsv')):
                print('Mapping raw KG to directed csv... it takes several minutes...')
                preprocess_kg(self.data_folder, split, test_size, one_hop, mask_ratio) 
                df = pd.read_csv(kg_path) 
            else:
                raise ValueError("KG file path does not exist...")
        
        if split == 'disease_eval':
            split_data_path = os.path.join(self.data_folder, self.split + '_' + str(disease_eval_idx))
        elif split == 'downstream_pred':
            split_data_path = os.path.join(self.data_folder, self.split + '_downstream_pred')
            disease_eval_idx = [11394.,  6353., 12696., 14183., 12895.,  9128., 12623., 15129.,
                                   12897., 12860.,  7611., 13113.,  4029., 14906., 13438., 13177.,
                                   13335., 12896., 12879., 12909.,  4815., 12766., 12653.]
        elif no_kg: 
            split_data_path = os.path.join(self.data_folder, self.split + '_no_kg_' + str(seed))
        elif test_size != 0.05: 
            split_data_path = os.path.join(self.data_folder, self.split + '_' + str(seed)) + '_frac' + str(test_size)
        elif one_hop:
            split_data_path = os.path.join(self.data_folder, self.split + '_' + str(seed)) + '_one_hop_ratio' + str(mask_ratio)
        elif split in ['cv5','cold_cpd', 'cold_dis']:
            split_data_path = os.path.join(self.data_folder, 'cv'+str(fold))
            for i in range(1,6):
                os.makedirs(os.path.join(self.data_folder, 'cv'+str(i)),exist_ok=True)
        else:
            split_data_path = os.path.join(self.data_folder, self.split + '_' + str(seed)) 
        
        if no_kg:
            sub_kg = ['CPD_DIS_ass']
            df = df[df.relation.isin(sub_kg)].reset_index(drop = True)        
               
        if not os.path.exists(os.path.join(split_data_path, 'train.csv')):
            os.makedirs(split_data_path,exist_ok=True)         
            print('Creating splits... it takes several minutes...')
            df_train, df_valid, df_test = create_split(self.data_folder, df, split, disease_eval_idx, split_data_path, seed, fold) 
        else:
            print('Loading split data....')
            df_train = pd.read_csv(os.path.join(split_data_path, 'train.csv')) 
            df_valid = pd.read_csv(os.path.join(split_data_path, 'valid.csv')) 
            df_test = pd.read_csv(os.path.join(split_data_path, 'test.csv')) 

        if split not in ['random', 'complex_disease', 'complex_disease_cv', 'disease_eval', 'full_graph', 'downstream_pred', 'few_edeges_to_CPD_DIS_ass', 'few_edeges_to_kg','cv5','cold_cpd','cold_dis']:
            # in disease area split
            df_test = process_disease_area_split(self.data_folder, df, df_test, split,split_data_path)

        if sparsify_mode != 'none':
            if sparsify_mode not in ["all", "context_only"]:
                raise ValueError("sparsify_mode must be one of ['none', 'all', 'context_only'].")
            df_train_original = df_train.copy()
            df_train = sparsify_train_edges(
                                            df_train=df_train,
                                            keep_ratio=edge_keep_ratio,
                                            seed=sparsify_seed,
                                            mode=sparsify_mode,
                                            target_rel=("CPD_DIS_ass", "rev_CPD_DIS_ass"),
                                            min_edges_per_rel=1,
                                            preserve_reverse_pair=True,
                                        )
            sparse_tag = f"sparse_{sparsify_mode}_keep{edge_keep_ratio}_seed{sparsify_seed}"
            sparse_out_dir = os.path.join(split_data_path, sparse_tag)
            os.makedirs(sparse_out_dir, exist_ok=True)
            df_train_original.to_csv(os.path.join(sparse_out_dir, "train_original.csv"),index=False )
            df_train.to_csv(os.path.join(sparse_out_dir, "train_sparse.csv"),index=False)

        g = create_dgl_graph(df_train, df) 
        self.G = g         
        self.df, self.df_train, self.df_valid, self.df_test = df, df_train, df_valid, df_test
        self.disease_eval_idx = disease_eval_idx
        self.no_kg = no_kg
        self.seed = seed

    def prepare_split_from_pkl(self,split,base_pkl_dir,out_pkl_dir,keep_ratio,seed,mode,target_rel,disease_eval_idx = None,no_kg = False):
        self.split = split
        self.G, _ = make_sparse_train_from_saved_pkl(base_pkl_dir,out_pkl_dir,keep_ratio,seed,mode,target_rel)
        self.df = pd.read_csv(os.path.join(self.data_folder, "kg_directed.csv"))
        self.df_train = None
        self.df_valid = None
        self.df_test = None
        self.disease_eval_idx = disease_eval_idx
        self.no_kg = no_kg
        self.seed = seed

              
    def retrieve_id_mapping(self):
        df = self.df 
        print(df.shape)
        df['x_id'] = df.x_id.apply(lambda x: convert2str(x))
        df['y_id'] = df.y_id.apply(lambda x: convert2str(x))

        idx2id_drug = dict(df[df.x_type == 'compound'][['x_idx', 'x_id']].drop_duplicates().values) 
        idx2id_drug.update(dict(df[df.y_type == 'compound'][['y_idx', 'y_id']].drop_duplicates().values))

        idx2id_disease = dict(df[df.x_type == 'disease'][['x_idx', 'x_id']].drop_duplicates().values)
        idx2id_disease.update(dict(df[df.y_type == 'disease'][['y_idx', 'y_id']].drop_duplicates().values))

        df_ = pd.read_csv(os.path.join(self.data_folder, 'kg.tsv'),sep='\t') 
        df_['x_id'] = df_.x_id.apply(lambda x: convert2str(x))
        df_['y_id'] = df_.y_id.apply(lambda x: convert2str(x))

        id2name_disease = dict(df_[df_.x_type == 'disease'][['x_id', 'x_name']].drop_duplicates().values) 
        id2name_disease.update(dict(df_[df_.y_type == 'disease'][['y_id', 'y_name']].drop_duplicates().values))

        id2name_drug = dict(df_[df_.x_type == 'compound'][['x_id', 'x_name']].drop_duplicates().values)  
        id2name_drug.update(dict(df_[df_.y_type == 'compound'][['y_id', 'y_name']].drop_duplicates().values))
        
        return {'id2name_drug': id2name_drug,
                'id2name_disease': id2name_disease,
                'idx2id_disease': idx2id_disease,
                'idx2id_drug': idx2id_drug
               }
    
class TokenEmbLoad:
    """
        {
            node_idx: {
                "emb": np.ndarray [L, D],
                "mask": np.ndarray [L],
                "input_ids": ...
            }
        }
    """
    def __init__(self, pkl_path, name="node_token"):
        self.pkl_path = pkl_path
        self.name = name
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"{name} pkl not found: {pkl_path}")
        with open(pkl_path, "rb") as f:
            self.node_token = pickle.load(f)
        self.node_keys = set(int(k) for k in self.node_token.keys())

    def get(self, idx):
        idx = int(idx)
        if idx not in self.node_keys:
            raise KeyError(f"{self.name}: idx={idx} not found in {self.pkl_path}. ")
        item = self.node_token[idx]
        emb = torch.as_tensor(item["emb"], dtype=torch.float32)
        if "mask" in item:
            mask = torch.as_tensor(item["mask"], dtype=torch.bool)
        else:
            mask = torch.ones(emb.shape[0], dtype=torch.bool)
        return emb, mask

class ShardedTokenStore:
    """
     compound token embedding。

        cpd_token_emb_index.pkl
        cpd_token_emb_part_0000.pkl
        cpd_token_emb_part_0001.pkl
        ...

    index pkl :
        {
            node_idx: shard_file_name
        }

    shard pkl :
        {
            node_idx: {
                "emb": np.ndarray [L, D],
                "mask": np.ndarray [L],
                "input_ids": np.ndarray [L]
            }
        }
    """

    def __init__(
        self,
        index_path,
        shard_dir=None,
        name="compound_token",
        cache_size=2,
    ):
        self.index_path = index_path
        self.name = name
        self.cache_size = cache_size

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"{name} index pkl not found: {index_path}")

        if shard_dir is None:
            shard_dir = os.path.dirname(index_path)

        self.shard_dir = shard_dir

        with open(index_path, "rb") as f:
            self.index = pickle.load(f)


        self.index = {int(k): v for k, v in self.index.items()}

        self.keys = set(self.index.keys())
        self.cache = {}
        self.cache_order = []

        print(
            f"[{self.name}] loaded index: {index_path}, "
            f"n={len(self.index)}, shard_dir={self.shard_dir}"
        )

    def _load_shard(self, shard_name):
        if shard_name in self.cache:
            return self.cache[shard_name]

        shard_path = os.path.join(self.shard_dir, shard_name)

        if not os.path.exists(shard_path):
            raise FileNotFoundError(
                f"{self.name}: shard file not found: {shard_path}"
            )

        with open(shard_path, "rb") as f:
            shard_data = pickle.load(f)

        shard_data = {int(k): v for k, v in shard_data.items()}

        self.cache[shard_name] = shard_data
        self.cache_order.append(shard_name)

        if len(self.cache_order) > self.cache_size:
            old = self.cache_order.pop(0)
            if old in self.cache:
                del self.cache[old]

        return shard_data

    def get(self, idx):
        idx = int(idx)

        if idx not in self.index:
            raise KeyError(
                f"{self.name}: idx={idx} not found in index file: {self.index_path}. "
                f"check compound x_idx cpd_token_emb_index.pkl  key 。"
            )

        shard_name = self.index[idx]
        shard_data = self._load_shard(shard_name)

        if idx not in shard_data:
            raise KeyError(
                f"{self.name}: idx={idx} is mapped to {shard_name}, "
                f"but not found inside that shard."
            )

        item = shard_data[idx]

        emb = torch.as_tensor(item["emb"], dtype=torch.float32)

        if "mask" in item:
            mask = torch.as_tensor(item["mask"], dtype=torch.bool)
        else:
            mask = torch.ones(emb.shape[0], dtype=torch.bool)

        return emb, mask
        
class EdgePairDataset(Dataset):
    def __init__(self, cpd_idx, dis_idx, labels):
        self.cpd_idx = torch.as_tensor(cpd_idx, dtype=torch.long)
        self.dis_idx = torch.as_tensor(dis_idx, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, i):
        return int(self.cpd_idx[i]), int(self.dis_idx[i]), float(self.labels[i])

def token_collate_fn(cpd_load, dis_load, h_fea, max_cpd_len=None, max_dis_len=None):
    def _pad_token_list(token_list, mask_list, max_len=None):
        B = len(token_list)
        D = token_list[0].shape[-1]
        if max_len is None:
            L = max(x.shape[0] for x in token_list)
        else:
            L = min(max_len, max(x.shape[0] for x in token_list))

        emb_out = torch.zeros(B, L, D, dtype=torch.float32)
        mask_out = torch.zeros(B, L, dtype=torch.bool)
        for i, (tok, m) in enumerate(zip(token_list, mask_list)):
            l = min(tok.shape[0], L)
            emb_out[i, :l] = tok[:l]
            mask_out[i, :l] = m[:l].bool()
        return emb_out, mask_out

    def collate(batch):
        cpd_idx, dis_idx, labels = zip(*batch)
        cpd_tokens = []
        cpd_masks = []
        dis_tokens = []
        dis_masks = []
        for c_idx, d_idx in zip(cpd_idx, dis_idx):
            c_tok, c_mask = cpd_load.get(c_idx)
            d_tok, d_mask = dis_load.get(d_idx)
            cpd_tokens.append(c_tok)
            cpd_masks.append(c_mask)
            dis_tokens.append(d_tok)
            dis_masks.append(d_mask)
        cpd_tok, cpd_mask = _pad_token_list(cpd_tokens, cpd_masks, max_cpd_len)
        dis_tok, dis_mask = _pad_token_list(dis_tokens, dis_masks, max_dis_len)
        cpd_idx_b = torch.as_tensor(cpd_idx, dtype=torch.long)
        dis_idx_b = torch.as_tensor(dis_idx, dtype=torch.long)
        h_cpd = h_fea["compound"][cpd_idx_b].float()
        h_dis = h_fea["disease"][dis_idx_b].float()
        labels_b = torch.as_tensor(labels, dtype=torch.float32)
        return h_cpd, h_dis, cpd_tok, cpd_mask, dis_tok, dis_mask, labels_b
    return collate

def extract_edges_from_graph(g_pos, g_neg, target_etypes):
    cpd_list = []
    dis_list = []
    y_list = []
    def _append_from_graph(g, label):
        for etype in target_etypes:
            if etype not in g.canonical_etypes:
                print(f"[Warning] {etype} not in graph, skip.")
                continue
            src_type, rel, dst_type = etype
            u, v = g.edges(etype=etype)
            if u.numel() == 0:
                print(f"[Warning] no edges for {etype}, skip.")
                continue
            u = u.detach().cpu().long()
            v = v.detach().cpu().long()
            if src_type == "compound" and dst_type == "disease":
                cpd_idx = u
                dis_idx = v
            elif src_type == "disease" and dst_type == "compound":
                cpd_idx = v
                dis_idx = u
            else:
                raise ValueError(f"Only compound-disease etypes are supported, got {etype}")
            cpd_list.append(cpd_idx)
            dis_list.append(dis_idx)
            y_list.append(torch.full((cpd_idx.shape[0],), float(label), dtype=torch.float32))
    _append_from_graph(g_pos, 1.0)
    _append_from_graph(g_neg, 0.0)

    if len(cpd_list) == 0:
        raise RuntimeError("No target edges were extracted from pos/neg graphs.")

    cpd_idx_all = torch.cat(cpd_list, dim=0)
    dis_idx_all = torch.cat(dis_list, dim=0)
    labels_all = torch.cat(y_list, dim=0)
    perm = torch.randperm(labels_all.shape[0])
    return cpd_idx_all[perm], dis_idx_all[perm], labels_all[perm]