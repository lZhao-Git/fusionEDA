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
import copy
import os
import random

import warnings
warnings.filterwarnings("ignore")
from .utils import sim_matrix, exponential, obtain_disease_profile, obtain_protein_random_walk_profile, convert2str


class ComplExPredictor(nn.Module):
    def __init__(self, n_hid, w_rels, G, rel2idx, proto, proto_num, sim_measure, bert_measure, agg_measure, num_walks, walk_mode, path_length, split, data_folder, exp_lambda, device, eval_dd_etypes):
        super().__init__()
        self.proto = proto 
        self.sim_measure = sim_measure 
        self.bert_measure = bert_measure 
        self.agg_measure = agg_measure 
        self.num_walks = num_walks 
        self.walk_mode = walk_mode 
        self.path_length = path_length 
        self.exp_lambda = exp_lambda 
        self.device = device
        self.W = w_rels 
        self.rel2idx = rel2idx        
        self.etypes_dd = [('compound', 'CPD_DIS_ass', 'disease'),
                           ('disease', 'rev_CPD_DIS_ass', 'compound')]
        self.eval_dd_etypes = eval_dd_etypes
        self.node_types_dd = ['disease', 'compound']
        
                
    def apply_edges(self, edges): 
        h_u = edges.src['h'] 
        h_v = edges.dst['h'] 
        rel_idx = self.rel2idx[edges._etype] 
        h_r = self.W[rel_idx] 
        real_head, img_head = torch.chunk(h_u, 2, dim=-1)
        real_tail, img_tail = torch.chunk(h_v, 2, dim=-1)
        real_rel, img_rel = torch.chunk(h_r, 2, dim=-1)
        score = real_head * real_tail * real_rel \
                + img_head * img_tail * real_rel \
                + real_head * img_tail * img_rel \
                - img_head * real_tail * img_rel
        score = torch.sum(score, -1)
        return {'score': score}

    def forward(self, graph, G, h, pretrain_mode, mode, block = None, only_relation = None):
        with graph.local_scope():
            scores = {}
            s_l = []
            
            if len(graph.canonical_etypes) == 1: 
                etypes_train = graph.canonical_etypes 
            else:
                etypes_train = self.etypes_dd         
            if only_relation is not None: 
                if only_relation == 'CPD_DIS_ass':
                    etypes_train = [('compound', 'CPD_DIS_ass', 'disease'), 
                                   ('disease', 'rev_CPD_DIS_ass', 'compound')]
                else:
                    return ValueError
            
            graph.ndata['h'] = h            
            if pretrain_mode:
                etypes_all = [i for i in graph.canonical_etypes if graph.edges(etype = i)[0].shape[0] != 0]
                for etype in etypes_all: 
                    graph.apply_edges(self.apply_edges, etype=etype)    
                    out = torch.sigmoid(graph.edges[etype].data['score']) 
                    s_l.append(out) 
                    scores[etype] = out 

            if pretrain_mode:
                s_l = torch.cat(s_l)   
            else: 
                s_l = torch.cat(s_l).reshape(-1,).detach().cpu().numpy() 
            return scores, s_l

class FusionMLPPredictor(nn.Module):
    def __init__(self, h_dim, hidden_dim, dropout, eval_dd_etypes, G, proto, proto_num, sim_measure, bert_measure, agg_measure, num_walks, walk_mode, path_length, data_folder, exp_lambda, device):
        super().__init__()
        self.eval_dd_etypes = eval_dd_etypes
        self.proto = proto
        self.sim_measure = sim_measure
        self.bert_measure = bert_measure
        self.agg_measure = agg_measure
        self.num_walks = num_walks
        self.walk_mode = walk_mode
        self.path_length = path_length
        self.exp_lambda = exp_lambda
        self.device = device
        self.node_types_dd = ['disease', 'compound']

        if proto:
            self.W_gate = {}
            for i in self.node_types_dd:
                temp_w = nn.Linear(h_dim * 2, 1)
                nn.init.xavier_uniform_(temp_w.weight)
                self.W_gate[i] = temp_w.to(self.device)
            self.k = proto_num
            self.m = nn.Sigmoid()
                   
            if sim_measure == 'llm':
                if "inp" not in G.nodes["compound"].data:
                    raise KeyError('G.nodes["compound"].data["inp"] not found.')
                num_cpd = G.num_nodes("compound")
                cpd_total_degree = torch.zeros(num_cpd,dtype=torch.long,device=G.device)
                for canonical_etype in G.canonical_etypes:
                    src_type, rel_type, dst_type = canonical_etype
                    if src_type == "compound":
                        cpd_total_degree += G.out_degrees(etype=canonical_etype)
                    if dst_type == "compound":
                        cpd_total_degree += G.in_degrees(etype=canonical_etype)
                self.cpd_total_degree = (cpd_total_degree.detach().cpu())
                self.candidate_ids = torch.where(self.cpd_total_degree > 0)[0]
                self.cold_cpd_ids = torch.where(self.cpd_total_degree == 0)[0]
                if self.candidate_ids.numel() == 0:
                    raise RuntimeError( "No compound with degree > 0 exists in training graph G.")
                cpd_inp = (G.nodes["compound"].data["inp"].detach().float().cpu())
                self.cpd_inp_all = cpd_inp
                candidate_inp = cpd_inp[self.candidate_ids]
                self.candidate_feat = F.normalize(candidate_inp,p=2, dim=1)
            else:                  
                if sim_measure in ['bert', 'profile+bert']:                
                    if "inp" not in G.nodes["disease"].data:
                        raise KeyError('G.nodes["disease"].data["inp"] not found. ')   
                
                self.diseases_profile = {}
                self.sim_all_etypes = {}
                self.diseaseid2id_etypes = {}
                self.diseases_profile_etypes = {}       
                disease_etypes = ['rev_GEN_DIS_ass','rev_MF_DIS_ass','rev_BP_DIS_ass','rev_CC_DIS_ass']
                disease_nodes = ['gene/protein','molecular_function','biological_process','cellular_component']
                disease_etypes_all = ['DIS_DIS', 'rev_GEN_DIS_ass','rev_MF_DIS_ass','rev_BP_DIS_ass','rev_CC_DIS_ass','rev_PWY_DIS_ass']
                disease_nodes_all = ['disease', 'gene/protein','molecular_function','biological_process','cellular_component','pathway']

                for etype in self.eval_dd_etypes:
                    src, dst = etype[0], etype[2]
                    if src == 'disease':
                        all_disease_ids = torch.where(G.out_degrees(etype=etype) != 0)[0]
                    elif dst == 'disease':
                        all_disease_ids = torch.where(G.in_degrees(etype=etype) != 0)[0]
                        
                    if sim_measure == 'all_nodes_profile':
                        diseases_profile = {i.item(): obtain_disease_profile(G, i, disease_etypes, disease_nodes) for i in all_disease_ids}
                    elif sim_measure == 'all_nodes_profile_more':
                        diseases_profile = {i.item(): obtain_disease_profile(G, i, disease_etypes_all, disease_nodes_all) for i in all_disease_ids}
                    elif sim_measure == 'protein_profile':
                        diseases_profile = {i.item(): obtain_disease_profile(G, i, ['rev_disease_protein'], ['gene/protein']) for i in all_disease_ids}
                    elif sim_measure == 'protein_random_walk':
                        diseases_profile = {i.item(): obtain_protein_random_walk_profile(i, num_walks, path_length, G, disease_etypes, disease_nodes, walk_mode) for i in all_disease_ids}
                    elif sim_measure == 'bert':
                        disease_inp = G.nodes["disease"].data["inp"].float().cpu()
                        diseases_profile = {i.item(): disease_inp[i.item()].detach().clone() for i in all_disease_ids}
                    elif sim_measure == 'profile+bert':
                        disease_inp = G.nodes["disease"].data["inp"].float().cpu()
                        diseases_profile = {i.item(): torch.cat((obtain_disease_profile(G, i, disease_etypes, disease_nodes).float().cpu(),disease_inp[i.item()].detach().clone())) for i in all_disease_ids}
                        
                    diseaseid2id = dict(zip(all_disease_ids.detach().cpu().numpy(), range(len(all_disease_ids))))
                    disease_profile_tensor = torch.stack([diseases_profile[i.item()] for i in all_disease_ids])
                    sim_all = sim_matrix(disease_profile_tensor, disease_profile_tensor)
                    
                    self.sim_all_etypes[etype] = sim_all
                    self.diseaseid2id_etypes[etype] = diseaseid2id
                    self.diseases_profile_etypes[etype] = diseases_profile

        node_dim = h_dim+768
        input_dim = node_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Linear(input_dim, 1),
        )

    def apply_edges(self, edges):
        h_u = edges.src["h"]
        h_v = edges.dst["h"]
        llm_u = edges.src["llm"]
        llm_v = edges.dst["llm"]
        z_u = torch.cat([h_u, llm_u], dim=-1)
        z_v = torch.cat([h_v, llm_v], dim=-1)
        x = torch.cat([z_u, z_v], dim=-1)
        score = self.mlp(x).squeeze(-1)
        return {"score": score}

    def _build_cold_embedding(self,query_ids,h_cpd,chunk_size=50000):
        """
        Construct KG-like embeddings for unseen compounds.
        Parameters
        ----------
        query_ids : LongTensor [B]
            Compound node indices with total degree == 0 in training KG.
        h_compound : Tensor [N_compound, H]
            RGCN output embeddings h for compound nodes.
        chunk_size : int
            Number of candidate compounds per similarity block.
        Returns
        -------
        cold_h : Tensor [B, H]
            Similarity-weighted KG embedding for each unseen compound.
        topk_ids : LongTensor [B, k]
            Actual node indices of top-k similar seen compounds.
        topk_sim : Tensor [B, k]
            Cosine similarities in pretrained molecular embedding space.
        """
        if query_ids.numel() == 0:
            return None, None, None
        h_device = h_cpd.device
        query_ids_cpu = query_ids.detach().cpu()
        query_feat = self.cpd_inp_all[query_ids_cpu]
        query_feat = F.normalize(query_feat, p=2,dim=1)

        candidate_ids_degree = self.candidate_ids
        candidate_feat_degree = self.candidate_feat
        n_candidate = candidate_ids_degree.numel()
        k = min(self.k,n_candidate)
        if k <= 0:
            raise RuntimeError("No valid compound prototype candidate.")

        best_sim = None
        best_ids = None
        query_feat_gpu = query_feat.to(h_device)
        for start in range(0, n_candidate,chunk_size):
            end = min(start + chunk_size,n_candidate)
            candidate_ids_chunk = candidate_ids_degree[start:end]
            candidate_feat_chunk = candidate_feat_degree[start:end].to(h_device)
            sim_chunk = torch.matmul(query_feat_gpu,candidate_feat_chunk.T)
            local_k = min(k,sim_chunk.shape[1])
            local_sim, local_pos = torch.topk(sim_chunk, k=local_k,dim=1,largest=True,sorted=True)

            local_ids = candidate_ids_chunk[local_pos.detach().cpu()].to(h_device)
            if best_sim is None:
                best_sim = local_sim
                best_ids = local_ids
            else:
                merged_sim = torch.cat([best_sim, local_sim],dim=1)
                merged_ids = torch.cat([best_ids, local_ids],dim=1)
                best_sim, keep_pos = torch.topk(merged_sim,k=k,dim=1,largest=True, sorted=True)
                best_ids = torch.gather(merged_ids,dim=1,index=keep_pos)
        topk_h = h_cpd[best_ids]
        weights = torch.softmax(best_sim,dim=1)
        cold_h = (topk_h* weights.unsqueeze(-1)).sum(dim=1)
        # cold_h = topk_h.mean(dim=1)
        return cold_h, best_ids, best_sim

    def forward(self, graph, G, h,llm_h, pretrain_mode, mode):
        with graph.local_scope():
            scores = {}
            score_list = []
            graph.ndata["llm"] = llm_h
            etypes_train = self.eval_dd_etypes
            for etype in etypes_train:
                if etype not in graph.canonical_etypes:
                    continue
                if graph.num_edges(etype) == 0:
                    continue
                src, dst = etype[0], etype[2]
                h_use = {ntype: emb for ntype, emb in h.items()}
                if (self.proto and self.sim_measure == 'llm'):
                    if src == 'compound':
                        query_ids = torch.where(graph.out_degrees(etype=etype) > 0)[0]
                    elif dst == 'compound':
                        query_ids = torch.where(graph.in_degrees(etype=etype) > 0)[0]
                    else:
                        query_ids = None
                    if (query_ids is not None and query_ids.numel() > 0):
                        query_ids_cpu = (query_ids.detach().cpu())
                        query_degree = (self.cpd_total_degree[query_ids_cpu])
                        unseen_mask_cpu = (query_degree == 0)
                        unseen_ids = query_ids[unseen_mask_cpu.to(query_ids.device)]
                        if unseen_ids.numel() > 0:
                            cold_h, topk_ids, topk_sim = (self._build_cold_embedding(query_ids=unseen_ids, h_cpd=h["compound"]))
                            h_use["compound"] = (h["compound"].clone())
                            h_use["compound"][unseen_ids] = cold_h                   
                elif (self.proto and self.sim_measure != 'llm'):
                    src_rel_idx = torch.where(graph.out_degrees(etype=etype) != 0)
                    dst_rel_idx = torch.where(graph.in_degrees(etype=etype) != 0)
                    src_h = h[src][src_rel_idx]
                    dst_h = h[dst][dst_rel_idx]

                    src_rel_idx_keys = torch.where(G.out_degrees(etype=etype) != 0)
                    dst_rel_idx_keys = torch.where(G.in_degrees(etype=etype) != 0)
                    src_h_keys = h[src][src_rel_idx_keys]
                    dst_h_keys = h[dst][dst_rel_idx_keys]

                    h_disease = {}
                    if src == 'disease':
                        h_disease['disease_query'] = src_h
                        h_disease['disease_key'] = src_h_keys
                        h_disease['disease_query_id'] = src_rel_idx
                        h_disease['disease_key_id'] = src_rel_idx_keys
                    elif dst == 'disease':
                        h_disease['disease_query'] = dst_h
                        h_disease['disease_key'] = dst_h_keys
                        h_disease['disease_query_id'] = dst_rel_idx
                        h_disease['disease_key_id'] = dst_rel_idx_keys

                    if self.sim_measure in ['all_nodes_profile', 'all_nodes_profile_more', 'protein_profile','protein_random_walk', 'bert', 'profile+bert', 'all_nodes_profile_more']:
                        try:
                            sim = self.sim_all_etypes[etype][np.array([self.diseaseid2id_etypes[etype][i.item()] for i in h_disease['disease_query_id'][0]])]
                        except:
                            disease_etypes = ['rev_GEN_DIS_ass','rev_MF_DIS_ass','rev_BP_DIS_ass','rev_CC_DIS_ass']
                            disease_nodes = ['gene/protein','molecular_function','biological_process','cellular_component']
                            disease_etypes_all = ['DIS_DIS', 'rev_GEN_DIS_ass','rev_MF_DIS_ass','rev_BP_DIS_ass','rev_CC_DIS_ass','rev_PWY_DIS_ass']
                            disease_nodes_all = ['disease', 'gene/protein','molecular_function','biological_process','cellular_component','pathway']  

                            # new disease not seen in the training set
                            for i in h_disease['disease_query_id'][0]:
                                if i.item() not in self.diseases_profile_etypes[etype]:
                                    if self.sim_measure == 'all_nodes_profile':
                                        self.diseases_profile_etypes[etype][i.item()] = obtain_disease_profile(G, i, disease_etypes, disease_nodes)
                                    elif self.sim_measure == 'all_nodes_profile_more':
                                        self.diseases_profile_etypes[etype][i.item()] = obtain_disease_profile(G, i, disease_etypes_all, disease_nodes_all)    
                                    elif self.sim_measure == 'protein_profile':
                                        self.diseases_profile_etypes[etype][i.item()] = obtain_disease_profile(G, i, ['rev_GEN_DIS_ass'], ['gene/protein'])
                                    elif self.sim_measure == 'protein_random_walk':
                                        self.diseases_profile_etypes[etype][i.item()] = obtain_protein_random_walk_profile(i, self.num_walks, self.path_length, G, disease_etypes, disease_nodes, self.walk_mode)
                                    elif self.sim_measure == "bert":
                                        disease_inp = G.nodes["disease"].data["inp"].float().cpu()
                                        self.diseases_profile_etypes[etype][i.item()] = (disease_inp[i.item()].detach().clone())
                                    elif self.sim_measure == "profile+bert":
                                        disease_inp = G.nodes["disease"].data["inp"].float().cpu()
                                        self.diseases_profile_etypes[etype][i.item()] = torch.cat((obtain_disease_profile(G, i, disease_etypes, disease_nodes).float().cpu(), disease_inp[i.item()].detach().clone()))
                                        
                            profile_query = [self.diseases_profile_etypes[etype][i.item()] for i in h_disease['disease_query_id'][0]]
                            profile_query = torch.cat(profile_query).view(len(profile_query), -1)
                            profile_keys = [self.diseases_profile_etypes[etype][i.item()] for i in h_disease['disease_key_id'][0]]
                            profile_keys = torch.cat(profile_keys).view(len(profile_keys), -1)

                            sim = sim_matrix(profile_query, profile_keys)

                        if src_h.shape[0] == src_h_keys.shape[0]:
                            # during training...
                            coef = torch.topk(sim, self.k + 1).values[:, 1:]
                            coef = F.normalize(coef, p=1, dim=1)
                            embed = h_disease['disease_key'][torch.topk(sim, self.k + 1).indices[:, 1:]]
                        else:
                            # during evaluation...
                            coef = torch.topk(sim, self.k).values[:, :]
                            coef = F.normalize(coef, p=1, dim=1)
                            embed = h_disease['disease_key'][torch.topk(sim, self.k).indices[:, :]]
                        out = torch.mul(embed, coef.unsqueeze(dim = 2).to(self.device)).sum(dim = 1)

                    if self.sim_measure in ['all_nodes_profile', 'all_nodes_profile_more', 'protein_profile', 'protein_random_walk', 'bert', 'profile+bert']:
                        # for protein profile, we are only looking at diseases for now...
                        if self.agg_measure == 'learn':
                            coef_all = self.m(self.W_gate['disease'](torch.cat((h_disease['disease_query'], out), dim = 1)))
                            proto_emb = (1 - coef_all)*h_disease['disease_query'] + coef_all*out
                        elif self.agg_measure == 'heuristics-0.8':
                            proto_emb = 0.8*h_disease['disease_query'] + 0.2*out
                        elif self.agg_measure == 'avg':
                            proto_emb = 0.5*h_disease['disease_query'] + 0.5*out
                        elif self.agg_measure == 'rarity':
                            if src == 'disease':
                                coef_all = exponential(G.out_degrees(etype=etype)[torch.where(graph.out_degrees(etype=etype) != 0)], self.exp_lambda).reshape(-1, 1)
                            elif dst == 'disease':
                                coef_all = exponential(G.in_degrees(etype=etype)[torch.where(graph.in_degrees(etype=etype) != 0)], self.exp_lambda).reshape(-1, 1)
                            proto_emb = (1 - coef_all)*h_disease['disease_query'] + coef_all*out
                        elif self.agg_measure == '100proto':
                            proto_emb = out
                        h_use["disease"] = (h["disease"].clone())
                        h_use['disease'][h_disease['disease_query_id']] = proto_emb
                graph.ndata['h'] = h_use
                graph.apply_edges(self.apply_edges, etype=etype)
                out = graph.edges[etype].data["score"]
                scores[etype] = out
                score_list.append(out)

            if len(score_list) == 0:
                raise RuntimeError("Predictor did not score any edge. " "Please check target etypes.")
            return scores, torch.cat(score_list, dim=0)

class AttHeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(AttHeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })
        
        self.attn_fc = nn.ModuleDict({
                name : nn.Linear(out_size * 2, 1, bias = False) for name in etypes
            })
    
    def edge_attention(self, edges):
        src_type = edges._etype[0]
        etype = edges._etype[1]
        dst_type = edges._etype[2]
        try:
            if src_type == dst_type:
                wh2 = torch.cat([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % etype]], dim=1)
            else:
                if etype[:3] == 'rev':
                    wh2 = torch.cat([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % etype[4:]]], dim=1)
                else:
                    wh2 = torch.cat([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % 'rev_' + etype]], dim=1)
        except:
            print(edges.src.keys())
            print(edges.dst.keys())
            raise ValueError
        a = self.attn_fc[etype](wh2)
        return {'e_%s' % etype: F.leaky_relu(a)}

    def message_func(self, edges):
        etype = edges._etype[1]
        return {'m': edges.src['Wh_%s' % etype], 'e': edges.data['e_%s' % etype]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['m'], dim=1)
        return {'h': h}
    
    def forward(self, G, feat_dict, return_att = False):
        with G.local_scope():        
            funcs = {}
            att = {}
            etypes_all = [i for i in G.canonical_etypes if G.edges(etype = i)[0].shape[0] != 0]
            for srctype, etype, dsttype in etypes_all:
                Wh = self.weight[etype](feat_dict[srctype])
                G.nodes[srctype].data['Wh_%s' % etype] = Wh
            
            for srctype, etype, dsttype in etypes_all:
                try:
                    G.apply_edges(self.edge_attention, etype=etype)
                except:
                    print(etype)
                    # Assuming 'etype' is your edge type of interest
                    src, dst, eid = G.edges(etype=etype, form='all')

                    print(src)
                    print(dst)
                    print(f"Edge type: {etype}")
                    print(f"Source type: {srctype} Keys:", G.nodes[srctype].data.keys())
                    print(f"Destination type: {dsttype} Keys:", G.nodes[dsttype].data.keys())
                    if G.nodes[srctype].data:
                        print("Keys:", G.nodes[srctype].data.keys())
                    if G.nodes[dsttype].data:
                        print("Keys:", G.nodes[dsttype].data.keys())
                    raise ValueError
                if return_att:
                    att[(srctype, etype, dsttype)] = G.edges[etype].data['e_%s' % etype].detach().cpu().numpy()
                funcs[etype] = (self.message_func, self.reduce_func)
                
            G.multi_update_all(funcs, 'sum')
            
            return {ntype : G.dstdata['h'][ntype] for ntype in list(G.dstdata['h'].keys())}, att
    
class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes): 
        super(HeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({name : nn.Linear(in_size, out_size) for name in etypes}) 
        self.in_size = in_size 
        self.out_size = out_size 
            
        self.gate_storage = {}
        self.gate_score_storage = {}
        self.gate_penalty_storage = {}
            
    def add_graphmask_parameter(self, gate, baseline, layer):
        self.gate = gate 
        self.baseline = baseline 
        self.layer = layer 
        
    def forward(self, G, feat_dict): 
        funcs = {}
        etypes_all = [i for i in G.canonical_etypes if G.edges(etype = i)[0].shape[0] != 0] 
        
        for srctype, etype, dsttype in etypes_all:  
            Wh = self.weight[etype](feat_dict[srctype]) 
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))  
        G.multi_update_all(funcs, 'sum')       
        return {ntype : G.dstdata['h'][ntype] for ntype in list(G.dstdata['h'].keys())} 
 
    def gm_online(self, edges): 
        etype = edges._etype[1]
        srctype = edges._etype[0]
        dsttype = edges._etype[2]
        
        if srctype == dsttype: 
            gate, penalty, gate_score, penalty_not_sum = self.gate[etype][self.layer]([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % etype]])
        else:
            if etype[:3] == 'rev':                
                gate, penalty, gate_score, penalty_not_sum = self.gate[etype][self.layer]([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % etype[4:]]])
            else:
                gate, penalty, gate_score, penalty_not_sum = self.gate[etype][self.layer]([edges.src['Wh_%s' % etype], edges.dst['Wh_%s' % 'rev_' + etype]])
                
        #self.penalty += len(edges.src['Wh_%s' % etype])/self.num_of_edges * penalty
        #self.penalty += penalty
        self.penalty.append(penalty) 
        
        self.num_masked += len(torch.where(gate.reshape(-1) != 1)[0]) 
        if self.no_base:
            message = gate.unsqueeze(-1) * edges.src['Wh_%s' % etype]
        else:
            message = gate.unsqueeze(-1) * edges.src['Wh_%s' % etype] + (1 - gate.unsqueeze(-1)) * self.baseline[etype][self.layer].unsqueeze(0)
        
        if self.return_gates:
            self.gate_storage[etype] = copy.deepcopy(gate.to('cpu').detach())
            self.gate_penalty_storage[etype] = copy.deepcopy(penalty_not_sum.to('cpu').detach())
            self.gate_score_storage[etype] = copy.deepcopy(gate_score.to('cpu').detach())
        return {'m': message} 
       
    def message_func_no_replace(self, edges):
        etype = edges._etype[1]  
        #self.msg_emb[etype] = edges.src['Wh_%s' % etype].to('cpu')
        return {'m': edges.src['Wh_%s' % etype]}
        
    def graphmask_forward(self, G, feat_dict, graphmask_mode, return_gates, no_base):
        self.no_base = no_base 
        self.return_gates = return_gates 
        self.penalty = []
        self.num_masked = 0
        self.num_of_edges = G.number_of_edges() 
        
        funcs = {}
        etypes_all = G.canonical_etypes
        
        for srctype, etype, dsttype in etypes_all: 
            Wh = self.weight[etype](feat_dict[srctype])  
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            
        for srctype, etype, dsttype in etypes_all:
            
            if graphmask_mode:  
                # replace the message!
                funcs[etype] = (self.gm_online, fn.mean('m', 'h'))
            else:
                # normal propagation!
                funcs[etype] = (self.message_func_no_replace, fn.mean('m', 'h'))
                
        G.multi_update_all(funcs, 'sum')
        
        
        if graphmask_mode: 
            self.penalty = torch.stack(self.penalty).reshape(-1,) 
            #penalty_mean = torch.mean(self.penalty)
            #penalty_relation_reg = torch.sum(torch.log(self.penalty) * self.penalty)
            #penalty = penalty_mean + 0.1 * penalty_relation_reg
            penalty = torch.mean(self.penalty) 
        else:
            penalty = 0 

        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}, penalty, self.num_masked
    
class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size, attention, proto, proto_num, sim_measure, bert_measure, agg_measure, num_walks, walk_mode, path_length, split, data_folder, exp_lambda, device,eval_dd_etypes):
        super(HeteroRGCN, self).__init__()
        self.raw_input_dims = {'compound': 768,
                                'disease': 768,
                                'gene/protein': 1280,
                                'pathway': 768,
                                'biological_process': 768,
                                'molecular_function': 768,
                                'cellular_component': 768,}
        self.input_proj = nn.ModuleDict()
        for ntype in G.ntypes:
            if ntype in self.raw_input_dims:
                key = self._ntype_key(ntype)
                self.input_proj[key] = nn.Sequential(
                                                    nn.Linear(self.raw_input_dims[ntype], in_size, bias=False),
                                                    nn.LayerNorm(in_size, eps=1e-5),
                                                    )

        if attention: 
            self.layer1 = AttHeteroRGCNLayer(in_size, hidden_size, G.etypes)
            self.layer2 = AttHeteroRGCNLayer(hidden_size, out_size, G.etypes)
        else:
            self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes) 
            self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes) 
        
        self.w_rels = nn.Parameter(torch.Tensor(len(G.canonical_etypes), out_size)) 
        nn.init.xavier_uniform_(self.w_rels, gain=nn.init.calculate_gain('relu'))
        rel2idx = dict(zip(G.canonical_etypes, list(range(len(G.canonical_etypes))))) 
               
        self.complex_pred = ComplExPredictor(n_hid = hidden_size, w_rels = self.w_rels, G = G, rel2idx = rel2idx, proto = proto, proto_num = proto_num, sim_measure = sim_measure, bert_measure = bert_measure, agg_measure = agg_measure, num_walks = num_walks, walk_mode = walk_mode, path_length = path_length, split = split, data_folder = data_folder, exp_lambda = exp_lambda, device = device, eval_dd_etypes = eval_dd_etypes)
        self.attention = attention 
        self.mlp_pred = FusionMLPPredictor(h_dim=out_size, hidden_dim=hidden_size, dropout=0.3,eval_dd_etypes = eval_dd_etypes, G= G, proto= proto, proto_num= proto_num, sim_measure= sim_measure, bert_measure= bert_measure, agg_measure= agg_measure, num_walks= num_walks, walk_mode= walk_mode, path_length= path_length, data_folder= data_folder, exp_lambda= exp_lambda, device = device)
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.etypes = G.etypes
        self.device = device

    def _ntype_key(self, ntype):
        """
        nn.ModuleDict key should not contain special characters such as '/'.
        """
        return str(ntype).replace('/', '__').replace('-', '_')

    def project_input_features(self, feat_dict):
        out = {}
        for ntype, x in feat_dict.items():
            key = self._ntype_key(ntype)
            if key in self.input_proj:
                out[ntype] = self.input_proj[key](x.float())
            else:
                out[ntype] = x.float()
        return out
    
    def forward_minibatch(self, pos_G, neg_G, blocks, G, mode = 'train', pretrain_mode = False):  
        input_dict = blocks[0].srcdata['inp'] 
        input_dict = self.project_input_features(input_dict)
        h_dict = self.layer1(blocks[0], input_dict) 
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h = self.layer2(blocks[1], h_dict) 
        
        scores, out_pos = self.complex_pred(pos_G, G, h, pretrain_mode, mode = mode + '_pos', block = blocks[1]) 
        scores_neg, out_neg = self.complex_pred(neg_G, G, h, pretrain_mode, mode = mode + '_neg', block = blocks[1])
        return scores, scores_neg, out_pos, out_neg
        
    def forward(self, G, neg_G, eval_pos_G = None, return_h = False, return_att = False, mode = 'train', pretrain_mode = False):
        with G.local_scope():
            raw_input_dict  = {ntype : G.nodes[ntype].data['inp'] for ntype in G.ntypes} 
            input_dict = self.project_input_features(raw_input_dict )
            if self.attention:
                h_dict, a_dict_l1 = self.layer1(G, input_dict, return_att)
                h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
                h, a_dict_l2 = self.layer2(G, h_dict, return_att)
            else:
                h_dict = self.layer1(G, input_dict) 
                h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
                h = self.layer2(G, h_dict) 

            if return_h: 
                return h

            if return_att: 
                return a_dict_l1, a_dict_l2

            if eval_pos_G is not None: #valid test
                scores, out_pos = self.mlp_pred(eval_pos_G, G, h, raw_input_dict,pretrain_mode, mode = mode + '_pos') 
                scores_neg, out_neg = self.mlp_pred(neg_G, G, h,raw_input_dict, pretrain_mode, mode = mode + '_neg')
                return scores, scores_neg, out_pos, out_neg
            elif eval_pos_G is None and mode != 'pred': #train, 
                scores, out_pos = self.mlp_pred(G, G, h, raw_input_dict,pretrain_mode, mode = mode + '_pos') 
                scores_neg, out_neg = self.mlp_pred(neg_G, G, h,raw_input_dict, pretrain_mode, mode = mode + '_neg')
                return scores, scores_neg, out_pos, out_neg
            if mode == 'pred':
                scores_pred, out_pred = self.mlp_pred(neg_G, G, h,raw_input_dict, pretrain_mode, mode = mode)
                return scores_pred, out_pred

    
    def graphmask_forward(self, G, pos_graph, neg_graph, graphmask_mode = False, return_gates = False, only_relation = None, no_base = False):                           
        with G.local_scope():
            input_dict = {ntype : G.nodes[ntype].data['inp'] for ntype in G.ntypes}
            h_dict_l1, penalty_l1, num_masked_l1 = self.layer1.graphmask_forward(G, input_dict, graphmask_mode, return_gates, no_base)
            h_dict = {k : F.leaky_relu(h) for k, h in h_dict_l1.items()}  
            h, penalty_l2, num_masked_l2 = self.layer2.graphmask_forward(G, h_dict, graphmask_mode, return_gates, no_base)         
            
            scores_pos, out_pos = self.pred(pos_graph, G, h, False, mode = 'train_pos', only_relation = only_relation) 
            scores_neg, out_neg = self.pred(neg_graph, G, h, False, mode = 'train_neg', only_relation = only_relation)
            return scores_pos, scores_neg, penalty_l1 + penalty_l2, [num_masked_l1, num_masked_l2]

    
    def enable_layer(self, layer, graphmask = True):
        print("Enabling layer "+str(layer))
        
        for name in self.etypes:
            if graphmask:
                for parameter in self.gates_all[name][layer].parameters():
                    parameter.requires_grad = True
                self.baselines_all[name][layer].requires_grad = True
            else:
                for parameter in self.gates_all[name].parameters():
                    parameter.requires_grad = True
    
    def count_layers(self):
        return 2
    
    def get_gates(self):
        return [self.layer1.gate_storage, self.layer2.gate_storage]
    
    def get_gates_scores(self):
        return [self.layer1.gate_score_storage, self.layer2.gate_score_storage]
    
    def get_gates_penalties(self):
        return [self.layer1.gate_penalty_storage, self.layer2.gate_penalty_storage]
    
    
    def add_graphmask_parameters(self, G, threshold = 0.5, remove_key_parts = False, use_top_k = False, k = 0.05, gate_hidden_size = 32):
        gates_all, baselines_all = {}, {}
        hidden_size = self.hidden_size 
        out_size = self.out_size 

        for name in G.etypes: 
            # for each relation type
            gates = []
            baselines = []

            vertex_embedding_dims = [hidden_size, out_size] 
            message_dims = [hidden_size, out_size]
            h_dims = message_dims 

            for v_dim, m_dim, h_dim in zip(vertex_embedding_dims, message_dims, h_dims):
                gate_input_shape = [m_dim, m_dim] 
                
                # different layers have different gates
                gate = torch.nn.Sequential(
                    MultipleInputsLayernormLinear(gate_input_shape, gate_hidden_size),
                    nn.ReLU(),
                    nn.Linear(gate_hidden_size, 1),
                    Squeezer(),
                    SoftConcrete(threshold, remove_key_parts, use_top_k, k)
                )

                gates.append(gate) 

                baseline = torch.FloatTensor(m_dim) 
                stdv = 1. / math.sqrt(m_dim)
                baseline.uniform_(-stdv, stdv)
                baseline = torch.nn.Parameter(baseline, requires_grad=True) 

                baselines.append(baseline) 

            gates = torch.nn.ModuleList(gates)
            gates_all[name] = gates 

            baselines = torch.nn.ParameterList(baselines)
            baselines_all[name] = baselines 

        self.gates_all = nn.ModuleDict(gates_all) 
        self.baselines_all = nn.ModuleDict(baselines_all) 

        # Initially we cannot update any parameters. They should be enabled layerwise
        for parameter in self.parameters():
            parameter.requires_grad = False
            
        self.layer1.add_graphmask_parameter(self.gates_all, self.baselines_all, 0)
        self.layer2.add_graphmask_parameter(self.gates_all, self.baselines_all, 1)