import numpy as np
import pandas as pd
import torch
import os
dirname = os.path.dirname(__file__)

class DataSplitter:     
    def __init__(self, kg_path): 
        self.kg, self.nodes, self.edges = self.load_kg(kg_path)
        self.edge_index = torch.LongTensor(self.edges.get(['x_index', 'y_index']).values.T)
        self.grouped_diseases = pd.read_csv(os.path.join(kg_path, 'dis_group.csv'))
        
    def load_kg(self, pth):
        kg = pd.read_csv(os.path.join(pth, 'kg.tsv'), sep = '\t', low_memory=False)
        nodes = pd.read_csv(os.path.join(pth, 'nodes.tsv'), sep = '\t', low_memory=False)
        edges = pd.read_csv(os.path.join(pth, 'edges.tsv'), sep = '\t', low_memory=False)
        return kg, nodes, edges
    
    
    def get_nodes_for_mondoid(self, code): 
        children = self.grouped_diseases[self.grouped_diseases['node_id'] == code]['children'].iloc[0]
        mondoids = [x.strip() for x in str(children).split(',') if x.strip()]
        mondoids.append(code)
        mondoids = list(set(mondoids))
        node_idx = self.nodes[self.nodes['node_id'].isin(mondoids)]['node_index'].values
        return node_idx
        
    def get_nodes_df_for_diod(self, code): 
        node_idx = self.get_nodes_for_doid(code)
        df = self.nodes.query('node_index in @node_idx')
        return df
    
    def get_one_hop_edge_group(self, nodes, mask_ratio = 0.1, add_cpd_dis=True):
        if add_cpd_dis: 
            x = self.edges.query('x_index in @nodes or y_index in @nodes').query('relation=="CPD_DIS_ass"')
            cpd_dis_edges = x.get(['x_index','y_index']).values.T
            print('cpd_dis_edges.shape: ', cpd_dis_edges.shape)
    
        from torch_geometric.utils import k_hop_subgraph
        subgraph_nodes, filtered_edge_index, node_map, edge_mask = k_hop_subgraph(list(nodes), 1, self.edge_index) #one hop 
        print('filtered_edge_index.shape: ', filtered_edge_index.shape)
        num_of_mask_edges = int(mask_ratio * filtered_edge_index.shape[1])
        print('num_of_mask_edges: ', num_of_mask_edges)
        sample_idx = np.random.choice(filtered_edge_index.shape[1], num_of_mask_edges, replace=False)
        sample_edges = filtered_edge_index[:, sample_idx].numpy()
        
        if add_cpd_dis:
            test_edges = np.concatenate([cpd_dis_edges, sample_edges], axis=1)
        else: 
            test_edges = sample_edges
        test_edges = np.unique(test_edges, axis=1)
        return test_edges 
            
    def get_edge_group(self, nodes, test_size = 0.05, add_cpd_dis=True):
        if add_cpd_dis: 
            x = self.edges.query('x_index in @nodes or y_index in @nodes').query('relation=="CPD_DIS_ass"')
            cpd_dis_edges = x.get(['x_index','y_index']).values.T
        
        if test_size > 1:
            print('using test size as static number of edges')
            test_num_edges = test_size + cpd_dis_edges.shape[1]
        else:
            test_num_edges = round(self.edge_index.shape[1]*test_size)

        if add_cpd_dis: 
            num_random_edges = test_num_edges - cpd_dis_edges.shape[1]
        else: 
            num_random_edges = test_num_edges


        from torch_geometric.utils import k_hop_subgraph
        subgraph_nodes, filtered_edge_index, node_map, edge_mask = k_hop_subgraph(list(nodes), 2, self.edge_index) 
        num_random_edges = min(num_random_edges, filtered_edge_index.shape[1])       
        sample_idx = np.random.choice(filtered_edge_index.shape[1], num_random_edges, replace=False)
        sample_edges = filtered_edge_index[:, sample_idx].numpy()
        
        if add_cpd_dis:
            test_edges = np.concatenate([cpd_dis_edges, sample_edges], axis=1)
        else: 
            test_edges = sample_edges
        test_edges = np.unique(test_edges, axis=1)
        return test_edges 
        
    def get_test_kg_for_disease(self, mondoid_code, test_size = 0.05, add_cpd_dis=True, one_hop = False, mask_ratio = 0.1): 
        disease_nodes = self.get_nodes_for_mondoid(mondoid_code)
        if one_hop:
            disease_edges = self.get_one_hop_edge_group(disease_nodes, mask_ratio = mask_ratio, add_cpd_dis=add_cpd_dis)
        else:
            disease_edges = self.get_edge_group(disease_nodes, test_size = test_size, add_cpd_dis=add_cpd_dis)
        disease_edges = pd.DataFrame(disease_edges.T, columns=['x_index','y_index'])
        select_kg = pd.merge(self.kg, disease_edges, 'right').drop_duplicates()
        return select_kg
    
    
