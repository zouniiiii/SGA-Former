# ------------------------ : new rwpse ----------------
from typing import Union, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import Data

import dgl


def filter_top_percent(adj, keep_rate):
    values = adj.flatten()  
    num_keep = int(keep_rate * values.numel())  

    if num_keep > 0:
        threshold = torch.topk(values, num_keep).values[-1]  
        mask = adj >= threshold  
    else:
        mask = torch.zeros_like(adj, dtype=torch.bool)  

    filtered_adj = torch.where(mask, adj, torch.tensor(0.0, dtype=adj.dtype))

    return filtered_adj

def get_3d_index_value(dense_tensor):
    non_zero_indices = torch.nonzero(torch.sum(abs(dense_tensor), dim=-1), as_tuple=False)
    values = dense_tensor[non_zero_indices[:, 0], non_zero_indices[:, 1], :]
    return non_zero_indices.T, values


def dense_to_sparse_tensor(dense_tensor):
    indices = torch.nonzero(dense_tensor)  # shape: (num_nonzero, 2)
    values = dense_tensor[indices[:, 0], indices[:, 1]]  # shape: (num_nonzero,)
    sparse_tensor = torch.sparse_coo_tensor(indices.t(), values, dense_tensor.size())
    return sparse_tensor


@torch.no_grad()
def add_sga(data,
                  low_rate = 1.0,
                  high_rate = 1.0,
                  filter_k=8,
                  add_identity=True,
                  **kwargs
                  ):
    device=data.edge_index.device
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    if edge_weight is None:
        edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32, device=device)

    # 构造稀疏邻接矩阵
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes), device=device)

    # 加上自环 (A + I)
    adj = adj.to_dense() + torch.eye(num_nodes, device=device)

    # 计算度数
    deg = adj.sum(dim=1) # shape [N]
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0

    # 构造对称归一化邻接矩阵: D^{-1/2} A D^{-1/2}
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    low_adj = D_inv_sqrt @ adj.to_dense() @ D_inv_sqrt
    high_adj = -low_adj

    low_pe_list= []
    high_pe_list = []
    if add_identity:
        low_pe_list.append(torch.eye(num_nodes, dtype=torch.float))
        high_pe_list.append(torch.eye(num_nodes, dtype=torch.float))

    low_out = low_adj
    high_out = high_adj
    low_pe_list.append(low_adj)
    high_pe_list.append(high_adj)

    if filter_k > 2:
        for j in range(1, filter_k):
            low_out = low_out @ low_adj
            high_out = high_out @ high_adj
            low_pe_list.append(low_out)
            high_pe_list.append(high_out)


    
    low_pe = torch.stack(low_pe_list, dim=-1) # n x n x k
    high_pe = torch.stack(high_pe_list, dim=-1) # n x n x k

    Spect_pe_val = []
    low_pe_idx, spect_pe_val = get_3d_index_value(low_pe)
    Spect_pe_val.append(spect_pe_val)
    high_pe_idx, spect_pe_val = get_3d_index_value(high_pe)
    Spect_pe_val.append(spect_pe_val)


    sorted_value = torch.sum(low_pe,dim=-1)
    low_filter_pe_dense = filter_top_percent(sorted_value,low_rate)
    low_filter_pe = dense_to_sparse_tensor(low_filter_pe_dense)

    sorted_value = torch.sum(high_pe,dim=-1)
    high_filter_pe_dense = filter_top_percent(sorted_value,high_rate)
    high_filter_pe = dense_to_sparse_tensor(high_filter_pe_dense)

    a_list = [low_filter_pe, high_filter_pe]
    
    g_id = []
    g_id.append(dgl.graph((low_pe_idx[0],low_pe_idx[1]),num_nodes=num_nodes))
    g_id.append(dgl.graph((high_pe_idx[0],high_pe_idx[1]),num_nodes=num_nodes))

    spect_devided_edge = []
    spect_devided_value = []
    for i, pe_ in enumerate(a_list):
       src,dst  = pe_.coalesce().indices()
       edge_id = g_id[i].edge_ids(src, dst)
       spect_i_edge = torch.cat([torch.tensor(src, dtype=torch.long).view(-1, 1), torch.tensor(dst, dtype=torch.long).view(-1, 1)], dim=1)
       spect_i_value = torch.sum(Spect_pe_val[i][edge_id],dim=-1).view(-1,1)

       spect_devided_edge.append(spect_i_edge.to(device))
       spect_devided_value.append(spect_i_value.to(device))


    data.spect_devided_edge = spect_devided_edge
    data.spect_devided_value = spect_devided_value

    return data

