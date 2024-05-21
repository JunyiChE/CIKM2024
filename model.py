import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import BertModel, BertConfig


class CrossAttentionFusionModel(nn.Module):
    def __init__(self, num_nodes, hidden_dims, out_feats, bert_pretrained_model_name):
        super().__init__()
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.config.is_decoder = True
        self.config.add_cross_attention = True
        self.bert_rep = BertModel.from_pretrained(bert_pretrained_model_name, self.config)
        self.gcn_model = GCN(num_nodes=num_nodes, hidden_dims=hidden_dims, out_feats=out_feats)
        self.fc = nn.Linear(hidden_dims, out_feats)

    def forward(self, input_ids, attention_mask, uid, uid_to_index, data):
        x = self.gcn_model.forward(data)
        h = self.gcn_model.get_credibility_representation(uid, uid_to_index, x)
        outputs = self.bert_rep(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=h)
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)
        return logits

    def prediction(self, fused_representation):
        return self.fc(fused_representation)


class GCN(nn.Module):
    def __init__(self, num_nodes, hidden_dims, out_feats):
        super(GCN, self).__init__()

        self.num_nodes = num_nodes
        self.hidden_dims = hidden_dims

        self.conv1 = GCNConv(hidden_dims, hidden_dims)
        self.conv2 = GCNConv(hidden_dims, hidden_dims)
        self.nn = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU()
        )
        self.embeddings = nn.Parameter(torch.empty(num_nodes, hidden_dims))
        self.temperature = 0.5
        self.epsilon = 1e-7
        self.init_embeddings()

    def init_embeddings(self):
        nn.init.normal_(self.embeddings)
        with open('data/Twitter16/graph.pkl', 'rb') as f:
            G = pickle.load(f)
        # Initial based on text embedding
        node_FC_features = torch.Tensor([G.nodes[node]['FC'] for node in G.nodes]).unsqueeze(1)
        self.embeddings.data += node_FC_features

    def get_credibility_representation(self, uids, uid_to_index, x):
        node_indices = [uid_to_index[uid] for uid in uids]
        embeddings = [x[node_index] for node_index in node_indices]
        embeddings_tensor = torch.stack(embeddings).unsqueeze(1)
        return embeddings_tensor

    def forward(self, data):
        x_initial, edge_index, edge_weight = self.embeddings, self.generate_edge_index(self.embeddings)\
                                # data.edge_index

        x1 = F.relu(self.conv1(x_initial, edge_index, edge_weight=edge_weight))
        x2 = self.conv2(x1, edge_index,edge_weight=edge_weight)

        x_mean = torch.mean(torch.stack([x1, x2], dim=0), dim=0)

        x_final = x_mean

        return x_final

    def generate_edge_index(self, x):
        z = self.nn(x)
        pi = torch.sigmoid(torch.matmul(z, z.t()))
        edge_index = []
        edge_weight_list = []
        for u in range(self.num_nodes):
            for v in range(u + 1, self.num_nodes):
                pi_uv = pi[u, v]
                logit = (1 / self.temperature) * (torch.log(pi_uv) - torch.log(1 - pi_uv) +
                                                  torch.log(self.epsilon) - torch.log(1 - self.epsilon))
                edge_weight = torch.sigmoid(logit)
                edge_index.append([u, v])
                edge_weight_list.append(edge_weight.item())
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_weight_list = torch.tensor(edge_weight_list, dtype=torch.float32)
        return edge_index, edge_weight_list