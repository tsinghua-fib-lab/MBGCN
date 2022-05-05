import os
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sparse
import torch.nn.functional as F

class model_base(nn.Module):
    def __init__(self, flags_obj, trainset, device):
        super().__init__()
        self.embed_size = flags_obj.embedding_size
        self.L2_norm = flags_obj.L2_norm
        self.device = device
        self.user_num = trainset.user_num
        self.item_num = trainset.item_num
        if flags_obj.create_embeddings=='True':
            self.item_embedding = nn.Parameter(torch.FloatTensor(self.item_num,self.embed_size))
            nn.init.xavier_normal_(self.item_embedding)
            self.user_embedding = nn.Parameter(torch.FloatTensor(self.user_num,self.embed_size))
            nn.init.xavier_normal_(self.user_embedding)
        else:
            load_data = torch.load(os.path.join(flags_obj.pretrain_path, 'model.pkl'), map_location='cpu')
            if flags_obj.pretrain_frozen==False:
                self.item_embedding = nn.Parameter(F.normalize(load_data['item_embedding']).to(self.device))
                self.user_embedding = nn.Parameter(F.normalize(load_data['user_embedding']).to(self.device))
            else:
                self.item_embedding = F.normalize(load_data['item_embedding']).to(self.device)
                self.user_embedding = F.normalize(load_data['user_embedding']).to(self.device)

    def propagate(self,*args,**kwargs):
        '''
        raw embeddings -> embeddings for predicting
        return (user's,POI's)
        '''
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        '''
        embeddings of targets for predicting -> scores
        return scores
        '''
        raise NotImplementedError

    def regularize(self, user_embeddings, item_embeddings):
        '''
        embeddings of targets for predicting -> extra loss(default: L2 loss...)
        '''
        return self.L2_norm*((user_embeddings**2).sum()+(item_embeddings**2).sum())

    def forward(self, users, items):
        users_feature, item_feature = self.propagate()
        item_embeddings = item_feature[items]
        user_embeddings = users_feature[users].expand(-1,items.shape[1],-1)
        pred = self.predict(user_embeddings, item_embeddings)
        L2_loss = self.regularize(user_embeddings, item_embeddings)
        return pred, L2_loss

    def evaluate(self, users):
        '''
        just for testing, compute scores of all POIs for `users` by `propagate_result`
        '''
        raise NotImplementedError

class MF(model_base):
    def __init__(self, flags_obj, trainset, device):
        super().__init__(flags_obj, trainset, device)

    def propagate(self, task='train'):
        return self.user_embedding, self.item_embedding

    def predict(self, user_embedding, item_embedding):
        return torch.sum(user_embedding*item_embedding,dim=2)

    def evaluate(self, propagate_result, users):
        users_feature, item_feature = propagate_result
        user_feature = users_feature[users]
        scores = torch.mm(user_feature, item_feature.t())

        return scores

class MBGCN(model_base):
    def __init__(self, flags_obj, trainset, device):
        super().__init__(flags_obj, trainset, device)
        self.relation_dict = trainset.relation_dict
        self.mgnn_weight = flags_obj.mgnn_weight
        self.item_graph = trainset.item_graph
        self.train_matrix = trainset.train_matrix.to(self.device)
        self.relation = trainset.relation
        self.lamb = flags_obj.lamb
        self.item_graph_degree = trainset.item_graph_degree
        self.user_behaviour_degree = trainset.user_behaviour_degree.to(self.device)
        self.message_drop = nn.Dropout(p=flags_obj.message_dropout)
        self.train_node_drop = nn.Dropout(p=flags_obj.node_dropout)
        self.node_drop = nn.ModuleList([nn.Dropout(p=flags_obj.node_dropout) for _ in self.relation_dict])
        self.__to_gpu()
        self.__param_init()

    def __to_gpu(self):
        for key in self.relation_dict:
            self.relation_dict[key] = self.relation_dict[key].to(self.device)
        for key in self.item_graph:
            self.item_graph[key] = self.item_graph[key].to(self.device)
        for key in self.item_graph_degree:
            self.item_graph_degree[key] = self.item_graph_degree[key].to(self.device)

    def __decode_weight(self):
        weight = nn.softmax(self.mgnn_weight).unsqueeze(-1)
        total_weight = torch.mm(self.user_behaviour_degree, weight)
        self.user_behaviour_weight = self.user_behaviour_degree.float() / (total_weight + 1e-8)

    def __param_init(self):
        self.mgnn_weight = nn.Parameter(torch.FloatTensor(self.mgnn_weight))
        self.item_behaviour_W = nn.ParameterList([nn.Parameter(torch.FloatTensor(self.embed_size * 2, self.embed_size * 2)) for _ in self.mgnn_weight])
        for param in self.item_behaviour_W:
            nn.init.xavier_normal_(param)
        self.item_propagate_W = nn.ParameterList([nn.Parameter(torch.FloatTensor(self.embed_size, self.embed_size)) for _ in self.mgnn_weight])
        for param in self.item_propagate_W:
            nn.init.xavier_normal_(param)
        self.W = nn.Parameter(torch.FloatTensor(self.embed_size, self.embed_size))
        nn.init.xavier_normal_(self.W)

    def forward(self, user, item):
        # node dropout on train matrix
        indices = self.train_matrix._indices()
        values = self.train_matrix._values()
        values = self.train_node_drop(values)
        train_matrix = torch.sparse.FloatTensor(indices, values, size = self.train_matrix.shape)

        weight = self.mgnn_weight.unsqueeze(-1)
        total_weight = torch.mm(self.user_behaviour_degree, weight)
        user_behaviour_weight = self.user_behaviour_degree * self.mgnn_weight.unsqueeze(0) / (total_weight + 1e-8)

        for i, key in enumerate(self.relation_dict):
            # node dropout
            indices = self.relation_dict[key]._indices()
            values = self.relation_dict[key]._values()
            values = self.node_drop[i](values)
            tmp_relation_matrix = torch.sparse.FloatTensor(indices, values, size = self.relation_dict[key].shape)

            tmp_item_propagation = torch.mm(torch.mm(self.item_graph[key].float(), self.item_embedding) / (self.item_graph_degree[key] + 1e-8), self.item_propagate_W[i])
            tmp_item_propagation = torch.cat((self.item_embedding, tmp_item_propagation), dim=1)
            tmp_item_embedding = tmp_item_propagation[item]
            tmp_user_neighbour = torch.mm(tmp_relation_matrix, self.item_embedding) / (self.user_behaviour_degree[:,i].unsqueeze(-1) + 1e-8)
            tmp_user_item_neighbour_p = torch.mm(tmp_relation_matrix, tmp_item_propagation) / (self.user_behaviour_degree[:,i].unsqueeze(-1) + 1e-8)
            if i==0:
                user_feature = user_behaviour_weight[:,i].unsqueeze(-1) * tmp_user_neighbour
                tbehaviour_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behaviour_W[i])
                tuser_tbehaviour_item_projection = tbehaviour_item_projection[user].expand(-1,item.shape[1],-1)
                score2 = torch.sum(tuser_tbehaviour_item_projection * tmp_item_embedding,dim=2)
            else:
                user_feature += user_behaviour_weight[:,i].unsqueeze(-1) * tmp_user_neighbour
                tbehaviour_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behaviour_W[i])
                tuser_tbehaviour_item_projection = tbehaviour_item_projection[user].expand(-1,item.shape[1],-1)
                score2 += torch.sum(tuser_tbehaviour_item_projection * tmp_item_embedding,dim=2)

        score2 = score2 / len(self.mgnn_weight)

        item_feature = torch.mm(train_matrix.t(), self.user_embedding)

        user_feature = torch.mm(user_feature, self.W)
        item_feature = torch.mm(item_feature, self.W)

        user_feature = torch.cat((self.user_embedding,user_feature),dim=1)
        item_feature = torch.cat((self.item_embedding,item_feature),dim=1)

        # message dropout
        user_feature = self.message_drop(user_feature)
        item_feature = self.message_drop(item_feature)

        tmp_user_feature = user_feature[user].expand(-1,item.shape[1],-1)
        tmp_item_feature = item_feature[item]
        score1 = torch.sum(tmp_user_feature * tmp_item_feature,dim=2)

        scores = score1 + self.lamb * score2

        L2_loss = self.regularize(tmp_user_feature, tmp_item_feature)

        return scores, L2_loss

    def evaluate(self, user):

        weight = self.mgnn_weight.unsqueeze(-1)
        total_weight = torch.mm(self.user_behaviour_degree, weight)
        user_behaviour_weight = self.user_behaviour_degree * self.mgnn_weight.unsqueeze(0) / (total_weight + 1e-8)

        for i, key in enumerate(self.relation_dict):

            tmp_item_propagation = torch.mm(torch.mm(self.item_graph[key].float(), self.item_embedding) / (self.item_graph_degree[key] + 1e-8), self.item_propagate_W[i])
            tmp_item_propagation = torch.cat((self.item_embedding, tmp_item_propagation), dim=1)

            tmp_user_neighbour = torch.mm(self.relation_dict[key], self.item_embedding) / (self.user_behaviour_degree[:,i].unsqueeze(-1) + 1e-8)
            tmp_user_item_neighbour_p = torch.mm(self.relation_dict[key], tmp_item_propagation) / (self.user_behaviour_degree[:,i].unsqueeze(-1) + 1e-8)
            if i==0:
                user_feature = user_behaviour_weight[:,i].unsqueeze(-1) * tmp_user_neighbour
                tbehaviour_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behaviour_W[i])
                tuser_tbehaviour_item_projection = tbehaviour_item_projection[user]
                score2 = torch.mm(tuser_tbehaviour_item_projection, tmp_item_propagation.t())
            else:
                user_feature += user_behaviour_weight[:,i].unsqueeze(-1) * tmp_user_neighbour
                tbehaviour_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behaviour_W[i])
                tuser_tbehaviour_item_projection = tbehaviour_item_projection[user]
                score2 += torch.mm(tuser_tbehaviour_item_projection, tmp_item_propagation.t())

        score2 = score2 / len(self.mgnn_weight)

        item_feature = torch.mm(self.train_matrix.t(), self.user_embedding)

        user_feature = torch.mm(user_feature, self.W)
        item_feature = torch.mm(item_feature, self.W)
        
        user_feature = torch.cat((self.user_embedding,user_feature),dim=1)
        item_feature = torch.cat((self.item_embedding,item_feature),dim=1)

        tmp_user_feature = user_feature[user]
        score1 = torch.mm(tmp_user_feature, item_feature.t())

        scores = score1 + self.lamb * score2

        return scores
