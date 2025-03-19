import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss
import os


class LightSGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LightSGCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['LightSGCL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.temp = float(args['-tau'])
        self.model = LightGCN_Encoder(self.data, self.emb_size,self.n_layers)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb= model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss= self.cal_cl_loss([user_idx, pos_idx], rec_user_emb, rec_item_emb)
                cl_loss = self.cl_rate * cl_loss
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx, user_view1,item_view1):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_sslLoss= self.contrastLoss_ln(user_view1,u_idx, self.temp,self.eps)
        item_sslLoss= self.contrastLoss_ln(item_view1,i_idx, self.temp,self.eps)
        sslLoss = user_sslLoss + item_sslLoss
        return sslLoss

    def contrastLoss_ln(self,embeds, nodes, temp,scale_factor):
        anchor_embeds = embeds[nodes]
        nosie = self.construct_noise_matrix(anchor_embeds,scale_factor).cuda()

        power_embeds = anchor_embeds + nosie

        anchor_embeds_norm = F.normalize(anchor_embeds + 1e-8, p=2)
        power_embeds_norm = F.normalize(power_embeds + 1e-8, p=2)

        nume = torch.exp(torch.sum(anchor_embeds_norm * power_embeds_norm, dim=-1) / temp)
        deno = torch.exp(anchor_embeds_norm @ power_embeds_norm.T / temp).sum(-1) + 1e-8
        sslloss = -torch.log(nume / deno).mean()

        return sslloss

    def construct_noise_matrix(self,A,eps=0.005):
        N = torch.randn_like(A)
        Q, R = torch.linalg.qr(A)
        N_orthogonal = N - Q @ Q.T @ N
        N_orthogonal *= eps
        return N_orthogonal

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()
        content = {
            'model':self.model,
        }

        model_dir = "../Models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(content, os.path.join(model_dir, 'lr_yelp.mod'))


    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class LightGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size,n_layers):
        super(LightGCN_Encoder, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings













