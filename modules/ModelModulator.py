import torch
from torch import nn
from utils import config
from BaseRecModel import Decoder

# 参考TaNP 或者 参考L2B的方法
class ModelModulator(nn.Module):
    def __init__(self, task_dim, h_dims) -> None:
        super(ModelModulator, self).__init__()
        self.task_dim = task_dim
        self.h1_dim = h_dims[0]
        self.h2_dim = h_dims[1]
        self.h3_dim = h_dims[2]

        self.film_layer_1_beta = nn.Linear(self.task_dim, self.h1_dim, bias=False)
        self.film_layer_1_gamma = nn.Linear(self.task_dim, self.h1_dim, bias=False)
        self.film_layer_2_beta = nn.Linear(self.task_dim, self.h2_dim, bias=False)
        self.film_layer_2_gamma = nn.Linear(self.task_dim, self.h2_dim, bias=False)
        self.film_layer_3_beta = nn.Linear(self.task_dim, self.h3_dim, bias=False)
        self.film_layer_3_gamma = nn.Linear(self.task_dim, self.h3_dim, bias=False)

        self.tanh_layer = nn.Tanh()
        
    def get_beta_gamma(self, layer_num, task_emb):
        if layer_num == 1:
            beta = self.tanh_layer(self.film_layer_1_beta(task_emb))
            gamma = self.tanh_layer(self.film_layer_1_gamma(task_emb))
            return beta, gamma
        elif layer_num == 2:
            beta = self.tanh_layer(self.film_layer_2_beta(task_emb))
            gamma = self.tanh_layer(self.film_layer_2_gamma(task_emb))
            return beta, gamma
        elif layer_num == 3:
            beta = self.tanh_layer(self.film_layer_3_beta(task_emb))
            gamma = self.tanh_layer(self.film_layer_3_gamma(task_emb))
            return beta, gamma
        else:
            assert(False)





class Modulated_Decoder(Decoder):
    def __init__(self, x_dim, task_dim, h_dims, y_dim, dropout_rate):
        super(Modulated_Decoder, self).__init__(x_dim, h_dims, y_dim, dropout_rate)
        self.task_dim = task_dim
        # self.decoder = Decoder(x_dim, h_dims, y_dim, dropout_rate)
        self.modulator = ModelModulator(self.task_dim, h_dims)

    def forward(self, x1, x2, task):
        inputs = torch.cat([x1, x2])
        hidden_1 = self.hidden_layer_1(inputs)
        beta_1, gamma_1 = self.modulator.get_beta_gamma(layer_num=1, task_emb=task)
        hidden_1 = torch.mul(hidden_1, gamma_1) + beta_1
        hidden_1 = self.dropout(hidden_1)
        hidden_2 = self.relu_layer(hidden_1)

        hidden_2 = self.hidden_layer_2(hidden_2)
        beta_2, gamma_2 = self.modulator.get_beta_gamma(layer_num=2, task_emb=task)
        hidden_2 = torch.mul(hidden_2, gamma_2) + beta_2
        hidden_2 = self.dropout(hidden_2)
        hidden_3 = self.relu_layer(hidden_2)

        hidden_3 = self.hidden_layer_3(hidden_3)
        beta_3, gamma_3 = self.modulator.get_beta_gamma(layer_num=3, task_emb=task)
        hidden_final = torch.mul(hidden_3, gamma_3) + beta_3
        hidden_final = self.dropout(hidden_final)
        hidden_final = self.relu_layer(hidden_final)

        y_pred = self.final_projection(hidden_final)
        return y_pred

    # def inner_forward(self, x1, x2, task, weights):
    #     return 1
    
    # def get_decoder_parameters(self):
    #     return {
    #         **{'hidden_layer_1.'+k:v for k,v in self.hidden_layer_1.named_parameters()},
    #         **{'hidden_layer_2.'+k:v for k,v in self.hidden_layer_2.named_parameters()},
    #         **{'hidden_layer_3.'+k:v for k,v in self.hidden_layer_3.named_parameters()},
    #         **{'final_projection.'+k:v for k,v in self.final_projection.named_parameters()}
    #     }

