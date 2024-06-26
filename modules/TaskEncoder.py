import torch
from torch import nn
import torch.nn.functional as F


class TaskEncoderSSL(nn.Module):
    def __init__(self, task_dim, num_class, emb_size, feat_num, max_shot, clusters_k, temperature, device) -> None:
        super().__init__()
        # 超参数
        self.task_dim = task_dim
        self.num_class = num_class
        self.emb_size = emb_size
        self.feat_num = feat_num
        self.max_shot = max_shot
        self.device = device
        self.clusters_k = clusters_k
        self.temperature = temperature

        self.predictor = MemoryUnit(self.clusters_k, self.task_dim, self.temperature)

        # 部分变量参数
        self.e_size = self.feat_num*self.emb_size
        self.s_size = self.emb_size
        self.v_size = int(self.emb_size / 2)

        # layers
        # statistic
        self.sta_inter_layer = nn.Linear(2*self.feat_num*self.emb_size+1, self.s_size)
        self.sta_inter_layer2 = nn.Linear(self.v_size*2+2, self.task_dim)
        # self.final_emb = nn.Linear(2*self.task_dim, self.task_dim)
        self.relu_layer = nn.ReLU()

        # encoder
        self.en_layer0 = nn.Sequential(
            nn.Linear(self.e_size, self.e_size),
            nn.ReLU(),
            nn.Linear(self.e_size, self.e_size)
        )

        self.en_layer = nn.Sequential(
            nn.Linear(self.emb_size, int(self.emb_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.emb_size/2), int(self.emb_size/2))
        )
    
    # Compute element-wise sample mean, var., and set cardinality
    # x:(n, es)
    def _statistic_pooling(self, x):
        var = torch.var(x, dim=0)
        mean = torch.mean(x, dim=0)
        if x.shape[0] == 1:
            var = torch.zeros_like(x)
        return var.reshape(-1), mean
    
    # 输入：x:(batch_size, feat_num*emb_size), y:(batch_size, 1)
    def _pre_forward(self, x, y):
        # encoder 1
        x = self.en_layer0(x)
        
        # statistic 1
        s = []; Nc_list = []; mean_list = []
        for c in range(self.num_class):
            if c not in y:
                continue
            c_mask = torch.eq(y, c) # cuda
            c_x = x[c_mask] # 不可用masked_select
            c_y = torch.masked_select(y, c_mask)
            N_c = torch.tensor([(len(c_y)-1.)/(self.max_shot-1.)], device=self.device)
            Nc_list.append(N_c)
            c_var, c_mean = self._statistic_pooling(c_x) # 这里cvar出现的nan
            s.append(torch.cat([c_var, c_mean, N_c], 0))
            mean_list.append(c_mean) # c_y, tensor([feat_num*emb_size])
        
        s = torch.stack(s, 0) # n_c, 2*f*es+1
        s = self.relu_layer(self.sta_inter_layer(s)) # interaction

        # encoder 2
        v = self.en_layer(s)

        # statistic 2
        v_var, v_mean = self._statistic_pooling(v)
        v_N = torch.mean(torch.tensor(Nc_list))
        
        len_mean_list = len(mean_list)
        mean_tensor = torch.cat(mean_list, dim=0).reshape(len_mean_list, -1) # list of tensor -> tensor
        mean_norm = mean_tensor / torch.max(torch.norm(mean_tensor, p=2, dim=-1, keepdim=True), torch.tensor(1e-7).to(self.device))
        cos_sim = torch.einsum('ni,mi->nm', mean_norm, mean_norm)
        sim_sum = torch.tensor([0.], device=self.device)
        for i in range(len_mean_list):
            for j in range(i+1, len_mean_list):
                sim_sum += (j-i)*cos_sim[i][j]
        
        v = torch.cat([v_var,v_mean,torch.tensor([v_N], device=self.device),sim_sum])
        task_emb = self.relu_layer(self.sta_inter_layer2(v))
        return task_emb

    def forward_on_tar(self, x, y):
        emb_online_ori = self._pre_forward(x, y)
        with torch.no_grad():
            emb_target = emb_online_ori.clone()
            emb_target.detach()
        emb_online = self.predictor(emb_online_ori)
        return emb_online, emb_target
    
    def forward(self, x, y):
        return self.predictor(self._pre_forward(x, y))
    
    # def get_taskmem(self, task_emb):
    #     task_mem_emb = self.task_mem(task_emb)
    #     return torch.cat([task_emb, task_mem_emb])

class TaskEncoderSSLMem(nn.Module):
    def __init__(self, task_dim, num_class, emb_size, feat_num, max_shot, clusters_k, temperature, device, config=False) -> None:
        super().__init__()
        # 超参数
        self.task_dim = task_dim
        self.num_class = num_class
        self.emb_size = emb_size
        self.feat_num = feat_num
        self.max_shot = max_shot
        self.device = device
        self.clusters_k = clusters_k
        self.temperature = temperature
        self.task_mem = MemoryUnit(self.clusters_k, self.task_dim, self.temperature)
        self.aug1 = config['aug1']
        self.aug2 = config['aug2']
        self.prob = config['prob']

        self.predictor = nn.Linear(self.task_dim, self.task_dim)
        
        # 部分变量参数
        self.e_size = self.feat_num*self.emb_size
        self.s_size = self.emb_size
        self.v_size = int(self.emb_size / 2)

        # layers
        # statistic
        self.sta_inter_layer = nn.Linear(2*self.feat_num*self.emb_size+1, self.s_size)
        self.sta_inter_layer2 = nn.Linear(self.v_size*2+2, self.task_dim)
        # self.final_emb = nn.Linear(2*self.task_dim, self.task_dim)
        self.relu_layer = nn.ReLU()

        # encoder
        self.en_layer0 = nn.Sequential(
            nn.Linear(self.e_size, self.e_size),
            nn.ReLU(),
            nn.Linear(self.e_size, self.e_size)
        )

        self.en_layer = nn.Sequential(
            nn.Linear(self.emb_size, int(self.emb_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.emb_size/2), int(self.emb_size/2))
        )
    
    # Compute element-wise sample mean, var., and set cardinality
    # x:(n, es)
    def _statistic_pooling(self, x):
        var = torch.var(x, dim=0)
        mean = torch.mean(x, dim=0)
        if x.shape[0] == 1:
            var = torch.zeros_like(x)
        return var.reshape(-1), mean
    
    # 输入：x:(batch_size, feat_num*emb_size), y:(batch_size, 1)
    def _pre_forward(self, x, y, aug1=False, aug2=False, prob=0.3, test=False):
        # augmentation one
        if aug1 and not test:
            row_mask = (torch.rand(x.size(0)) < prob).nonzero().squeeze(1) # select the sample
            if len(row_mask) >= 1:
                x = x[row_mask]
                y = y[row_mask]

        # encoder 1
        x = self.en_layer0(x)
        
        # statistic 1
        s = []; Nc_list = []; mean_list = []
        for c in range(self.num_class):
            if c not in y:
                continue
            c_mask = torch.eq(y, c) # cuda

            # augmentation two
            if aug2 and not test:
                true_indices = torch.nonzero(c_mask).squeeze(1)
                if len(true_indices) > 1:
                    select_idx = torch.randint(0, len(true_indices), (1,))
                    c_mask[:] = 0
                    c_mask[true_indices[select_idx]] = 1
            
            c_x = x[c_mask] # 不可用masked_select
            c_y = torch.masked_select(y, c_mask)
            N_c = torch.tensor([(len(c_y)-1.)/(self.max_shot-1.)], device=self.device)
            Nc_list.append(N_c)
            c_var, c_mean = self._statistic_pooling(c_x) # 这里cvar出现的nan
            s.append(torch.cat([c_var, c_mean, N_c], 0))
            mean_list.append(c_mean) # c_y, tensor([feat_num*emb_size])
        
        s = torch.stack(s, 0) # n_c, 2*f*es+1
        s = self.relu_layer(self.sta_inter_layer(s)) # interaction

        # encoder 2
        v = self.en_layer(s)

        # statistic 2
        v_var, v_mean = self._statistic_pooling(v)
        v_N = torch.mean(torch.tensor(Nc_list))
        
        len_mean_list = len(mean_list)
        mean_tensor = torch.cat(mean_list, dim=0).reshape(len_mean_list, -1) # list of tensor -> tensor
        mean_norm = mean_tensor / torch.max(torch.norm(mean_tensor, p=2, dim=-1, keepdim=True), torch.tensor(1e-7).to(self.device))
        cos_sim = torch.einsum('ni,mi->nm', mean_norm, mean_norm)
        sim_sum = torch.tensor([0.], device=self.device)
        for i in range(len_mean_list):
            for j in range(i+1, len_mean_list):
                sim_sum += (j-i)*cos_sim[i][j]
        
        v = torch.cat([v_var,v_mean,torch.tensor([v_N], device=self.device),sim_sum])
        task_emb = self.relu_layer(self.sta_inter_layer2(v))
        return task_emb

    def forward_on_tar(self, x, y): # all items
        emb_online_ori = self._pre_forward(x, y)
        with torch.no_grad():
            emb_target = emb_online_ori.clone()
            emb_target.detach()
            # emb_target = F.dropout(emb_target, 0.3) # 不应该有drop
        emb_online = self.predictor(emb_online_ori)
        return emb_online, emb_target
    
    def forward(self, x, y):
        return self.predictor(self._pre_forward(x, y))
        # return self._pre_forward(x, y)
    
    def forward_nossl(self, x, y):
        return self._pre_forward(x, y)
    
    def gen_ssl_on_tar(self, x, y): # droped items
        emb_online_ori = self._pre_forward(x, y, self.aug1, self.aug2, self.prob)
        with torch.no_grad():
            emb_target = emb_online_ori.clone()
            emb_target.detach()
            # emb_target = F.dropout(emb_target, 0.3) # 不应该有drop
        emb_online = self.predictor(emb_online_ori)
        return emb_online, emb_target
    
    def get_taskmem(self, task_emb):
        task_mem_emb = self.task_mem(task_emb)
        return torch.cat([task_emb, task_mem_emb])

class TaskMeanMemEncoderSSL(nn.Module):
    def __init__(self, task_dim, num_class, emb_size, feat_num, max_shot, clusters_k, temperature, device, config=False) -> None:
        super().__init__()
        # 超参数
        self.task_dim = task_dim
        self.emb_size = emb_size
        self.device = device
        self.feat_num = feat_num
        self.clusters_k = clusters_k
        self.temperature = temperature
        self.task_mem = MemoryUnit(self.clusters_k, self.task_dim, self.temperature)
        self.aug1 = config['aug1']
        self.aug2 = config['aug2']
        self.prob = config['prob']

        # 部分变量参数
        self.encoder_layer = nn.Linear(self.feat_num*self.emb_size, self.task_dim)
        self.relu_layer = nn.ReLU()
        self.predictor = nn.Linear(self.task_dim, self.task_dim)

    # 输入：x:(batch_size, feat_num*emb_size), y:(batch_size, 1)
    def _pre_forward(self, x, y, aug1=False, aug2=False, prob=0.3, test=False): # no mem
        if aug1 and not test:
            row_mask = (torch.rand(x.size(0)) < prob).nonzero().squeeze(1) # select the sample
            if len(row_mask) >= 1:
                x = x[row_mask]
                y = y[row_mask]
        task_emb = torch.mean(self.relu_layer(self.encoder_layer(x)), dim=0)
        # task_mem_emb = self.task_mem(task_emb)
        return task_emb
    
    def forward_on_tar(self, x, y): # all items
        emb_online_ori = self._pre_forward(x, y)
        with torch.no_grad():
            emb_target = emb_online_ori.clone()
            emb_target.detach()
        emb_online = self.predictor(emb_online_ori)
        return emb_online, emb_target
    
    def forward(self, x, y):
        return self.predictor(self._pre_forward(x, y))
    
    def forward_nossl(self, x, y):
        return self._pre_forward(x, y)
    
    def gen_ssl_on_tar(self, x, y): # droped items
        emb_online_ori = self._pre_forward(x, y, self.aug1, self.aug2, self.prob)
        with torch.no_grad():
            emb_target = emb_online_ori.clone()
            emb_target.detach()
            # emb_target = F.dropout(emb_target, 0.3) # 不应该有drop
        emb_online = self.predictor(emb_online_ori)
        return emb_online, emb_target
    
    def get_taskmem(self, task_emb):
        task_mem_emb = self.task_mem(task_emb)
        return torch.cat([task_emb, task_mem_emb])


class TaskEncoderSSL98(nn.Module):
    def __init__(self, task_dim, num_class, emb_size, feat_num, max_shot, device):
        super().__init__()
        self.task_dim = task_dim
        self.num_class = num_class
        self.emb_size = emb_size
        self.feat_num = feat_num
        self.max_shot = max_shot
        self.device = device

        self.emb_y = torch.nn.Embedding(num_embeddings=self.num_class, embedding_dim=self.emb_size)
        self.in_dim = 2 * (self.feat_num+1) * self.emb_size # user feat num + item feat num + y emb
        self.hid_dim = int(self.in_dim/2)
        
        # 与taskencoder保持相当的size（taskencoder也是两个mlp)
        self.en_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.task_dim)
            )

        # projection head, for cl
        self.predictor = nn.Linear(self.task_dim, self.task_dim)

    def _statistic_pooling(self, x):
        var = torch.var(x, dim=0)
        mean = torch.mean(x, dim=0)
        return var.reshape(-1), mean
    
    def _pre_forward(self, x, y):
        y_emb = self.emb_y(y)
        inputs = torch.cat((x, y_emb), 1)
        inputs_var, inputs_mean = self._statistic_pooling(inputs)
        return self.en_layer(torch.cat([inputs_var, inputs_mean]))

    # generate online and target view
    def forward_on_tar(self, x, y):
        emb_online_ori = self._pre_forward(x, y)
        with torch.no_grad():
            emb_target = emb_online_ori.clone()
            emb_target.detach()
            emb_target = F.dropout(emb_target, 0.3)
        emb_online = self.predictor(emb_online_ori)
        return emb_online, emb_target

    def forward(self, x, y):
        return self.predictor(self._pre_forward(x, y))

class TaskEncoderSSLMem98(nn.Module):
    def __init__(self, task_dim, num_class, emb_size, feat_num, max_shot, clusters_k, temperature, device):
        super().__init__()
        self.task_dim = task_dim
        self.num_class = num_class
        self.emb_size = emb_size
        self.feat_num = feat_num
        self.max_shot = max_shot
        self.device = device
        self.clusters_k = clusters_k
        self.temperature = temperature
        self.task_mem = MemoryUnit(self.clusters_k, self.task_dim, self.temperature)

        self.emb_y = torch.nn.Embedding(num_embeddings=self.num_class, embedding_dim=self.emb_size)
        self.in_dim = 2 * (self.feat_num+1) * self.emb_size # user feat num + item feat num + y emb
        self.hid_dim = int(self.in_dim/2)
        
        # 与taskencoder保持相当的size（taskencoder也是两个mlp)
        self.en_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.task_dim)
            )

        # projection head, for cl
        self.predictor = nn.Linear(self.task_dim, self.task_dim)

    def _statistic_pooling(self, x):
        var = torch.var(x, dim=0)
        mean = torch.mean(x, dim=0)
        return var.reshape(-1), mean
    
    def _pre_forward(self, x, y):
        y_emb = self.emb_y(y)
        inputs = torch.cat((x, y_emb), 1)
        inputs_var, inputs_mean = self._statistic_pooling(inputs)
        return self.en_layer(torch.cat([inputs_var, inputs_mean]))

    # generate online and target view
    def forward_on_tar(self, x, y):
        emb_online_ori = self._pre_forward(x, y)
        with torch.no_grad():
            emb_target = emb_online_ori.clone()
            emb_target.detach()
        emb_online = self.predictor(emb_online_ori)
        return emb_online, emb_target

    def forward(self, x, y):
        return self.predictor(self._pre_forward(x, y))

    def get_taskmem(self, task_emb):
        task_mem_emb = self.task_mem(task_emb)
        return torch.cat([task_emb, task_mem_emb])
   
class TaskEncoderSSL000(nn.Module):
    def __init__(self, task_dim, num_class, emb_size, feat_num, max_shot, device):
        super().__init__()
        self.task_dim = task_dim
        self.num_class = num_class
        self.emb_size = emb_size
        self.feat_num = feat_num
        self.max_shot = max_shot
        self.device = device

        self.emb_y = torch.nn.Embedding(num_embeddings=self.num_class, embedding_dim=self.emb_size)
        self.in_dim = (self.feat_num+1) * self.emb_size # user feat num + item feat num + y emb
        self.hid_dim = int(self.in_dim/2)
        
        # 与taskencoder保持相当的size（taskencoder也是两个mlp)
        self.en_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.task_dim)
            )

        # projection head, for cl
        self.predictor = nn.Linear(self.task_dim, self.task_dim)

    def _statistic_pooling(self, x):
        var = torch.var(x, dim=0)
        mean = torch.mean(x, dim=0)
        return var.reshape(-1), mean
    
    def _aggregate(self, x):
        return torch.mean(x, dim=0)
    
    def _pre_forward(self, x, y):
        y_emb = self.emb_y(y)
        inputs = torch.cat((x, y_emb), 1)
        # inputs_var, inputs_mean = self._statistic_pooling(inputs)
        outs = self.en_layer(inputs)
        return self._aggregate(outs)

    # generate online and target view
    def forward_on_tar(self, x, y):
        emb_online_ori = self._pre_forward(x, y)
        with torch.no_grad():
            emb_target = emb_online_ori.clone()
            emb_target.detach()
        emb_online = self.predictor(emb_online_ori)
        return emb_online, emb_target

    def forward(self, x, y):
        return self.predictor(self._pre_forward(x, y))
   

class TaskEncoder(nn.Module):
    def __init__(self, task_dim, num_class, emb_size, feat_num, max_shot, device) -> None:
        super().__init__()
        # 超参数
        self.task_dim = task_dim
        self.num_class = num_class
        self.emb_size = emb_size
        self.feat_num = feat_num
        self.max_shot = max_shot
        self.device = device

        # 部分变量参数
        self.s_size = self.emb_size
        self.v_size = int(self.emb_size / 2)

        # layers
        # statistic
        self.sta_inter_layer = nn.Linear(2*self.feat_num*self.emb_size+1, self.s_size)
        self.sta_inter_layer2 = nn.Linear(self.v_size*2+2, self.task_dim)
        self.relu_layer = nn.ReLU()

        # encoder
        self.en_layer = nn.Sequential(
            nn.Linear(self.emb_size, int(self.emb_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.emb_size/2), int(self.emb_size/2))
        )
        

    # Compute element-wise sample mean, var., and set cardinality
    # x:(n, es)
    def _statistic_pooling(self, x):
        var = torch.var(x, dim=0)
        mean = torch.mean(x, dim=0)
        if x.shape[0] == 1:
            var = torch.zeros_like(x)
        return var.reshape(-1), mean
        

    # 输入：x:(batch_size, feat_num*emb_size), y:(batch_size, 1)
    def forward(self, x, y):
        # statistic 1
        s = []; Nc_list = []; mean_list = []
        for c in range(self.num_class):
            if c not in y:
                continue
            c_mask = torch.eq(y, c) # cuda
            c_x = x[c_mask] # 不可用masked_select
            c_y = torch.masked_select(y, c_mask)
            N_c = torch.tensor([(len(c_y)-1.)/(self.max_shot-1.)], device=self.device)
            Nc_list.append(N_c)
            c_var, c_mean = self._statistic_pooling(c_x) # 这里cvar出现的nan
            s.append(torch.cat([c_var, c_mean, N_c], 0))
            mean_list.append(c_mean) # c_y, tensor([feat_num*emb_size])
        
        s = torch.stack(s, 0) # n_c, 2*f*es+1
        s = self.relu_layer(self.sta_inter_layer(s)) # interaction

        # encoder 2
        v = self.en_layer(s)

        # statistic 2
        v_var, v_mean = self._statistic_pooling(v)
        v_N = torch.mean(torch.tensor(Nc_list))
        
        len_mean_list = len(mean_list)
        mean_tensor = torch.cat(mean_list, dim=0).reshape(len_mean_list, -1) # list of tensor -> tensor
        mean_norm = mean_tensor / torch.max(torch.norm(mean_tensor, p=2, dim=-1, keepdim=True), torch.tensor(1e-7).to(self.device))
        cos_sim = torch.einsum('ni,mi->nm', mean_norm, mean_norm)
        sim_sum = torch.tensor([0.], device=self.device)
        for i in range(len_mean_list):
            for j in range(i+1, len_mean_list):
                sim_sum += (j-i)*cos_sim[i][j]
        
        v = torch.cat([v_var,v_mean,torch.tensor([v_N], device=self.device),sim_sum])
        
        return self.relu_layer(self.sta_inter_layer2(v))


class Attention(torch.nn.Module):
    def __init__(self, n_k):
        super(Attention, self).__init__()
        self.n_k = n_k
        self.relu_layer = torch.nn.ReLU()
        self.fc_layer = torch.nn.Linear(self.n_k, self.n_k)
        self.soft_max_layer = torch.nn.Softmax()

    def forward(self, pu, mp):
        expanded_pu = pu.repeat(1, len(mp)).view(len(mp), -1)  # shape, n_k, pu_dim
        inputs = self.cos_sim(expanded_pu, mp)
        fc_layers = self.relu_layer(self.fc_layer(inputs))
        attention_values = self.soft_max_layer(fc_layers)
        return attention_values

    def cos_sim(input1, input2):
        query_norm = torch.sqrt(torch.sum(input1**2+0.00001, 1))
        doc_norm = torch.sqrt(torch.sum(input2**2+0.00001, 1))

        prod = torch.sum(torch.mul(input1, input2), 1)
        norm_prod = torch.mul(query_norm, doc_norm)

        cos_sim_raw = torch.div(prod, norm_prod)
        return cos_sim_raw


class MemoryUnit2(nn.Module):
    # clusters_k is k keys
    def __init__(self, clusters_k, emb_size, temperature, device):
        super(MemoryUnit, self).__init__()
        self.clusters_k = clusters_k
        self.embed_size = emb_size
        self.temperature = temperature
        self.device = device
        self.array = nn.Parameter(torch.nn.init.xavier_uniform_(torch.FloatTensor(self.clusters_k, self.embed_size)))
        self.att_model = Attention(self.n_k).to(self.device)

    def forward(self, task_embed):
        atten = self.att_model(task_embed, self.array).to(self.device)
        # res = torch.norm(task_embed-self.array, p=2, dim=1, keepdim=True)
        # res = torch.pow((res / self.temperature) + 1, (self.temperature + 1) / -2)
        # # 1*k
        # C = torch.transpose(res / res.sum(), 0, 1)
        # 1*k, k*d, 1*d
        value = torch.mm(atten, self.array)
        # simple add operation
        new_task_embed = torch.cat([task_embed, value], dim=1)
        # calculate target distribution
        return new_task_embed

class MemoryUnit(nn.Module):
    # clusters_k is k keys
    def __init__(self, clusters_k, emb_size, temperature):
        super(MemoryUnit, self).__init__()
        self.clusters_k = clusters_k
        self.embed_size = emb_size
        self.temperature = temperature
        self.array = nn.Parameter(torch.nn.init.xavier_uniform_(torch.FloatTensor(self.clusters_k, self.embed_size)))

    def forward(self, task_embed):
        res = torch.norm(task_embed-self.array, p=2, dim=1, keepdim=True)
        res = torch.pow((res / self.temperature) + 1, (self.temperature + 1) / -2)
        # 1*k
        C = torch.transpose(res / res.sum(), 0, 1)
        # 1*k, k*d, 1*d
        value = torch.mm(C, self.array)
        return value.view(-1)

class TaskMemEncoder(nn.Module):
    def __init__(self, task_dim, num_class, emb_size, feat_num, max_shot, clusters_k, temperature, device) -> None:
        super().__init__()
        # 超参数
        self.task_dim = task_dim
        self.num_class = num_class
        self.emb_size = emb_size
        self.feat_num = feat_num
        self.max_shot = max_shot
        self.device = device
        self.clusters_k = clusters_k
        self.temperature = temperature
        self.task_mem = MemoryUnit(self.clusters_k, self.task_dim, self.temperature)

        # 部分变量参数
        self.e_size = self.feat_num*self.emb_size
        self.s_size = self.emb_size
        self.v_size = int(self.emb_size / 2)

        # layers
        # statistic
        self.sta_inter_layer = nn.Linear(2*self.feat_num*self.emb_size+1, self.s_size)
        self.sta_inter_layer2 = nn.Linear(self.v_size*2+2, self.task_dim)
        # self.final_emb = nn.Linear(2*self.task_dim, self.task_dim)
        self.relu_layer = nn.ReLU()

        # encoder
        self.en_layer0 = nn.Sequential(
            nn.Linear(self.e_size, self.e_size),
            nn.ReLU(),
            nn.Linear(self.e_size, self.e_size)
        )

        self.en_layer = nn.Sequential(
            nn.Linear(self.emb_size, int(self.emb_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.emb_size/2), int(self.emb_size/2))
        )
    
    # Compute element-wise sample mean, var., and set cardinality
    # x:(n, es)
    def _statistic_pooling(self, x):
        var = torch.var(x, dim=0)
        mean = torch.mean(x, dim=0)
        if x.shape[0] == 1:
            var = torch.zeros_like(x)
        return var.reshape(-1), mean
    
    # 输入：x:(batch_size, feat_num*emb_size), y:(batch_size, 1)
    def forward(self, x, y):
        # encoder 1
        x = self.en_layer0(x)
        
        # statistic 1
        s = []; Nc_list = []; mean_list = []
        for c in range(self.num_class):
            if c not in y:
                continue
            c_mask = torch.eq(y, c) # cuda
            c_x = x[c_mask] # 不可用masked_select
            c_y = torch.masked_select(y, c_mask)
            N_c = torch.tensor([(len(c_y)-1.)/(self.max_shot-1.)], device=self.device)
            Nc_list.append(N_c)
            c_var, c_mean = self._statistic_pooling(c_x) # 这里cvar出现的nan
            s.append(torch.cat([c_var, c_mean, N_c], 0))
            mean_list.append(c_mean) # c_y, tensor([feat_num*emb_size])
        
        s = torch.stack(s, 0) # n_c, 2*f*es+1
        s = self.relu_layer(self.sta_inter_layer(s)) # interaction

        # encoder 2
        v = self.en_layer(s)

        # statistic 2
        v_var, v_mean = self._statistic_pooling(v)
        v_N = torch.mean(torch.tensor(Nc_list))
        
        len_mean_list = len(mean_list)
        mean_tensor = torch.cat(mean_list, dim=0).reshape(len_mean_list, -1) # list of tensor -> tensor
        mean_norm = mean_tensor / torch.max(torch.norm(mean_tensor, p=2, dim=-1, keepdim=True), torch.tensor(1e-7).to(self.device))
        cos_sim = torch.einsum('ni,mi->nm', mean_norm, mean_norm)
        sim_sum = torch.tensor([0.], device=self.device)
        for i in range(len_mean_list):
            for j in range(i+1, len_mean_list):
                sim_sum += (j-i)*cos_sim[i][j]
        
        v = torch.cat([v_var,v_mean,torch.tensor([v_N], device=self.device),sim_sum])
        task_emb = self.relu_layer(self.sta_inter_layer2(v))
        task_mem_emb = self.task_mem(task_emb)
        # print(task_emb.shape, task_mem_emb.shape)
        
        # return self.relu_layer(self.final_emb(torch.cat([task_emb, task_mem_emb])))
        return torch.cat([task_emb, task_mem_emb])



class TaskMeanEncoder(nn.Module):
    def __init__(self, task_dim, num_class, emb_size, feat_num, max_shot, device) -> None:
        super().__init__()
        # 超参数
        self.task_dim = task_dim
        # self.num_class = num_class
        self.emb_size = emb_size
        self.feat_num = feat_num
        # self.max_shot = max_shot
        self.device = device

        # 部分变量参数
        self.encoder_layer = nn.Linear(self.feat_num*self.emb_size, self.task_dim)
        self.relu_layer = nn.ReLU()

        # 输入：x:(batch_size, feat_num*emb_size), y:(batch_size, 1)
    def forward(self, x, y):
        return torch.mean(self.relu_layer(self.encoder_layer(x)), dim=0)

class TaskMeanMemEncoder(nn.Module):
    def __init__(self, task_dim, num_class, emb_size, feat_num, max_shot, clusters_k, temperature, device) -> None:
        super().__init__()
        # 超参数
        self.task_dim = task_dim
        self.emb_size = emb_size
        self.device = device
        self.feat_num = feat_num
        self.clusters_k = clusters_k
        self.temperature = temperature
        self.task_mem = MemoryUnit(self.clusters_k, self.task_dim, self.temperature)

        # 部分变量参数
        self.encoder_layer = nn.Linear(self.feat_num*self.emb_size, self.task_dim)
        self.relu_layer = nn.ReLU()

        # 输入：x:(batch_size, feat_num*emb_size), y:(batch_size, 1)
    def forward(self, x, y):
        task_emb = torch.mean(self.relu_layer(self.encoder_layer(x)), dim=0)
        task_mem_emb = self.task_mem(task_emb)
        return torch.cat([task_emb, task_mem_emb])
