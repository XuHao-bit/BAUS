# from pickle import NEXT_BUFFER
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from modules.MyOptim import MyAdam
import pandas as pd

from utils import *
from DataLoading import *
import robust_loss_pytorch.general
from torch.nn.functional import cosine_similarity
# from torch.utils.tensorboard import SummaryWriter



# 优化器，参考few_shot_learning_system.py的一部分（原来的代码不太好懂, 以及TaNP的Trainer
class Modeling:
    def __init__(self, model, config_settings, mode='Meta') -> None:
        self.device = config_settings['device']
        self.n_inner_loop = config_settings['inner_loop_steps']
        self.n_epoch = config_settings['n_epoch']
        self.batch_size = config_settings['batch_size']
        self.meta_lr = config_settings['meta_lr'] # global lr
        self.local_lr = config_settings['local_lr'] # local lr for melu
        # self.tdmeta_lr = config_settings['tdmeta_lr']
        self.meta_wd = config_settings['meta_wd'] # global wd
        # self.writer = SummaryWriter(config_settings['writer_log'])
        self.train_writer = SummaryWriter(config_settings['train_log'])
        self.use_writer = config_settings['use_writer']
        self.use_gen_hypr = config_settings['use_gen_hypr']
        self.use_grad_clip = config_settings['use_grad_clip']
        self.pstep_only = config_settings['pstep_only']
        self.player_only = config_settings['player_only']
        self.clip_norm = config_settings['clip_norm']
        self.model = model.to(self.device)
        self.local_model = deepcopy(self.model.base_model)
        self.model_name = config_settings['model_name']
        self.early_stop = EarlyStopping(config_settings['early_step'], path=f'./saved_models/{self.model_name}_cp.pth')
        
        # ablation
        self.ada_lr_only = config_settings['ada_lr_only']
        self.ada_wd_only = config_settings['ada_wd_only']

        # adaptive weighted loss
        self.use_aloss = config_settings['use_aloss']
        # adaptive reg loss
        self.use_aregloss = config_settings['use_aregloss']
        self.use_mrgloss = config_settings['use_mrgloss']   # meta rgloss
        self.use_aregloss_combine = config_settings['use_aregloss_combine']
        self.use_adap_rgloss = config_settings['use_adap_rgloss']
        self.reg_weight = config_settings['reg_weight']
        # perturb data
        self.use_sslencoder = config_settings['use_sslencoder']
        self.use_ssl = config_settings['use_ssl']
        self.ssl_weight = config_settings['ssl_weight']
        # self.view_ratio = config_settings['view_ratio']
        self.dataset = self.model.dataset
        self.use_tmem = config_settings['use_tmem']
        # self.i_in_dim = self.model
        self.use_earlystop = config_settings['use_earlystop']
        self.use_scheduler = config_settings['use_scheduler']
        self.min_lr_coeff = config_settings['min_lr_coeff']
        # self.min_lr = self.min_lr_coeff * self.meta_lr
        self.min_lr = config_settings['min_lr']
        self.sch_epoch = config_settings['sch_epoch']
        self.adap_loss_info = {'task':[],'step':[],'ga':[],'c':[]}
        

        mode = mode.lower()
        # task adaptive optimizer init
        if mode == 'tdmeta':
            phi_copy = self.model.base_model.state_dict()
            self.model.task_adaptive_optimizer_init(phi_copy)

            self.optimizer = torch.optim.Adam([
                {'params': self.model.task_encoder.parameters()},
                {'params': self.model.task_adaptive_optimizer.parameters()},
            ], lr=self.meta_lr, amsgrad=False, weight_decay=self.meta_wd)

            if self.use_scheduler:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.sch_epoch,
                                                            eta_min=self.min_lr)

            # global phi optimizer
            self.global_optimizer = MyAdam(phi_copy, weight_decay=self.meta_wd)

        elif mode == 'melu':
            # local update optimizer should be sgd
            # melu model didn't use weight decay item when local updating
            self.local_optimizer = torch.optim.SGD(self.model.decoder.parameters(), lr=self.local_lr)
            # self.local_optimizer_adam = torch.optim.Adam(self.model.decoder.parameters(), lr=self.local_lr)
            # when global updating, it doesn't use weight decay too.
            self.global_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)
        else:
            self.local_optimizer = torch.optim.SGD(self.model.decoder.parameters(), lr=self.local_lr)
            self.global_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)

        # update rules
        if self.use_aloss:
            self.ori_loss = nn.MSELoss(reduction='none')
        
        self.loss_fn = nn.MSELoss()

        if self.use_aregloss:
            self.robust_loss = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims = 1, float_dtype=np.float32, device=self.device)
            self.loss_optim = torch.optim.Adam(self.robust_loss.parameters(), lr=self.local_lr)
            if self.use_mrgloss:
                self.loss_optim_global = MyAdam(self.robust_loss.state_dict(), weight_decay=self.meta_wd)
            

        # self.local_optim = torch.optim.Adam([
        #     {'params': self.model.parameters()},
        # ], lr=self.meta_lr, amsgrad=False)
    
    # def loss_fn(self, yhat, y, use_aloss=False):
    #     # loss = (yhat - y) ** 2
    #     return self.ori_loss(yhat, y)

    def write_board(self, writer_info, ga, c, step):
        if writer_info:
            # writer = writer_info['writer']
            task = writer_info['task']
            self.adap_loss_info['task'].append(task)
            self.adap_loss_info['step'].append(step)
            self.adap_loss_info['ga'].append(ga)
            self.adap_loss_info['c'].append(c)

    # [local update] ===================================
    def local_update(self, sup_x1, sup_x2, sup_y, optimizer, theta=None):
        # load theta for meta learning local update
        if theta:
            self.model.decoder.load_state_dict(theta)
        
        for i_loop in range(self.n_inner_loop):
            # applying task adaptive local update
            y_hat = self.model(sup_x1, sup_x2)
            # print(y_hat, sup_y)
            sup_loss = self.loss_fn(y_hat, sup_y.view(-1, 1))
            optimizer.zero_grad()
            sup_loss.backward()
            optimizer.step()

    def tdmeta_local_update(self, sup_x1, sup_x2, sup_y, que_x1, que_x2, que_y, task_emb, writer_info=None):
        # with torch.autograd.set_detect_anomaly(True):
        model_keys = self.model.base_model.state_dict().keys()
        task_losses = []
        for i_loop in range(self.n_inner_loop):
            if i_loop == 0:
                this_model = self.model.base_model
            else:
                this_model = self.local_model
        
            # applying task adaptive local update
            # self.model.base_model.load_state_dict(theta)
            y_hat = this_model(sup_x1, sup_x2)
            # print(y_hat, sup_y)

            # aloss
            if self.use_aloss:
                sup_loss = self.ori_loss(y_hat, sup_y.view(-1, 1))
                weight = self.model.task_adaptive_optimizer.gen_aloss_weight(task_emb)
                sup_loss = sup_loss.squeeze(1) * weight
                sup_loss = torch.mean(sup_loss)
            else:
                sup_loss = self.loss_fn(y_hat, sup_y.view(-1, 1))
            
            sup_grad = torch.autograd.grad(sup_loss, this_model.parameters(), retain_graph=True)            
            
            # sup_grad_dict = dict(zip(model_keys, sup_grad))
            # for key, grad in sup_grad_dict.items():
            #     if grad is None:
            #         print('Grads not found for inner loop parameter', key)
            #     sup_grad_dict[key] = sup_grad_dict[key].sum(dim=0)

            # grad clip here.
            if self.use_grad_clip:
                apply_grad_clip_norm(sup_grad, max_norm=self.clip_norm)
            
            sup_grad_dict = dict(zip(model_keys, sup_grad))
            for key, grad in sup_grad_dict.items():
                if grad is None:
                    print('Grads not found for inner loop parameter', key)
                sup_grad_dict[key] = sup_grad_dict[key].sum(dim=0)

            # get generated hyper parameters
            gen_alpha_dict, gen_beta_dict = {}, {}
            # If generated hyper parameters are not used, the rec_model will be MeLU
            if self.use_gen_hypr:
                # get task info by theta and grad
                per_step_task_info = []
                for v in this_model.state_dict().values(): # self.model.base_model.state_dict().values()
                    per_step_task_info.append(v.mean())
                for grad in sup_grad_dict.values():
                    per_step_task_info.append(grad.mean())
                per_step_task_info = torch.stack(per_step_task_info)
                
                # generate per-layer adaptive hyper-params by task_diff and task_info
                gen_alpha, gen_beta = self.model.task_adaptive_optimizer.gen_hyper_params(task_emb, sup_loss.reshape(1),\
                                                                                        per_step_task_info)
            
                # print(gen_alpha.shape)
                # make the generated params have the same key as theta
                for idx, key in enumerate(model_keys): # self.model.base_model.state_dict().keys()
                    # ablation
                    if self.ada_lr_only:
                        gen_alpha_dict[key] = gen_alpha[idx]
                        gen_beta_dict[key] = 1
                    elif self.ada_wd_only:
                        gen_alpha_dict[key] = 1
                        gen_beta_dict[key] = gen_beta[idx]
                    else:
                        gen_alpha_dict[key] = gen_alpha[idx]
                        gen_beta_dict[key] = gen_beta[idx]
            else:
                for idx, key in enumerate(model_keys):
                    gen_alpha_dict[key] = 1
                    gen_beta_dict[key] = 1

            if i_loop == 0:
                # 第1轮, local_model经过model计算得来, local_model变为non-leaf
                self.model.task_adaptive_optimizer.update_params4(model=self.model.base_model, # self.model.base_model.state_dict().values()
                                                            local_model=self.local_model,
                                                            names_grads_dict=sup_grad_dict,
                                                            gen_alpha_dict=gen_alpha_dict,
                                                            gen_beta_dict=gen_beta_dict,
                                                            num_step=i_loop,
                                                            writer_info=writer_info)
                # if not use per-step adaptive update
                if self.player_only: 
                    self.gen_alpha = gen_alpha_dict
                    self.gen_beta = gen_beta_dict
            else:
                # if not use per-step adaptive update
                if self.player_only: 
                    gen_alpha_dict = self.gen_alpha
                    gen_beta_dict = self.gen_beta
                    this_step = 0
                else:
                    this_step = i_loop
                # 经过update_param之后，is_leaf为false
                self.model.task_adaptive_optimizer.update_params3(model=self.local_model,
                                                                names_grads_dict=sup_grad_dict,
                                                                gen_alpha_dict=gen_alpha_dict,
                                                                gen_beta_dict=gen_beta_dict,
                                                                num_step=this_step,
                                                                writer_info=writer_info)
            # self.model.zero_grad()
            que_yhat = self.local_model(que_x1, que_x2)
            # print(que_yhat, que_y)
            que_loss = self.loss_fn(que_yhat, que_y.view(-1, 1))
            # que_loss.backward(retain_graph=True)
            task_losses.append(que_loss)
            # print('[local] task encoder--')
            # grads = torch.autograd.grad(que_loss, self.model.task_encoder.parameters(), retain_graph=True, allow_unused=True)
            # for n, g in zip(self.model.task_encoder.state_dict().keys(), grads):
            #     print(n, g)
            
        return task_losses

    def tdmeta_local_update_regloss(self, sup_x1, sup_x2, sup_y, que_x1, que_x2, que_y, task_emb, writer_info=None):
        # with torch.autograd.set_detect_anomaly(True):
        model_keys = self.model.base_model.state_dict().keys()
        task_losses = []

        for i_loop in range(self.n_inner_loop):
            if i_loop == 0:
                this_model = self.model.base_model
            else:
                this_model = self.local_model
        
            # applying task adaptive local update
            # self.model.base_model.load_state_dict(theta)
            y_hat = this_model(sup_x1, sup_x2)

            # aloss
            if self.use_aloss:
                sup_loss = self.ori_loss(y_hat, sup_y.view(-1, 1))
                weight = self.model.task_adaptive_optimizer.gen_aloss_weight(task_emb)
                sup_loss = sup_loss.squeeze(1) * weight
                sup_loss = torch.mean(sup_loss)
            elif self.use_aregloss:
                if self.use_adap_rgloss:
                    al_coef, sca_coef = self.model.task_adaptive_optimizer.gen_reg_weight(task_emb)
                    robust_loss = torch.mean(self.robust_loss.lossfun_adap(y_hat - sup_y.view(-1, 1), al_coef, sca_coef))
                else:
                    robust_loss = torch.mean(self.robust_loss.lossfun(y_hat - sup_y.view(-1, 1)))
                if self.use_aregloss_combine:
                    ori_loss = self.loss_fn(y_hat, sup_y.view(-1, 1))
                    sup_loss = (robust_loss + ori_loss) / 2
                else:
                    sup_loss = robust_loss
                
                # write adap. robust loss hyper parameter
                self.write_board(writer_info, self.robust_loss.alpha().item(),\
                                                self.robust_loss.scale().item(), i_loop)

            else:
                sup_loss = self.loss_fn(y_hat, sup_y.view(-1, 1))
                        
            if self.use_aregloss:
                sup_grad = torch.autograd.grad(sup_loss, this_model.parameters(), retain_graph=True)
                self.loss_optim.zero_grad()
                sup_loss.backward(retain_graph=True)
                self.loss_optim.step()
            else:
                sup_grad = torch.autograd.grad(sup_loss, this_model.parameters(), retain_graph=True)

            # grad clip here.
            if self.use_grad_clip:
                apply_grad_clip_norm(sup_grad, max_norm=self.clip_norm)
            
            sup_grad_dict = dict(zip(model_keys, sup_grad))
            for key, grad in sup_grad_dict.items():
                if grad is None:
                    print('Grads not found for inner loop parameter', key)
                sup_grad_dict[key] = sup_grad_dict[key].sum(dim=0)

            # get generated hyper parameters
            gen_alpha_dict, gen_beta_dict = {}, {}
            # If generated hyper parameters are not used, the rec_model will be MeLU
            if self.use_gen_hypr:
                # get task info by theta and grad
                per_step_task_info = []
                for v in this_model.state_dict().values(): # self.model.base_model.state_dict().values()
                    per_step_task_info.append(v.mean())
                for grad in sup_grad_dict.values():
                    per_step_task_info.append(grad.mean())
                per_step_task_info = torch.stack(per_step_task_info)
                
                # generate per-layer adaptive hyper-params by task_diff and task_info
                gen_alpha, gen_beta = self.model.task_adaptive_optimizer.gen_hyper_params(task_emb, sup_loss.reshape(1),\
                                                                                        per_step_task_info)
            
                # print(gen_alpha.shape)
                # make the generated params have the same key as theta
                for idx, key in enumerate(model_keys): # self.model.base_model.state_dict().keys()
                    # ablation
                    if self.ada_lr_only:
                        gen_alpha_dict[key] = gen_alpha[idx]
                        gen_beta_dict[key] = 1
                    elif self.ada_wd_only:
                        gen_alpha_dict[key] = 1
                        gen_beta_dict[key] = gen_beta[idx]
                    else:
                        gen_alpha_dict[key] = gen_alpha[idx]
                        gen_beta_dict[key] = gen_beta[idx]
            else:
                for idx, key in enumerate(model_keys):
                    gen_alpha_dict[key] = 1
                    gen_beta_dict[key] = 1

            if i_loop == 0:
                # 第1轮, local_model经过model计算得来, local_model变为non-leaf
                self.model.task_adaptive_optimizer.update_params4(model=self.model.base_model, # self.model.base_model.state_dict().values()
                                                            local_model=self.local_model,
                                                            names_grads_dict=sup_grad_dict,
                                                            gen_alpha_dict=gen_alpha_dict,
                                                            gen_beta_dict=gen_beta_dict,
                                                            num_step=i_loop,
                                                            writer_info=writer_info)
                # if not use per-step adaptive update
                if self.player_only: 
                    self.gen_alpha = gen_alpha_dict
                    self.gen_beta = gen_beta_dict
            else:
                # if not use per-step adaptive update
                if self.player_only: 
                    gen_alpha_dict = self.gen_alpha
                    gen_beta_dict = self.gen_beta
                    this_step = 0
                else:
                    this_step = i_loop
                # 经过update_param之后，is_leaf为false
                self.model.task_adaptive_optimizer.update_params3(model=self.local_model,
                                                                names_grads_dict=sup_grad_dict,
                                                                gen_alpha_dict=gen_alpha_dict,
                                                                gen_beta_dict=gen_beta_dict,
                                                                num_step=this_step,
                                                                writer_info=writer_info)
            # self.model.zero_grad()
            que_yhat = self.local_model(que_x1, que_x2)
            # print(que_yhat, que_y)
            # if self.use_aregloss:
            #     if self.use_adap_rgloss:
            #         al_coef, sca_coef = self.model.task_adaptive_optimizer.gen_reg_weight(task_emb)
            #         robust_loss = torch.mean(self.robust_loss.lossfun_adap(que_yhat - que_y.view(-1, 1), al_coef, sca_coef))
            #     else:
            #         robust_loss = torch.mean(self.robust_loss.lossfun(que_yhat - que_y.view(-1, 1)))
            #     if self.use_aregloss_combine:
            #         ori_loss = self.loss_fn(que_yhat, que_y.view(-1, 1))
            #         que_loss = (robust_loss + ori_loss) / 2
            #     else:
            #         que_loss = robust_loss
                
            #     if not self.use_mrgloss:
            #         self.loss_optim.zero_grad()
            #         que_loss.backward(retain_graph=True)
            #         self.loss_optim.step()
            # if self.use_aregloss:
            # que_loss = torch.mean(self.robust_loss.lossfun(que_yhat - que_y.view(-1, 1)))
            que_loss = self.loss_fn(que_yhat, que_y.view(-1, 1)) # 0.770

            # que_loss.backward(retain_graph=True)
            task_losses.append(que_loss)
            # print('[local] task encoder--')
            # grads = torch.autograd.grad(que_loss, self.model.task_encoder.parameters(), retain_graph=True, allow_unused=True)
            # for n, g in zip(self.model.task_encoder.state_dict().keys(), grads):
            #     print(n, g)
            
        return task_losses

    def test(self, data, tr_epoch=0, perturb=False, prob=0.):
        all_loss = 0
        rmse = []
        mae = []
        ndcg3, ndcg5 = [], []
        # phi = deepcopy(self.model.base_model.state_dict())
        if self.use_aregloss:
            reg_phi = deepcopy(self.robust_loss.state_dict())

        for i in tqdm(range(len(data))):
            # # theta = deepcopy(phi)
            # self.model.base_model.load_state_dict(theta)
            sup_x1, sup_x2, sup_y, que_x1, que_x2, que_y = data[i]
            if perturb:
                sup_y = perturb_item2(sup_y, self.dataset, prob)
            sup_x1, sup_x2, sup_y, que_x1, que_x2, que_y = sup_x1.to(self.device), sup_x2.to(self.device),\
                                                                sup_y.to(self.device), que_x1.to(self.device),\
                                                                que_x2.to(self.device), que_y.to(self.device)

            # if not self.use_ssl:
            #         task_emb = self.model.get_task_emb_nossl(sup_x1, sup_x2, sup_y)
            #         if self.use_tmem:
            #             task_emb = self.model.task_encoder.get_taskmem(task_emb)
            task_emb = self.model.get_task_embedding(sup_x1, sup_x2, sup_y)
            if self.use_tmem:
                task_emb = self.model.task_encoder.get_taskmem(task_emb)

            writer_info = None
            if self.use_writer:
                writer_info = {'task':i, 'writer':None, 'stage':'test', 'epoch':tr_epoch}

            # local update
            if not self.use_aregloss:
                que_losses = self.tdmeta_local_update(sup_x1=sup_x1, sup_x2=sup_x2, sup_y=sup_y, 
                                        que_x1=que_x1, que_x2=que_x2, que_y=que_y, task_emb=task_emb, writer_info=writer_info)
            else:
                self.robust_loss.load_state_dict(reg_phi)
                que_losses = self.tdmeta_local_update_regloss(sup_x1=sup_x1, sup_x2=sup_x2, sup_y=sup_y, 
                                        que_x1=que_x1, que_x2=que_x2, que_y=que_y, task_emb=task_emb, writer_info=writer_info)
            
            # print('after theta:{}'.format(theta['hidden_layer_1.weight']))
            # que loss
            with torch.no_grad():
                que_y_hat = self.local_model(que_x1, que_x2).cpu()
                all_loss += torch.mean(torch.stack(que_losses)).item()
                # print(que_y_hat)

            # que_y_pred = torch.argmax(que_y_hat, dim=1) # range: 0-4
            mae.append(MAE(que_y_hat.view(-1), que_y.cpu()))
            rmse.append(RMSE(que_y_hat.view(-1), que_y.cpu()))
            ndcg3.append(NDCG(que_y_hat.view(-1), que_y.cpu(), 3))
            ndcg5.append(NDCG(que_y_hat.view(-1), que_y.cpu(), 5))
        
        # visual experiment
        if self.use_writer:
            df2 = pd.DataFrame(self.adap_loss_info)
            df2.to_csv('/home/zhaoxuhao/UASRec/res/visual/ada_loss_test.csv', sep=',', index=False, header=True)


        mmae = sum(mae) / len(mae)
        mrmse = sum(rmse) / len(rmse)
        mndcg3 = sum(ndcg3) / len(ndcg3)
        mndcg5 = sum(ndcg5) / len(ndcg5)
        mloss = all_loss / len(data) 
        
        # print(mae, rmse)
        print(f'mae:{mmae.item()}, rmse:{mrmse.item()}, ndcg3:{mndcg3}, ndcg5:{mndcg5}')
        return mmae, mrmse, mloss

    # new train
    def train(self, data, test_data):
        # init phi
        phi = deepcopy(self.model.base_model.state_dict())
        if self.use_mrgloss:
            phi_l = deepcopy(self.robust_loss.state_dict())

        best_model, best_mae = None, None
        for epoch in range(self.n_epoch):
            if self.use_scheduler:
                self.scheduler.step(epoch=epoch)
            print(f'epoch: {epoch}')
            all_loss = 0
            all_ssl_loss = 0
            len_data = len(data)
            n_batch = int(np.ceil(len_data / self.batch_size))

            for i_batch in tqdm(range(n_batch)):
                start = i_batch*self.batch_size
                end = (i_batch+1)*self.batch_size if (i_batch+1)*self.batch_size <= len_data else len_data

                # batch_data = data[: ]
                batch_que_losses = []
                # batch_que_losses_value = 0.
                batch_que_grads = []
                if self.use_mrgloss:
                    batch_que_grads_l = []
                if self.use_ssl:
                    batch_ssl_losses = []

                for i in range(start, end):
                    # init theta: theta <- phi
                    # theta is the local param (which means the model's param)
                    if i > 0:
                        self.model.base_model.load_state_dict(phi)
                        if self.use_mrgloss:
                            self.robust_loss.load_state_dict(phi_l)
                    
                    # self.local_model.load_state_dict(phi)
                    sup_x1, sup_x2, sup_y, que_x1, que_x2, que_y = data[i]
                    sup_x1, sup_x2, sup_y, que_x1, que_x2, que_y = sup_x1.to(self.device), sup_x2.to(self.device),\
                                                                    sup_y.to(self.device), que_x1.to(self.device),\
                                                                    que_x2.to(self.device), que_y.to(self.device)

                    # no.1 get per-step task embedding.
                    if not self.use_ssl:
                        task_emb = self.model.get_task_embedding(sup_x1, sup_x2, sup_y)
                        if self.use_tmem:
                            task_emb = self.model.task_encoder.get_taskmem(task_emb)
                    else:
                        # generate another view
                        sup_x1_con, sup_x2_con, sup_y_con = sup_x1.clone(), sup_x2.clone(), sup_y.clone()
                        # sup_x2_con = perturb_item(sup_x2_con, self.dataset, self.view_ratio)
                        # calculate contrastive loss
                        task_emb_on, task_emb_tar = self.model.get_contrastive_taskemb(sup_x1, sup_x2, sup_y) # all items
                        contr_emb_on, contr_emb_tar = self.model.get_sslview_taskemb(sup_x1_con, sup_x2_con, sup_y_con) # droped items
                        # ssl_loss = (cosine_similarity(task_emb_on, contr_emb_tar) + \
                        #             cosine_similarity(task_emb_tar, contr_emb_on)) / 2
                        # orign -> contrastive
                        loss_oc = 1 - cosine_similarity(task_emb_on, contr_emb_tar.detach(), dim=-1).mean()
                        # contras -> origin
                        loss_co = 1 - cosine_similarity(contr_emb_on, task_emb_tar.detach(), dim=-1).mean()
                        batch_ssl_losses.append((loss_oc + loss_co).mean())
                        # task_emb = task_emb_on
                        if self.use_tmem:
                            task_emb = self.model.task_encoder.get_taskmem(task_emb_on)
                        else:
                            task_emb = task_emb_on

                    # no.2 local loop
                    writer_info = None
                    if self.use_writer and epoch==2:
                        writer_info = {'task':i, 'writer':None, 'stage':'train', 'epoch':epoch}
                    
                    if not self.use_aregloss:
                        que_losses = self.tdmeta_local_update(sup_x1=sup_x1, sup_x2=sup_x2, sup_y=sup_y, 
                                            que_x1=que_x1, que_x2=que_x2, que_y=que_y, task_emb=task_emb, writer_info=writer_info)
                    else:
                        que_losses = self.tdmeta_local_update_regloss(sup_x1=sup_x1, sup_x2=sup_x2, sup_y=sup_y, 
                                            que_x1=que_x1, que_x2=que_x2, que_y=que_y, task_emb=task_emb, writer_info=writer_info)
                        
                    # no.3 que_set loss
                    que_loss_sum = torch.sum(torch.stack(que_losses))
                    
                    self.local_model.zero_grad()
                    # phi_grads = torch.autograd.grad(que_losses[-1], self.local_model.parameters(), retain_graph=True, allow_unused=True)
                    phi_grads = torch.autograd.grad(que_loss_sum, self.local_model.parameters(), retain_graph=True, allow_unused=True)
                    # grad clip here.
                    if self.use_grad_clip:
                        apply_grad_clip_norm(phi_grads, max_norm=self.clip_norm)
                    # append loss, grad
                    batch_que_losses.append(que_loss_sum)
                    batch_que_grads.append(phi_grads)
                    # batch_que_losses_value += torch.mean(torch.stack(que_losses)).item()
                    
                    if self.use_mrgloss:
                        phi_grads_l = torch.autograd.grad(que_loss_sum, self.robust_loss.parameters(), retain_graph=True, allow_unused=True)
                        if self.use_grad_clip:
                            apply_grad_clip_norm(phi_grads_l, max_norm=self.clip_norm)
                        batch_que_grads_l.append(phi_grads_l)

                # after the batch
                que_loss = torch.mean(torch.stack(batch_que_losses))
                all_loss += que_loss.item()
                if self.use_ssl:
                    ssl_loss = torch.mean(torch.stack(batch_ssl_losses))
                    que_loss = que_loss + self.ssl_weight * ssl_loss
                    all_ssl_loss += ssl_loss.item()
                
                # que grads calculate the avg value
                if self.use_scheduler:
                    cur_lr = self.scheduler.get_lr()[0]
                else:
                    cur_lr = self.meta_lr
                
                phi_grads = self.global_optimizer.cal_mean_grads(batch_que_grads)
                # update phi
                self.global_optimizer.step(phi, phi_grads, cur_lr)

                if self.use_mrgloss:
                    phi_grads_l = self.loss_optim_global.cal_mean_grads(batch_que_grads_l)
                    self.loss_optim_global.step(phi_l, phi_grads_l, cur_lr)

                if end == len_data:
                    self.model.base_model.load_state_dict(phi)
                    if self.use_mrgloss:
                        self.robust_loss.load_state_dict(phi_l)

                # update robust loss & task_adaptive_optim, task_encoder
                # if self.use_aregloss:
                #     self.loss_optim.zero_grad()
                self.optimizer.zero_grad()
                que_loss.backward()
                # grad clip here.
                if self.use_grad_clip:
                    nn.utils.clip_grad_norm_(self.model.task_encoder.parameters(), max_norm=self.clip_norm)
                    nn.utils.clip_grad_norm_(self.model.task_adaptive_optimizer.parameters(), max_norm=self.clip_norm)
                # if self.use_aregloss:
                #     self.loss_optim.step()
                self.optimizer.step()

            #         
            if self.use_writer:
                df = pd.DataFrame(self.model.task_adaptive_optimizer.adap_info)
                df.to_csv('/home/zhaoxuhao/UASRec/res/visual/ada_hp.csv', sep=',', index=False, header=True)
                df2 = pd.DataFrame(self.adap_loss_info)
                df2.to_csv('/home/zhaoxuhao/UASRec/res/visual/ada_loss.csv', sep=',', index=False, header=True)

                
            train_ssl_loss = all_ssl_loss / n_batch
            if self.use_ssl:
                train_all_loss = all_loss / n_batch
                train_loss = train_all_loss + self.ssl_weight * train_ssl_loss
                print('train loss:{:.4f}, ori loss:{:.4f}, ssl loss:{:.4f}'.format(train_loss, train_all_loss, train_ssl_loss))
            else:
                train_loss = all_loss / n_batch
                print('train loss:{:.4f}'.format(train_loss))
            
            mmae, mrmse, val_loss = self.test(test_data, epoch)

            # write loss
            self.train_writer.add_scalar(f'{config_settings["n_train_log"]}/train_loss', train_loss, epoch)
            self.train_writer.add_scalar(f'{config_settings["n_train_log"]}/val_loss', val_loss, epoch)

            self.early_stop(val_loss=mmae, model=self.model)
            if self.early_stop.early_stop and self.use_earlystop:    
                print('Early_stop!')    
                break

        return best_model

