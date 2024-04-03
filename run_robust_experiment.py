import argparse
import datetime
import os
from itertools import product

from TaskDiffModel import TaskDiffModel, DeepModel
from Trainer import Modeling
from DataLoading import *
from utils import config, dataset_info, config_settings, seed_all
from copy import deepcopy

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='dbook')
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--mode', type=str, default='tdmeta')
parser.add_argument('--save', type=str, default='False')
# parser.add_argument('--use_writer', type=str, default='False')
parser.add_argument('--save_name', type=str, default=f'model_{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--sch_epoch', type=int, default=100)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--use_gen_hypr', type=str, default='True')
parser.add_argument('--use_lea_hypr', type=str, default='True')
parser.add_argument('--use_ginfo_only', type=str, default='False')   # 代表generating adaptive hyper-parameters的时候，仅使用avg grad 以及avg weight信息；
parser.add_argument('--use_tinfo_only', type=str, default='False')   # 代表generating adaptive hyper-parameters的时候，仅使用task emb信息；
parser.add_argument('--nshot', type=int, default=15)
parser.add_argument('--use_tmem', type=boolean_string, default=True)

# adaptive weighted loss
parser.add_argument('--use_aloss', type=boolean_string, default=False)
# adaptive regression loss
parser.add_argument('--use_aregloss', type=boolean_string, default=True)
parser.add_argument('--use_mrgloss', type=boolean_string, default=False) # use meta rgloss
parser.add_argument('--use_aregloss_combine', type=boolean_string, default=False)
parser.add_argument('--use_adap_rgloss', type=boolean_string, default=False)
# use ssl encoder
parser.add_argument('--use_sslencoder', type=boolean_string, default=True)
parser.add_argument('--use_ssl', type=boolean_string, default=True)
parser.add_argument('--aug1', type=boolean_string, default=True)
parser.add_argument('--aug2', type=boolean_string, default=False)
parser.add_argument('--prob', type=float, default=0.3)
# use_grad_clip
parser.add_argument('--use_grad_clip', type=boolean_string, default=True)

parser.add_argument('--use_earlystop', type=boolean_string, default=True)
parser.add_argument('--early_step', type=int, default=20)
parser.add_argument('--use_scheduler', type=boolean_string, default=False)

# === prepare log ===
today_time = datetime.datetime.now().strftime("%Y-%m-%d")
log_dir = f'./log/{today_time}'
os.makedirs(log_dir, exist_ok=True)

# === prepare args === 
args = vars(parser.parse_args())
# seed_all(args['seed'])

for key in args.keys():
    config_settings[key] = args[key]

config_settings['n_epoch'] = args['epoch']
# if args['cuda'] ==
if torch.cuda.is_available():
    config_settings['device'] = 'cuda:{}'.format(args['cuda'])
if args['use_gen_hypr'] == 'False':
    config_settings['use_gen_hypr'] = False
if args['use_lea_hypr'] == 'False':
    config_settings['use_learnable_lr'] = False
    config_settings['use_learnable_wd'] = False
if args['use_ginfo_only'] == 'True':
    config_settings['use_ginfo_only'] = True
if args['use_tinfo_only'] == 'True':
    config_settings['use_tinfo_only'] = True
if args['nshot'] > 15:
    args['nshot'] = 15
if args['use_tmem'] == 'True':
    config_settings['use_tmem'] = True
# if config_settings['use_sslencoder']:
#     config_settings['task_dim'] = 100

# config_settings['use_aloss'] = args['use_aloss']

print(args)

# === prepare data ===
train_user, test_user = get_train_test_user_set(args['dataset'])
train_dataset = fix_load_dataset(train_user,args['dataset'],args['nshot'],25)
test_dataset = fix_load_dataset(test_user,args['dataset'],args['nshot'],25)
# val_dataset = load_dataset(val_user,'movielens',15,25)
# config_settings['device'] = 'cuda:1'

# === prepare hyper-param ===
# inner_loop_steps = [3, 5]
# batch_sizes = [1, 5, 15]
# meta_lrs = [0.001, 0.0005, 0.0001] # global update rate, global < local
# min_lr_coeff = [0.001]
# local_lrs = [0.1, 0.5] # local update rate
# reg_weights = [0.01]
# meta_wds = [0, 5e-4]
# bk-2: (5, 1, 0.001, 0.003, 0.1, 0.7, 20, 0.0007, 2020, True), epoch=4
# bk: (5, 1, 0.001, 0.001, 0.001, 0.5, 20, 0.0005, 2020, False), epoch=3
# db: (3, 1, 0.001, 0.001, 0.001, 0.5, 20, 0.0005, 2020, False), epoch=14
# ml: (3, 15, 0.001, 0.1, 0.1, 0.5, 20, 0, 2020, True)
inner_loop_steps = [5]
batch_sizes = [1]
meta_lrs = [0.001] # global update rate, global < local
local_lrs = [0.003] # local update rate
# ssl_weights = [1.]
# view_ratios = [0.7] 
ssl_weights = [0.1]
probs = [0.7] # keep prob samples, drop 1-prop samples
task_dims = [20]
meta_wds = [0.0007]
seeds = [2020]
use_schedulers = [True]
epochs = [4]
# seeds = [2020, 42]

hyper_ls = [inner_loop_steps, batch_sizes, meta_lrs, local_lrs, ssl_weights, probs, task_dims, meta_wds, seeds, use_schedulers, epochs]
hyper_ls_names = ['inner_loop_steps', 'batch_size', 'meta_lr', 'local_lr', 'ssl_weight', 'prob', 'task_dim', 'meta_wd', 'seed', 'use_scheduler', 'n_epoch']

# tdmeta_lrs = [0.001]
hyper_com = ','.join(hyper_ls_names)

print("="*5+" [robust experiment] "+"="*5)

param_settings = list(product(*hyper_ls))
for param in param_settings:
    for k, v in zip(hyper_ls_names, param):
        config_settings[k] = v
    
    config_settings['init_lr'] = config_settings['local_lr'] 
    # config_settings['min_lr'] = config_settings['meta_lr']
    # log
    config_settings['model_name'] = f'UASrec+adam_{args["dataset"]}_{param}'
    config_settings['train_log'] = f'{log_dir}/{config_settings["model_name"]}' # train and val loss log
    # config_settings['writer_log'] = f'{log_dir}/{config_settings["model_name"]}'  # hyper parameters log

    now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(config_settings['train_log']):    # 同一个模型，训练的第几次；
        config_settings['n_train_log'] = f"time_{now_time}_no_{len(os.listdir(config_settings['train_log']))}"

    seed_all(config_settings['seed'])
    print(config_settings)
    print(f'hyper param tuple ({hyper_com}): {param}')
    print(f'time:{now_time}')

    model = {
        'tdmeta': TaskDiffModel
    }[args['mode'].lower()](args['dataset'], emb_info=config, dataset_info=dataset_info, config_settings=config_settings)

    # train and val
    modeling = Modeling(model, config_settings=config_settings, mode=args['mode'])
    train_dict = {
        'tdmeta': modeling.train
    }
    _ = train_dict[args['mode'].lower()](train_dataset, test_dataset)

    # save param 
    model_s = deepcopy(model.state_dict())
    modeling_s = deepcopy(modeling.robust_loss.state_dict())
    torch.save(model_s, 'res/robust-experiment/models/{}-model.pth'.format(config_settings['model_name']))
    torch.save(modeling_s, 'res/robust-experiment/models/{}-modeling.pth'.format(config_settings['model_name']))

    probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    for prob in probs:
        print(f'prob:{prob}')
        modeling.model.load_state_dict(model_s)
        modeling.robust_loss.load_state_dict(modeling_s)
        seed_all(2020)
        mmae, mrmse, val_loss = modeling.test(test_dataset, 1, True, prob)
