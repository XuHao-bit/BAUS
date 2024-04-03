import argparse
import datetime
import os
from itertools import product

from TaskDiffModel import TaskDiffModel, DeepModel
from Trainer import Modeling
from DataLoading import *
from utils import config, dataset_info, config_settings, seed_all

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movielens')
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--mode', type=str, default='tdmeta')
parser.add_argument('--save', type=str, default='False')
parser.add_argument('--use_writer', type=boolean_string, default=False)
parser.add_argument('--save_name', type=str, default=f'model_{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--use_gen_hypr', type=str, default='True')
parser.add_argument('--use_lea_hypr', type=str, default='True')
parser.add_argument('--use_ginfo_only', type=str, default='False')   # 代表generating adaptive hyper-parameters的时候，仅使用avg grad 以及avg weight信息；
parser.add_argument('--use_tinfo_only', type=str, default='False')   # 代表generating adaptive hyper-parameters的时候，仅使用task emb信息；
parser.add_argument('--nshot', type=int, default=15)
parser.add_argument('--use_tmem', type=boolean_string, default=True) # 

# adaptive weighted loss
parser.add_argument('--use_aloss', type=boolean_string, default=False)
# adaptive regression loss
parser.add_argument('--use_aregloss', type=boolean_string, default=True) # 
parser.add_argument('--use_mrgloss', type=boolean_string, default=False) # use meta rgloss
parser.add_argument('--use_aregloss_combine', type=boolean_string, default=False)
parser.add_argument('--use_adap_rgloss', type=boolean_string, default=False)
# use ssl encoder
parser.add_argument('--use_sslencoder', type=boolean_string, default=True) # 
parser.add_argument('--use_ssl', type=boolean_string, default=True) #
parser.add_argument('--aug1', type=boolean_string, default=True) #
parser.add_argument('--aug2', type=boolean_string, default=False)
parser.add_argument('--prob', type=float, default=0.3)
# use_grad_clip
parser.add_argument('--use_grad_clip', type=boolean_string, default=True)

parser.add_argument('--use_earlystop', type=boolean_string, default=True)
parser.add_argument('--early_step', type=int, default=10)
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
# bk: 0.001, 0.001
# db: 0.0001, 0.1
# ours:(3, 15, 0.001, 0.1, 0.1, 0.5, 20, 0, 2020, True),
inner_loop_steps = [5]
batch_sizes = [15]
meta_lrs = [0.001] # global update rate, global < local
local_lrs = [0.1] # local update rate
# ssl_weights = [1.]
# view_ratios = [0.7] 
ssl_weights = [0.1]
probs = [0.5] # keep prob samples, drop 1-prop samples
task_dims = [20]
meta_wds = [0]
seeds = [2020]
use_schedulers = [True]
# seeds = [2020, 42]

hyper_ls = [inner_loop_steps, batch_sizes, meta_lrs, local_lrs, ssl_weights, probs, task_dims, meta_wds, seeds, use_schedulers]
hyper_ls_names = ['inner_loop_steps', 'batch_size', 'meta_lr', 'local_lr', 'ssl_weight', 'prob', 'task_dim', 'meta_wd', 'seed', 'use_scheduler']

# start experiment
for hyper_name, hyper_list in zip(hyper_ls_names, hyper_ls):
    print(f'{hyper_name}: {hyper_list}')

# tdmeta_lrs = [0.001]
hyper_com = ','.join(hyper_ls_names)

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
    best_model = train_dict[args['mode'].lower()](train_dataset, test_dataset)

    # save param 
    if args['save'] == "True":
        torch.save(best_model, 'saved_models/{}.pth'.format(args['save_name']))
