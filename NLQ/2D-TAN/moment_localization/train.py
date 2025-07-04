from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import pprint
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter #PER TENSORBOARD
import torch.optim as optim
from tqdm import tqdm
import datasets
import models
from core.config import config, update_config
from core.engine import Engine
from core.utils import AverageMeter
from core import eval
from core.utils import create_logger
import models.loss as loss
import math

# Abilita il debug CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

########### fix everything ###########
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.autograd.set_detect_anomaly(True)
def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--tag', help='tags shown in log', type=str)
    parser.add_argument('--debug', help='debug mode', action='store_true')
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATA_DIR = args.dataDir
    if args.modelDir:
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.tag:
        config.TAG = args.tag
    if args.debug:
        config.DEBUG = True
        print('=============== debug mode ==============')


if __name__ == '__main__':

    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir = create_logger(config, args.cfg, config.TAG)
    logger.info('\n'+pprint.pformat(args))
    logger.info('\n'+pprint.pformat(config))

    # SEZIONE AGGIUNTA PER IL TENSORBOARD ------------------------------
    
    #writer = None
    #if config.LOG_DIR:
    #    log_dir = os.path.join(config.LOG_DIR, config.TAG or 'default')
    #    os.makedirs(log_dir, exist_ok=True)
    #    print(f"Writing TensorBoard logs to: {log_dir}")
    #    writer = SummaryWriter(log_dir=log_dir)

    writer = None
    log_dir = os.path.join('runs', config.TAG or 'default')
    os.makedirs(log_dir, exist_ok=True)
    print(f"Writing TensorBoard logs to: {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)

    # ------------------------------------------------------------------
    
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    dataset_name = config.DATASET.NAME
    model_name = config.MODEL.NAME

    train_dataset = getattr(datasets, dataset_name)('train')
    if config.TEST.EVAL_TRAIN:
        eval_train_dataset = getattr(datasets, dataset_name)('train')
    if not config.DATASET.NO_VAL:
        val_dataset = getattr(datasets, dataset_name)('val')
#    test_dataset = getattr(datasets, dataset_name)('test')

    model = getattr(models, model_name)()
    if config.MODEL.CHECKPOINT and config.TRAIN.CONTINUE:
        model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(),lr=config.TRAIN.LR, betas=(0.9, 0.999), weight_decay=config.TRAIN.WEIGHT_DECAY)
    # optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.LR, momentum=0.9, weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.TRAIN.FACTOR, patience=config.TRAIN.PATIENCE, verbose=config.VERBOSE)


    def iterator(split):
        if split == 'train':
            dataloader = DataLoader(train_dataset,
                                    batch_size=config.TRAIN.BATCH_SIZE,
                                    shuffle=config.TRAIN.SHUFFLE,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn)
        elif split == 'val':
            dataloader = DataLoader(val_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn)
#        elif split == 'test':
#            dataloader = DataLoader(test_dataset,
#                                    batch_size=config.TEST.BATCH_SIZE,
#                                    shuffle=False,
#                                    num_workers=config.WORKERS,
#                                    pin_memory=False,
#                                    collate_fn=datasets.collate_fn)
        elif split == 'train_no_shuffle':
            dataloader = DataLoader(eval_train_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn)
        else:
            raise NotImplementedError

        return dataloader

    def network(sample):
        anno_idxs = sample['batch_anno_idxs']
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        visual_input = sample['batch_vis_input'].cuda()
        map_gt = sample['batch_map_gt'].cuda()
        duration = sample['batch_duration']

        prediction, map_mask = model(textual_input, textual_mask, visual_input)   

        #print("prediction shape:", prediction.shape)
        #print("prediction dtype:", prediction.dtype)
        #print("prediction contains NaN:", torch.isnan(prediction).any())  # Controlla NaN
        #print("prediction contains Inf:", torch.isinf(prediction).any())  # Controlla infiniti

        #print("map_mask shape:", map_mask.shape)
        #print("map_mask dtype:", map_mask.dtype)
        #print("map_mask contains NaN:", torch.isnan(map_mask).any())
        #print("map_mask contains Inf:", torch.isinf(map_mask).any())

        #print("map_gt shape:", map_gt.shape)
        #print("map_gt dtype:", map_gt.dtype)
        #print("map_gt contains NaN:", torch.isnan(map_gt).any())
        #print("map_gt contains Inf:", torch.isinf(map_gt).any())

        # DEBUG: controlliamo che i target siano nel range [0,1]
        #print("map_gt min:", map_gt.min().item(), "max:", map_gt.max().item())
        #assert (map_gt >= 0).all() and (map_gt <= 1).all(), "Errore: map_gt fuori dal range [0,1]"

        # ===== DEBUG: Controllo NaN in map_gt =====
        # if torch.isnan(map_gt).any():
        #     print("==== ERRORE: map_gt contiene NaN ====")
        #     print("sample keys:", sample.keys())
        #     print("batch_anno_idxs:", sample['batch_anno_idxs'])
        #     print("batch_duration:", sample['batch_duration'])
        #     print("map_gt:", map_gt)
        #     # volendo puoi stampare anche visual_input.shape ecc.

        loss_value, joint_prob = getattr(loss, config.LOSS.NAME)(prediction, map_mask, map_gt, config.LOSS.PARAMS)

        sorted_times = None if model.training else get_proposal_results(joint_prob, duration)

        return loss_value, sorted_times

    def get_proposal_results(scores, durations):
        # assume all valid scores are larger than one
        out_sorted_times = []
        for score, duration in zip(scores, durations):
            T = score.shape[-1]
            score_cpu = score.cpu().detach().numpy()
            sorted_indexs = np.dstack(np.unravel_index(np.argsort(score_cpu.ravel())[::-1], (T, T))).tolist()
            sorted_indexs = np.array([item for item in sorted_indexs[0] if item[0] <= item[1]]).astype(float)
            sorted_scores = np.array([score_cpu[0, int(x[0]),int(x[1])] for x in sorted_indexs])

            sorted_indexs[:,1] = sorted_indexs[:,1] + 1
            sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
            target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
            sorted_time = (sorted_indexs.float() / target_size * duration).tolist()
            out_sorted_times.append([[t[0], t[1], s] for t, s in zip(sorted_time, sorted_scores)])

        return out_sorted_times

    def on_start(state):
        state['loss_meter'] = AverageMeter()
        state['test_interval'] = int(len(train_dataset)/config.TRAIN.BATCH_SIZE*config.TEST.INTERVAL)
        state['t'] = 1
        model.train()
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=state['test_interval'])

    def on_forward(state):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        state['loss_meter'].update(state['loss'].item(), 1)

    def on_update(state):  # Save All
        if config.VERBOSE:
            state['progress_bar'].update(1)
    
        # Test and log at test_interval
        if state['t'] % state['test_interval'] == 0:
            state['test_step'] = state['t']
            model.eval()
    
            if config.VERBOSE:
                state['progress_bar'].close()
    
            loss_message = '\niter: {} train loss {:.4f}'.format(state['t'], state['loss_meter'].avg)
            table_message = ''
    
            if config.TEST.EVAL_TRAIN:
                train_state = engine.test(network, iterator('train_no_shuffle'), 'train')
                train_table = eval.display_results(
                    train_state['Rank@N,mIoU@M'],
                    train_state['miou'],
                    'performance on training set'
                )
                table_message += '\n' + train_table
    
            if not config.DATASET.NO_VAL:
                val_state = engine.test(network, iterator('val'), 'val')
    
                # Libera la memoria
                torch.cuda.empty_cache()
                import gc
                gc.collect()
    
                # Log validation loss to TensorBoard
                if writer is not None:
                    writer.add_scalar('Loss/val', val_state['loss_meter'].avg, global_step=state['t'])
    
                state['scheduler'].step(-val_state['loss_meter'].avg)
    
                loss_message += ' val loss {:.4f}'.format(val_state['loss_meter'].avg)
                val_state['loss_meter'].reset()
    
                val_table = eval.display_results(
                    val_state['Rank@N,mIoU@M'],
                    val_state['miou'],
                    'performance on validation set'
                )
                table_message += '\n' + val_table
    
            # Salvataggio del modello
            saved_model_filename = os.path.join(
                config.MODEL_DIR,
                '{}/{}/iter{:06d}-{:.4f}-{:.4f}.pkl'.format(
                    dataset_name,
                    model_name + '_' + config.DATASET.VIS_INPUT_TYPE,
                    state['t'],
                    train_state['Rank@N,mIoU@M'][0, 0],
                    train_state['Rank@N,mIoU@M'][0, 1]
                )
            )
    
            # Crea le directory se mancano
            rootfolder1 = os.path.dirname(saved_model_filename)
            rootfolder2 = os.path.dirname(rootfolder1)
            rootfolder3 = os.path.dirname(rootfolder2)
    
            for folder in [rootfolder3, rootfolder2, rootfolder1]:
                if not os.path.exists(folder):
                    print('Make directory %s ...' % folder)
                    os.mkdir(folder)
    
            # Salva lo stato del modello
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), saved_model_filename)
            else:
                torch.save(model.state_dict(), saved_model_filename)
    
            if config.VERBOSE:
                state['progress_bar'] = tqdm(total=state['test_interval'])
    
            model.train()
            state['loss_meter'].reset()

    def on_end(state):
        if config.VERBOSE:
            state['progress_bar'].close()
            
       # SEZIONE AGGIUNTA PER IL TENSORBOARD ------------------------------
        print("Training completed.")
        if writer:
            writer.close()
        import sys
        sys.exit(0)
       # ------------------------------------------------------------------

    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []
        if config.VERBOSE:
            if state['split'] == 'train':
                state['progress_bar'] = tqdm(total=math.ceil(len(train_dataset)/config.TEST.BATCH_SIZE))
            elif state['split'] == 'val':
                state['progress_bar'] = tqdm(total=math.ceil(len(val_dataset)/config.TEST.BATCH_SIZE))
#            elif state['split'] == 'test':
#                state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset)/config.TEST.BATCH_SIZE))
            else:
                raise NotImplementedError

    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        state['loss_meter'].update(state['loss'].item(), 1)

        min_idx = min(state['sample']['batch_anno_idxs'])
        batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
        sorted_segments = [state['output'][i] for i in batch_indexs]
        state['sorted_segments_list'].extend(sorted_segments)

    def on_test_end(state):
        annotations = state['iterator'].dataset.annotations
        merge = (state['split'] != 'train')
        state['Rank@N,mIoU@M'], state['miou'] = eval.eval_predictions(state['sorted_segments_list'], annotations, verbose=False, merge_window=merge)
      
        torch.cuda.empty_cache() # SVUOTA RAM
        import gc # SVUOTA RAM
        gc.collect() # SVUOTA RAM
       
        if config.VERBOSE:
            state['progress_bar'].close()

        # SEZIONE AGGIUNTA PER IL TENSORBOARD ------------------------------
        if writer and state['split'] == 'val':
            #writer.add_scalar('Validation/Loss', state['loss_meter'].val, global_step=state['t'])
            #writer.add_scalar('Validation/Loss', state['loss_meter'].val, global_step=state.get('test_step',state['t']))
            writer.add_scalar('Validation/mIoU', state['miou'], global_step=state['t'])
        # ------------------------------------------------------------------
    
    engine = Engine()
    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_update'] = on_update
    engine.hooks['on_end'] = on_end
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.train(network,
                 iterator('train'),
                 maxepoch=config.TRAIN.MAX_EPOCH,
                 optimizer=optimizer,
                 scheduler=scheduler)
