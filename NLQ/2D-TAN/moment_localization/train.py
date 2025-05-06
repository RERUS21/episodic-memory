
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
import datasets
import models
from core.config import config, update_config
from core.engine import Engine
from core.utils import AverageMeter
from core import eval
from core.utils import create_logger
import models.loss as loss

# Debug CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Set seeds
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Run validation only')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return parser.parse_args()

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

def iterator(split, dataset):
    return DataLoader(dataset,
                      batch_size=config.TEST.BATCH_SIZE,
                      shuffle=False,
                      num_workers=config.WORKERS,
                      pin_memory=False,
                      collate_fn=datasets.collate_fn)

def network(sample):
    textual_input = sample['batch_word_vectors'].cuda()
    textual_mask = sample['batch_txt_mask'].cuda()
    visual_input = sample['batch_vis_input'].cuda()
    map_gt = sample['batch_map_gt'].cuda()
    duration = sample['batch_duration']
    prediction, map_mask = model(textual_input, textual_mask, visual_input)
    loss_value, joint_prob = getattr(loss, config.LOSS.NAME)(prediction, map_mask, map_gt, config.LOSS.PARAMS)
    sorted_times = get_proposal_results(joint_prob, duration)
    return loss_value, sorted_times

def get_proposal_results(scores, durations):
    out_sorted_times = []
    for score, duration in zip(scores, durations):
        T = score.shape[-1]
        score_cpu = score.cpu().detach().numpy()
        sorted_indexs = np.dstack(np.unravel_index(np.argsort(score_cpu.ravel())[::-1], (T, T))).tolist()
        sorted_indexs = np.array([item for item in sorted_indexs[0] if item[0] <= item[1]]).astype(float)
        sorted_scores = np.array([score_cpu[0, int(x[0]), int(x[1])] for x in sorted_indexs])
        sorted_indexs[:,1] += 1
        sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
        target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
        sorted_time = (sorted_indexs.float() / target_size * duration).tolist()
        out_sorted_times.append([[t[0], t[1], s] for t, s in zip(sorted_time, sorted_scores)])
    return out_sorted_times

if __name__ == '__main__':
    args = parse_args()
    reset_config(config, args)
    logger, _ = create_logger(config, args.cfg, config.TAG)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    dataset_name = config.DATASET.NAME
    model_name = config.MODEL.NAME
    val_dataset = getattr(datasets, dataset_name)('val')

    model = getattr(models, model_name)()
    if config.MODEL.CHECKPOINT:
        model.load_state_dict(torch.load(config.MODEL.CHECKPOINT))
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()

    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=len(iterator('val', val_dataset)))

    def on_test_forward(state):
        state['loss_meter'].update(state['loss'].item(), 1)
        min_idx = min(state['sample']['batch_anno_idxs'])
        batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
        sorted_segments = [state['output'][i] for i in batch_indexs]
        state['sorted_segments_list'].extend(sorted_segments)

    def on_test_end(state):
        annotations = state['iterator'].dataset.annotations
        state['Rank@N,mIoU@M'], state['miou'] = eval.eval_predictions(
            state['sorted_segments_list'], annotations, verbose=True, merge_window=True)
        print("Validation mIoU:", state['miou'])

    engine = Engine()
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end

    engine.test(network, iterator('val', val_dataset), 'val')
