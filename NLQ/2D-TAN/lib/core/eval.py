import json
import argparse
import numpy as np
import pickle as pkl
from terminaltables import AsciiTable

from core.config import config, update_config

def iou(pred, gt): # require pred and gt is numpy
    assert isinstance(pred, list) and isinstance(gt,list)
    pred_is_list = isinstance(pred[0],list)
    gt_is_list = isinstance(gt[0],list)
    if not pred_is_list: pred = [pred]
    if not gt_is_list: gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    #PATCH: sostituiamo i NaN in gt con 0.0
    gt = np.nan_to_num(gt, nan=0.0)
    inter_left = np.maximum(pred[:,0,None], gt[None,:,0])
    inter_right = np.minimum(pred[:,1,None], gt[None,:,1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:,0,None], gt[None,:,0])
    union_right = np.maximum(pred[:,1,None], gt[None,:,1])
    union = np.maximum(0.0, union_right - union_left)

    #overlap = 1.0 * inter / union
    #if not gt_is_list:
    #    overlap = overlap[:,0]
    #if not pred_is_list:
    #    overlap = overlap[0]
    #return overlap

    eps = 1e-8  # piccola costante per evitare divisione per zero
    overlap = inter / (union + eps)
    
    if not gt_is_list and overlap.ndim > 1:
        overlap = overlap[:, 0]
    if not pred_is_list and overlap.ndim > 1:
        overlap = overlap[0]
    
    return overlap

def rank(pred, gt):
    return pred.index(gt) + 1

def nms(dets, thresh=0.4, top_k=-1):
    """Pure Python NMS baseline."""
    if len(dets) == 0: return []
    order = np.arange(0,len(dets),1)
    dets = np.array(dets)
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    lengths = x2 - x1
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]

def eval(segments, data):
    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU,str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL,str) else [config.TEST.RECALL]

    eval_result = [[[] for _ in recalls] for _ in tious]
    max_recall = max(recalls)
    average_iou = []
    for seg, dat in zip(segments, data):
        overlap = iou(seg, [dat['times']])
        average_iou.append(np.mean(np.sort(overlap[0])[-3:]))

        for i,t in enumerate(tious):
            for j,r in enumerate(recalls):
                eval_result[i][j].append((overlap > t)[:r].any())
    eval_result = np.array(eval_result).mean(axis=-1)
    miou = np.mean(average_iou)


    return eval_result, miou

def eval_predictions(segments, data, verbose=True, merge_window=False):
    if merge_window:
        merge_seg = {}
        merge_data = {}
        for seg, dat in zip(segments, data):
            pair_id = dat['query_uid'] # + '_' + str(dat['query_idx'])
            if pair_id not in merge_seg.keys(): # new 
                merge_data[pair_id] = {
                    'video': dat['video'],
                    'duration': dat['clip_duration'],
                    'times': dat['times'],
                    'description': dat['description'],
                }
                merge_seg[pair_id] = []
            offset = dat['window'][0]
            merge_seg[pair_id].extend([[se[0]+offset, se[1]+offset, se[2]] for se in seg])
        segments, data = [], []
        for k in merge_seg.keys():
            # random.shuffle(merge_seg)
            segments.append(sorted(merge_seg[k], key=lambda x: x[2], reverse=True))
            data.append(merge_data[k])

    segments = [nms(seg, thresh=config.TEST.NMS_THRESH, top_k=5).tolist() for seg in segments]

    with open('results.pickle', 'wb') as f:
        pkl.dump({'prediction':segments,'data':data},f)

    eval_result, miou = eval(segments, data)
    if verbose:
        print(display_results(eval_result, miou, ''))

    return eval_result, miou


def display_results(eval_result, miou, title=None):
    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU,str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL,str) else [config.TEST.RECALL]

    display_data = [['Rank@{},mIoU@{}'.format(i,j) for i in recalls for j in tious]+['mIoU']]
    eval_result = eval_result*100
    miou = miou*100
    display_data.append(['{:.02f}'.format(eval_result[j][i]) for i in range(len(recalls)) for j in range(len(tious))]
                        +['{:.02f}'.format(miou)])
    table = AsciiTable(display_data, title)
    for i in range(len(tious)*len(recalls)):
        table.justify_columns[i] = 'center'
    return table.table


def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.verbose:
        config.VERBOSE = args.verbose

if __name__ == '__main__':
    args = parse_args()
    reset_config(config, args)
    train_data = json.load(open('/data/home2/hacker01/Data/DiDeMo/train_data.json', 'r'))
    val_data = json.load(open('/data/home2/hacker01/Data/DiDeMo/val_data.json', 'r'))

    moment_frequency_dict = {}
    for d in train_data:
        times = [t for t in d['times']]
        for time in times:
            time = tuple(time)
            if time not in moment_frequency_dict.keys():
                moment_frequency_dict[time] = 0
            moment_frequency_dict[time] += 1

    prior = sorted(moment_frequency_dict, key=moment_frequency_dict.get, reverse=True)
    prior = [list(item) for item in prior]
    prediction = [prior for d in val_data]

    eval_predictions(prediction, val_data)
