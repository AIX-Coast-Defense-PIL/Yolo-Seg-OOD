import torch
import torchvision
import torch.backends.cudnn as cudnn

import os
import numpy as np
import random
import time
import pickle
import json

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import sys 
work_path = os.path.join(os.getcwd(), os.pardir) if '/ood' in os.getcwd() else os.getcwd()
work_path = work_path if 'Yolo-Seg-OOD' in work_path else os.path.join(work_path, 'Yolo-Seg-OOD')
sys.path.append(work_path)

from utils.converter import convert_annForm_bbox2img
from utils.metrics import calc_iou_performance
from utils.file_processing import save_file, load_file

from args_loader import get_args
from data_loader import get_train_loader, get_test_loader
from ood_scores import calc_distance_score

import warnings
warnings.filterwarnings('ignore')


def seed_setting(num):  # 이해, 정리 x
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    np.random.seed(num)
    random.seed(num)
    # cudnn.benchmark = False
    cudnn.deterministic = True

def load_model(args):
    if args.backbone_arch == 'resnet50':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        # model = torchvision.models.resnet50(pretrained=True)
    elif args.backbone_arch == 'resnet50_tune':
        ckpt = torch.load(args.backbone_weight).state_dict()
        model = torchvision.models.resnet50()
        model.load_state_dict(ckpt)
    else:
        print('wrong backbone_arch')
    
    state_dict = {k:model.state_dict()[k] for k in model.state_dict() if not k in ['fc.bias', 'fc.weight']}
    model.load_state_dict(state_dict, strict=False)

    return model

def get_bbox_infos(args, mode, feat_path, model, cluster=None, threshold=None):
    if mode == 'train':
        if os.path.exists(feat_path):
            feat_infos = load_file(feat_path)
            print(f'Loaded the features from {feat_path}')
            return feat_infos, {}
        data_dir_paths = [args.train_data]
        if args.add_train_data not in [None, 'None']:
            data_dir_paths.append(args.add_train_data)
            print(f'Using additional train dataset! ({args.add_train_data})')
        data_loader = get_train_loader(data_dir_paths, batch_size=args.train_bs)
    
    elif mode == 'test':
        data_loader = get_test_loader(args.test_data, batch_size=1)

    model.to(device)
    model.eval()

    feat_infos = []
    pred_infos = []
    start_time = time.time()
    with torch.no_grad():
        print('start bbox info extracting...')
        for batch_idx, (images, img_names, labels, bboxes) in enumerate(data_loader):
            start_batch_time = time.time()

            images = images.to(device)
            feats = model(images)
            feats = feats.data.cpu().numpy()
            
            if mode == 'train':
                for idx in range(len(labels)):
                    feat_infos.append({'img_name':img_names[idx], 'label':labels[idx], 'bbox':[int(bboxes[coordicate_idx][idx]) for coordicate_idx in range(4)], 'feat':feats[idx]})

            elif mode == 'test':
                ood_scores = calc_distance_score(cluster, feats, args.score_matrix, 'test', args.cov_matrix_path)
                for idx in range(len(labels)):
                    pred_infos.append({'img_name':img_names[idx], 'label':0 if ood_scores[idx] > threshold else 1, 'bbox':[int(bboxes[coordicate_idx][idx]) for coordicate_idx in range(4)], 'ood_score':ood_scores[idx]})

            if batch_idx % 20 == 0:
                batch_time = time.time() - start_batch_time
                print(f'[{batch_idx}/{len(data_loader)} batch] {batch_time:.04f} sec per batch')
            
            # memory management
            del feats
            del labels
            torch.cuda.empty_cache()
    total_time = time.time() - start_time
    
    if mode == 'train':
        save_file(feat_infos, feat_path)
        time_per_img = total_time / len(feat_infos)
        print(f'[Saved the features] path : {feat_path}, total time : {total_time:.4f}, time : {time_per_img:.4f} s/f')
    
    elif mode == 'test': 
        save_file(pred_infos, args.score_path)
        time_per_img = total_time / len(pred_infos)
        print(f'[Saved the scores] path : {args.score_path}, total time : {total_time:.4f}, time : {time_per_img:.4f} s/f')
    
    return feat_infos, pred_infos
        
def train_cluster(args, feats):

    start_time = time.time()
    if args.cluster == 'kmeans':
        cluster = KMeans(n_clusters=args.num_cluster, random_state=0, n_init='auto').fit(feats)
    elif args.cluster == 'GM':
        cluster = GaussianMixture(n_components=args.num_cluster, max_iter=500, random_state=0).fit(feats)
    
    total_time = time.time() - start_time
    print(f'time to fitting cluster : {total_time:.4f} s')

    # save cluster model
    save_file(cluster, args.cluster_path)
    print('[saved cluster model] path :', args.cluster_path)

    return cluster

def eval_ood_score(args, cluster, feat_infos, threshold):
    print(f'ood threshold : {threshold}')
    feats = [feat_info.pop('feat') for feat_info in feat_infos]
    ood_scores = calc_distance_score(cluster, feats, args.score_matrix, 'test', args.cov_matrix_path)

    pred_infos = []

    for idx, feat_info in enumerate(feat_infos):
        feat_info.update({'label' : 0 if ood_scores[idx] > threshold else 1, 'ood_score':ood_scores[idx]}) # 0 : unknown(not in training dataset)
        pred_infos.append(feat_info)

    save_file(pred_infos, args.score_path)

    return pred_infos

def ood_train(args):
    print('start ood training...')
    backbone = load_model(args)     # load backbone model
    feat_infos, _ = get_bbox_infos(args, 'train', args.train_feat_path, backbone)
    feats = [feat_info['feat'] for feat_info in feat_infos]

    cluster = train_cluster(args, feats)

    # find thresholds
    ood_scores = calc_distance_score(cluster, feats, args.score_matrix, 'train', args.cov_matrix_path)
    ood_scores = sorted(ood_scores, reverse=True)
    thresholds_in_rate = {str(in_rate)+"%" : round(ood_scores[round((1-(in_rate*0.01)) * len(ood_scores))], 2) for in_rate in range(1, 101, 1)}
    save_file(thresholds_in_rate, args.threshold_path)
    print('[saved ood thresholds] path :', args.threshold_path)

    return thresholds_in_rate

def ood_test(args, threshold = 10):
    print('start ood testing...')
    backbone = load_model(args)     # load backbone model
    cluster = load_file(args.cluster_path)

    _, ood_infos = get_bbox_infos(args, 'test', args.test_feat_path, backbone, cluster, threshold)

    # call and set format of pred_infos & ann_infos
    ood_infos = convert_annForm_bbox2img(ood_infos)
    ann_file_path = os.path.join(args.test_data, 'annotations/all.json')
    ann_infos = load_file(ann_file_path)
    
    # calc precision and recall
    performance = calc_iou_performance(ood_infos, ann_infos)

    return performance

if __name__=='__main__':
    args = get_args(work_path)
    device = torch.device(f'cuda:{args.gpu}')
    seed_setting(0)

    try:
        thresholds_in_rate = load_file(args.threshold_path)
        print(f'Loaded thresholds from {args.threshold_path}')
    except FileNotFoundError:
        thresholds_in_rate = ood_train(args)
    print(f'ood thresholds : {thresholds_in_rate}')

    if args.mode in ['both', 'test']:
        threshold = thresholds_in_rate['18%']
        performance = ood_test(args, threshold = threshold)
        print(performance)