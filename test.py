
import os
import torch

import os, sys

root_path = os.getcwd()

yolo_path = os.path.join(root_path, 'yolov7')
sys.path.append(yolo_path)
os.chdir(yolo_path)
from yolov7.detect import detect, set_yolo_args

ood_path = os.path.join(root_path, 'ood')
sys.path.append(ood_path)
os.chdir(ood_path)
from ood.args_loader import get_args
from ood.main import load_model
from ood.yolo_ood_classifier import yolo_ood_classifier


from utils.file_processing import load_file

def test_whole():
    ood_args = get_args(root_path)
    ood_backbone = load_model(ood_args)     # load backbone model
    cluster = load_file(ood_args.cluster_path)
    thresholds_in_rate = load_file(ood_args.threshold_path)

    os.chdir(yolo_path)
    yolo_args = set_yolo_args(ood_args.test_data)
    ood_classifier = {'model' : ood_backbone, 'cluster':cluster, 'pred_func': yolo_ood_classifier, 'thresholds':thresholds_in_rate}
    with torch.no_grad():
        detect(yolo_args, second_classifier=ood_classifier)
    return 0



if __name__=='__main__':
    test_whole()