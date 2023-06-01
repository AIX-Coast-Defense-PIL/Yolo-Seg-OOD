
import os
import torch

import os, sys

root_path = os.getcwd()
root_path = root_path if 'Yolo-Seg-OOD' in root_path else os.path.join(root_path, 'Yolo-Seg-OOD')
sys.path.append(root_path)
os.chdir(root_path)

sys.path.append(os.path.join(root_path, 'yolov7'))
from yolov7.detect import detect, get_args

sys.path.append(os.path.join(root_path, 'ood'))
from ood.main import load_model
from ood.yolo_ood_classifier import yolo_ood_classifier

from utils.file_processing import load_file


def test_whole():
    args = get_args()
    ood_backbone = load_model(args)     # load backbone model
    cluster = load_file(args.cluster_path)
    thresholds_in_rate = load_file(args.threshold_path)

    ood_classifier = {'backbone_model' : ood_backbone, 'cluster':cluster, 'pred_func': yolo_ood_classifier, 'thresholds':thresholds_in_rate}
    
    with torch.no_grad():
        detect(args, second_classifier=ood_classifier)



if __name__=='__main__':
    test_whole()