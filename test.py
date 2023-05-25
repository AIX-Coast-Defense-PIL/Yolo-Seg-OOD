
import os
import torch

import os, sys

root_path = os.getcwd()
root_path = root_path if 'Yolo-Seg-OOD' in root_path else os.path.join(root_path, 'Yolo-Seg-OOD')
sys.path.append(root_path)
os.chdir(root_path)

sys.path.append(os.path.join(root_path, 'yolov7'))
from yolov7.detect import detect, set_yolo_args

sys.path.append(os.path.join(root_path, 'ood'))
from ood.args_loader import get_args
from ood.main import load_model
from ood.yolo_ood_classifier import yolo_ood_classifier


from utils.file_processing import load_file

def test_whole():
    ood_args = get_args(root_path)
    ood_backbone = load_model(ood_args)     # load backbone model
    cluster = load_file(ood_args.cluster_path)
    thresholds_in_rate = load_file(ood_args.threshold_path)

    yolo_args = set_yolo_args(dataset=ood_args.test_data, datatype='image')
    ood_classifier = {'model' : ood_backbone, 'cluster':cluster, 'pred_func': yolo_ood_classifier, 'thresholds':thresholds_in_rate}
    
    with torch.no_grad():
        detect(yolo_args, second_classifier=ood_classifier)



if __name__=='__main__':
    test_whole()