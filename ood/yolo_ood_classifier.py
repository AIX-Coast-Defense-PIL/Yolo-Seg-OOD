import cv2
import numpy as np
import torch
import os

from yolov7.yoloUtils.general import xyxy2xywh, xywh2xyxy, scale_coords
from ood.ood_scores import calc_distance_score
from ood.args_loader import get_args

def yolo_ood_classifier(yolo_preds, filter_masks, fe_model, cluster, thresholds, img, im0):
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0

    for b_idx, (yolo_pred, filter_mask) in enumerate(zip(yolo_preds, filter_masks)):  # per image
        if yolo_pred is not None and len(yolo_pred):
            yolo_pred = yolo_pred.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(yolo_pred[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            yolo_pred[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], yolo_pred[:, :4], im0[b_idx].shape)

            # Classes
            ims = []
            unfiltered_idx = []
            for idx, bbox in enumerate(yolo_pred):  # per item
                # bbox value 확인 및 img, im0, filter_mask shape 확인하기 ###########################
                if not filter_mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].all():  # yolo predictions(bbox) Filtering
                    cutout = im0[b_idx][int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    im = cv2.resize(cutout, (224, 224))  # BGR
                    # cv2.imwrite('test%i.jpg' % j, cutout)

                    im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                    im /= 255.0  # 0 - 255 to 0.0 - 1.0
                    ims.append(im)
                    unfiltered_idx.append(idx)
            
            pred_cls = torch.Tensor([2]*len(yolo_preds[b_idx]))
            if len(ims):
                # ood classify
                feats = fe_model(torch.Tensor(ims).to(yolo_pred.device))  # classifier prediction
                feats = feats.data.cpu().numpy()
                ood_args = get_args(os.path.join(os.getcwd(), '..'))
                ood_scores = calc_distance_score(cluster, feats, ood_args.score_matrix, 'test', ood_args.cov_matrix_path)
                threshold = thresholds['87%']
                pred_cls[unfiltered_idx] = torch.Tensor([0 if ood_score > threshold else 1 for ood_score in ood_scores])
            
            # x[i][:, 4] = torch.Tensor(ood_scores) # change conf to ood_scores
            yolo_preds[b_idx][:, 5] = pred_cls

    return yolo_preds