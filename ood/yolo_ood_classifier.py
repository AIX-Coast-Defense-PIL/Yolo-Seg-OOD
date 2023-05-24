import cv2
import numpy as np
import torch
import os

from yolov7.yoloUtils.general import xyxy2xywh, xywh2xyxy, scale_coords
from ood.ood_scores import calc_distance_score
from ood.args_loader import get_args

def yolo_ood_classifier(x, model, cluster, thresholds, img, im0):
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0

    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            # ood classify
            feats = model(torch.Tensor(ims).to(d.device))  # classifier prediction
            feats = feats.data.cpu().numpy()
            ood_args = get_args(os.path.join(os.getcwd(), '..'))
            ood_scores = calc_distance_score(cluster, feats, ood_args.score_matrix, 'test', ood_args.cov_matrix_path)
            threshold = thresholds['87%']
            pred_cls2 = [0 if ood_score > threshold else 1 for ood_score in ood_scores]

            # x[i][:, 4] = torch.Tensor(ood_scores) # change conf to ood_scores
            x[i][:, 5] = torch.Tensor(pred_cls2)
    return x