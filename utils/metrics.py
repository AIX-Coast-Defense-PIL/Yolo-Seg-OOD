import json
import os

from utils.converter import convert_annForm_bbox2img

def compute_iou(bbox1, bbox2):
    intersection_x_length = max(min(bbox1['xbr'], bbox2['xbr']) - max(bbox1['xtl'], bbox2['xtl']), 0)
    intersection_y_length = max(min(bbox1['ybr'], bbox2['ybr']) - max(bbox1['ytl'], bbox2['ytl']), 0)

    area1 = (bbox1['xbr'] - bbox1['xtl'])*(bbox1['ybr'] - bbox1['ytl'])
    area2 = (bbox2['xbr'] - bbox2['xtl'])*(bbox2['ybr'] - bbox2['ytl'])
    intersect_area = intersection_x_length*intersection_y_length
    whole_area = area1 + area2 - intersect_area
    
    iou = intersect_area / whole_area
    return iou

def calc_iou_performance(pred_infos, ann_infos, ood_threshold=None, conf_threshold=None):
    img_ids = set(ann_infos.keys()).union(pred_infos.keys())
    TP_per_imgs = [0] * len(img_ids)
    True_per_imgs = [0] * len(img_ids) # num of annotation true
    Pos_per_imgs = [0] * len(img_ids) # num of prediction true

    for img_idx, img_id in enumerate(img_ids):
        if img_id in pred_infos.keys():
            pred_infos[img_id]['bboxes'] = [bbox for bbox in pred_infos[img_id]['bboxes'] if bbox['label'] == 0]
        if (ood_threshold) and (img_id in pred_infos.keys()):
            pred_infos[img_id]['bboxes'] = [bbox for bbox in pred_infos[img_id]['bboxes'] if bbox['ood_score'] > ood_threshold]
        if (conf_threshold) and (img_id in pred_infos.keys()):
            pred_infos[img_id]['bboxes'] = [bbox for bbox in pred_infos[img_id]['bboxes'] if bbox['score'] > conf_threshold]
        True_per_imgs[img_idx] = len(ann_infos[img_id]['bboxes']) if img_id in ann_infos.keys() else 0
        Pos_per_imgs[img_idx] = len(pred_infos[img_id]['bboxes']) if img_id in pred_infos.keys() else 0

        if True_per_imgs[img_idx] != 0 and Pos_per_imgs[img_idx] != 0:
            for ann_bbox in ann_infos[img_id]['bboxes']:
                for pred_bbox in pred_infos[img_id]['bboxes']:
                    iou = compute_iou(ann_bbox, pred_bbox)
                    pred_label = pred_bbox['label'] if 'label' in pred_bbox.keys() else 0
                    if iou > 0.5 and pred_label == ann_bbox['label']:
                        TP_per_imgs[img_idx] += 1
                        break

    precision = round(sum(TP_per_imgs) / (sum(Pos_per_imgs) + 1e-15), 4)
    recall = round(sum(TP_per_imgs) / (sum(True_per_imgs) + 1e-15), 4)

    
    return {
        'precision':precision,
        'recall':recall
    }


if __name__=='__main__':
    ann_path = 'datasets/custom102/annotations/all.json'
    pred_path = 'datasets/custom102/yolov7_preds/yolov7_predictions.json'

    with open(ann_path, "r") as json_file:
        ann_info = json.load(json_file)
    with open(pred_path, "r") as json_file:
        pred_info = convert_annForm_bbox2img(json.load(json_file))
    
    print(calc_iou_performance(pred_info, ann_info, conf_threshold=0.05))