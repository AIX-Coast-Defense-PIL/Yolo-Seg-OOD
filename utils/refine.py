
import json
import os

from converter import convert_annForm_bbox2img, convert_annForm_img2bbox
from metrics import compute_iou

def remove_overlap_bbox(img_infos):
    new_img_infos = img_infos
    for img_id in img_infos.keys():
        bboxes = [bbox for bbox in img_infos[img_id]['bboxes'] if bbox['score']]
        flags = [0]*len(bboxes)
        for i in range(len(bboxes)):
            for j in range(i+1, len(bboxes)):
                if flags[i]==0 and flags[j]==0:
                    iou = compute_iou(bboxes[i], bboxes[j])
                    if iou > 0.5:
                        remove_flag = i if bboxes[i]['score'] < bboxes[j]['score'] else j
                        flags[remove_flag] = 1
                else:
                    continue
        new_img_infos[img_id]['bboxes'] = [bbox for idx, bbox in enumerate(bboxes) if flags[idx] == 0]
    
    return new_img_infos

if __name__ == '__main__':
    dataset = 'custom102'
    pred_path = os.path.join('datasets', dataset, 'yolov7_preds/yolov7_predictions.json')
    with open(pred_path, "r") as json_file:
        pred_infos = convert_annForm_bbox2img(json.load(json_file))
    
    refined_pred_infos = remove_overlap_bbox(pred_infos)

    refined_pred_infos = convert_annForm_img2bbox(refined_pred_infos)
    save_path = os.path.join('datasets', dataset, 'yolov7_preds/yolov7_preds_refined.json')
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(refined_pred_infos, file, indent="\t")