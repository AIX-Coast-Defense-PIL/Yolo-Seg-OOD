import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import json
import platform

import sys 
root_path = os.path.join(os.getcwd(), os.pardir) if '/yolov7' in os.getcwd() else os.getcwd()
root_path = root_path if 'Yolo-Seg-OOD' in root_path else os.path.join(root_path, 'Yolo-Seg-OOD')
sys.path.append(root_path)

from models.experimental import attempt_load
from yoloUtils.datasets import LoadStreams, LoadImages
from yoloUtils.general import check_img_size, check_imshow, non_max_suppression, \
    scale_coords, strip_optimizer, set_logging, increment_path
from yoloUtils.plots import plot_one_box
from yoloUtils.torch_utils import select_device, time_synchronized, TracedModel

from utils.converter import convert_txt_to_json
from utils.file_processing import load_file
from utils.metrics import calc_iou_performance

import seg.model.models as seg_models
from seg.model.utils import load_weights
from seg.model.inference import Predictor

import warnings
warnings.filterwarnings(action='ignore')
from PIL import Image, ImageDraw
import platform

# Colors corresponding to each segmentation class # (1 = water, 2 = sky, 0 = obstacles)
BIN_COLORS = np.array([
    [0, 0, 0],
    [247, 195, 37],
    [255, 105, 180],
], np.uint8)


def detect(opt, second_classifier):
    save_img = not opt.no_save and not opt.source.endswith('.txt')  # save inference images
    webcam = opt.source.isnumeric() or opt.source.endswith('.txt') or opt.source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    if opt.save_txt or save_img:
        (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    bnd_dir = Path('./ood/datasets/boundary_data')
    bnd_json = str(bnd_dir / 'yolov7_preds/yolov7_preds_filtered.json')
    if opt.save_boundary_data:
        if os.path.exists(bnd_json):
            with open(bnd_json, 'rb') as file:
                jdict = json.load(file)
        else:
            (bnd_dir / 'images').mkdir(parents=True, exist_ok=True)
            (bnd_dir / 'yolov7_preds').mkdir(parents=True, exist_ok=True)
            jdict = []

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(opt.img_size, s=stride)  # check img_size

    if not opt.no_trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    ## Segmentation
    seg_model = seg_models.get_model(opt.seg_model, pretrained=False)
    state_dict = load_weights(opt.seg_weights)
    seg_model.load_state_dict(state_dict)
    seg_predictor = Predictor(seg_model, half)

    ## OOD
    fe_model = second_classifier['backbone_model']
    fe_model.to(device).eval()
    cluster = second_classifier['cluster']
    second_classify = second_classifier['pred_func']
    ood_thres_dict = second_classifier['thresholds']
    bnd_thres = 50
    low_bound = ood_thres_dict[f"{max(1, int(opt.ood_thres)-bnd_thres)}%"]
    up_bound = ood_thres_dict[f"{min(100, int(opt.ood_thres)+bnd_thres)}%"]

    # Set Dataloader
    view_img = opt.view_img
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(opt.source, img_size=imgsz, stride=stride)
        print('The number of test datasets: ', len(dataset))

    # Get names and colors
    names = ['unknown', 'known', 'filtered']
    colors = [[0,0,255], [0, 0, 0], [0, 255, 0]]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    t0 = time_synchronized()
    time_records, windows = [], []
    break_point = False
    
    for img_i, [path, img, im0s, vid_cap] in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        # (prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        ground_sky_bin = seg_predictor.bbox_filtering(im0s, img_size=imgsz, stride=stride)
        t4 = time_synchronized()
        pred = second_classify(pred, ground_sky_bin, fe_model, cluster, ood_thres_dict[opt.ood_thres+'%'], 
                               img, im0s, opt.score_matrix, opt.cov_matrix_path, filter_thres=opt.filter_thres)
        t5 = time_synchronized()

        totalT, objT, nmsT, segT, oodT = t5-t1, t2-t1, t3-t2, t4-t3, t5-t4
        time_records.append([totalT, objT, nmsT, segT, oodT])

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, f'[{img_i}/{len(dataset)}] ', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            img_dir = os.path.join(save_dir, 'images')
            if save_img:
                os.makedirs(img_dir, exist_ok=True)
            save_path = os.path.join(img_dir, p.name.split('.')[0]+'.jpg') # img.jpg
            video_path = str(save_dir / 'video')
            txt_path = str(save_dir / 'labels' / p.stem)  # img.txt
            bnd_fname = p.name.split('.')[0]+'_'+str(img_i) if webcam else p.name.split('.')[0]
            bnd_img_path = os.path.join(bnd_dir, 'images', bnd_fname+'.jpg') # img.jpg

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cat_id, ood_scr, cls in reversed(det):
                    xyxy = (torch.tensor(xyxy).view(1, 4).to(torch.int32)).view(-1).tolist()
                    if opt.save_txt:  # Write to file
                        line = (cls, *xyxy, conf, cat_id)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        if (not opt.draw_interm) and (int(cls) != 0):
                            continue
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, color=colors[int(cls)], line_thickness=2)

                    if opt.save_boundary_data and (low_bound < ood_scr < up_bound):
                        jdict.append({'image_id': bnd_fname,
                                    'category_id': int(cat_id.item()),
                                    'bbox': [round(x, 3) for x in xyxy],
                                    'score': round(conf.item(), 5),
                                    'is_known': -1})  
                        Image.fromarray(im0[:,:,[2,1,0]]).save(bnd_img_path)
                
            if opt.draw_interm:
                im0 = Image.blend(Image.fromarray(BIN_COLORS[ground_sky_bin[0]]), Image.fromarray(im0), 0.6)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * totalT):.1f}ms, {(1/totalT):.1f} fps) Total, ({(1E3 * objT):.1f}ms) Object Recognize, ({(1E3 *nmsT):.1f}ms) NMS, ({(1E3 *segT):.1f}ms) BBox filtering, ({(1E3*oodT):.1f}ms) OOD,')

            # Stream results
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    break_point=True
                    break

            # Save results (image with detections)
            if save_img:
                if (dataset.mode in ['video', 'stream']) or ('modd' in opt.source):
                    im0 = np.asarray(im0)
                    if vid_path != video_path:  # new video
                        vid_path = video_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 20, im0.shape[1], im0.shape[0]
                        video_path += '.mp4'
                        vid_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
                elif dataset.mode == 'image':
                    im0 = Image.fromarray(im0[:,:,[2,1,0]]) if isinstance(im0, np.ndarray) else im0
                    im0.save(save_path)
                    print(f" The image with the result is saved in: {save_path}")

        if break_point:
            break

    print('\n')

    if opt.save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if opt.save_txt else ''
        #print(f"Results saved to {save_dir}{s}")
        
    if opt.save_boundary_data and len(jdict):
        print('saving %s...' % bnd_json)
        with open(bnd_json, 'w') as f:
            json.dump(jdict, f)
    
    if opt.calc_performance:
        # save json
        txt_dir = str(save_dir / 'labels')
        json_dir = str(save_dir / 'json')
        os.makedirs(json_dir, exist_ok=True)
        convert_txt_to_json(txt_dir, json_dir)
            
        # call and set format of pred_infos & ann_infos
        ood_infos = load_file(os.path.join(json_dir, 'preds.json'))
        ann_file_path = os.path.join(opt.source, '../annotations/all.json')
        ann_infos = load_file(ann_file_path)
        
        # calc precision and recall
        performance = calc_iou_performance(ood_infos, ann_infos)
        print(performance)


    print(f'Done. ({time_synchronized() - t0:.3f}s)')
    mTotalT, mObjT, mNmsT, mSegT, mOodT = np.mean(time_records, axis=0)
    print(f'mean time per frame : ({(1E3 * mTotalT):.1f}ms, {(1/mTotalT):.1f} fps) Total, ({(1E3 * mObjT):.1f}ms) Object Recognize, ({(1E3 *mNmsT):.1f}ms) NMS, ({(1E3 *mSegT):.1f}ms) BBox Filtering, ({(1E3*mOodT):.1f}ms) OOD,')


def get_args():
    parser = argparse.ArgumentParser()
    dataset = 'nexreal/01'
    datatype = 'videos'
    parser.add_argument('--weights', nargs='+', type=str, default='./yolov7/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=f'./datasets/{dataset}/{datatype}', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.05, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.001, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-boundary-data', action='store_true', help='save boundary data')
    parser.add_argument('--calc-performance', action='store_true', help='save results to preds.json')
    parser.add_argument('--no-save', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', default=True, help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='./runs/detect', help='save results to project/name')
    parser.add_argument('--name', default=dataset, help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--draw-interm', action='store_true', help='save intermediate results')

    # Segmentation
    parser.add_argument("--seg_model", default='wodis', type=str, choices=seg_models.model_list, help="Model architecture.")
    parser.add_argument("--seg_weights", default='./seg/weights/20230412132104_wodis_cwsl_brightness.ckpt',
                        type=str, help="Path to the model weights or a model checkpoint.")
    parser.add_argument('--filter-thres', type=float, default=0.7, help='filtering threshold')
    
    # Feature extractor
    parser.add_argument('--backbone_arch', default='resnet50', choices=['resnet50', 'resnet50_tune'], type=str, help='')
    parser.add_argument('--backbone_weight', default='./ood/backbone/resnet_funed_e100.pth', type=str, help='Path to backbone weight')

    # OOD
    parser.add_argument('--ood-thres', type=str, default='18', help='OOD threshold')
    parser.add_argument('--score_matrix', default='euclidean', type=str, choices=['euclidean', 'mahalanobis', 'cosineSim'])
    parser.add_argument('--threshold_path', default='./ood/cache/threshold/kmeans_resnet50_seaships.json', type=str, help='Path to threshold')
    parser.add_argument('--cluster_path', default='./ood/cache/cluster/kmeans_resnet50_seaships.pkl', type=str, help='Path to cluster model')
    parser.add_argument('--cov_matrix_path', default='./ood/cache/cov_matrix/kmeans_resnet50_seaships.pkl', type=str, help='Path to cov matrix')

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = get_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect(opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt)