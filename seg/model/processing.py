
import cv2
import numpy as np
from PIL import Image

def ground_sky_filtering(seg_pred, obs_bin, surr_sky_ratio_eps = 0.3, width_eps = 0.95, dilate_eps = 5):

    ## Connected Component Labeling 
    ccl_num_labels, ccl_labels, ccl_stats, ccl_centroids = cv2.connectedComponentsWithStats(obs_bin)

    ## Ground Extraction
    ground_sky_bin = np.uint8(np.where(seg_pred[:,:,0]==90,1,0)) # binary mask for sky components
    for ccl_idx, (x,y,w,h,a) in enumerate(ccl_stats):
        if ccl_idx==0: continue

        ccl_bin = np.uint8(np.where(ccl_labels==ccl_idx, 1, 0))
        surr_pix = cv2.dilate(ccl_bin, np.ones((dilate_eps,dilate_eps), np.uint8), iterations=1) - ccl_bin
        # Image.fromarray(surr_pix*255).save('/home/leeyoonji/workspace/yolov7-GMM/seg/output/wodis_seaships/temp2.png')
        sum_pix = surr_pix.sum()
        surr_pix = np.multiply(seg_pred[:,:,0], surr_pix.astype(int))
        if surr_pix.sum() != 0:
            if ((surr_pix==90).sum() / sum_pix > surr_sky_ratio_eps) or (w > obs_bin.shape[1]*width_eps): 
                ground_sky_bin += ccl_bin
        #         print(ccl_idx, ' True')
        # Image.fromarray(ccl_bin*255).save('/home/leeyoonji/workspace/yolov7-GMM/seg/output/wodis_seaships/temp.png')
    
    ground_sky_bin = ground_sky_bin > 0
    return ground_sky_bin