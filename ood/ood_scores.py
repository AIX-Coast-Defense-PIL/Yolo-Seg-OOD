import numpy as np
import time
import os

from utils.file_processing import save_file, load_file

def calc_cov_matrixes(class_means, feat_log, file_path):
    print('calculating cov_matrixes...')
    start_time = time.time()
    cov_matrixes = np.zeros((len(class_means), len(class_means[0]), len(class_means[0])))
    for cls_idx, class_mean in enumerate(class_means):
        cov_matrix = np.zeros((len(class_mean), len(class_mean)))
        for feat in feat_log:
            feat_minus_mean = np.array(feat - class_mean).reshape(-1, 1)
            cov_matrix += feat_minus_mean @ np.transpose(feat_minus_mean)
        cov_matrix /= len(feat_log)
        cov_matrixes[cls_idx] = cov_matrix

    save_file(cov_matrixes, file_path)
    print(f'finish cov calculatime [time : {time.time() - start_time}]')

    return cov_matrixes

def calc_euclidean_distance(cluster, feat_log):
    distances = cluster.transform(feat_log)
    scores = [0]*len(distances)
    for idx, distance in enumerate(distances):
        scores[idx] = min(distance)
    
    return scores

def calc_mahalanobis_distance(cluster, feat_log, mode, file_path):
    # calc mean and cov matrix per classes
    class_means = cluster.cluster_centers_
    if mode == 'train':
        cov_matrixes = calc_cov_matrixes(class_means, feat_log, file_path)
    else:
        cov_matrixes = load_file(file_path)

    # calculate mahalanobis distance
    scores = np.zeros(len(feat_log))
    for feat_idx, feat in enumerate(feat_log):
        distances = []
        for cls_idx, class_mean in enumerate(class_means):
            feat_minus_mean = np.array(feat - class_mean).reshape(1, -1)
            left = feat_minus_mean @ np.linalg.inv(cov_matrixes[cls_idx])
            distance = left @ np.transpose(feat_minus_mean)
            distances.append(distance[0][0])
        scores[feat_idx] = min(distances)

    return scores

def calc_cosine_similarity(cluster, feat_log):
    def cos_sim(A, B):
        return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

    class_means = cluster.cluster_centers_
    scores = [-max([cos_sim(class_mean, feat) for class_mean in class_means]) for feat in feat_log]
    
    return scores


def calc_distance_score(cluster, feat_log, method, mode=None, save_path=None):
    if method == "euclidean" : return calc_euclidean_distance(cluster, feat_log)
    elif method == "mahalanobis" : return calc_mahalanobis_distance(cluster, feat_log, mode, save_path)
    elif method == "cosineSim" : return calc_cosine_similarity(cluster, feat_log)

if __name__=='__main__':
    # calc_ood_score("cluster", )
    pass