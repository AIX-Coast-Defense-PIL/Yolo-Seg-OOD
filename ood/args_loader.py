import argparse
import os

def get_args(root):
    parser = argparse.ArgumentParser(description='OOD detection')

    # learning setting
    parser.add_argument('--feat', default=True, type=bool, help='image feature is not exist. So need to make')
    parser.add_argument('--mode', default='both', type=str, choices=['train', 'test', 'both'], help='train or eval')
    parser.add_argument('--train_bs', default=2, type=int, help='training batch size')
    parser.add_argument('--gpu', default=0, type=int, help='gpu number')

    # data path
    parser.add_argument('--data_root', default=os.path.join(root, 'datasets'), type=str, help='root directory path of all data')
    parser.add_argument('--train_data', default='seaships', type=str, choices=['coco', 'custom102', 'seaships'])
    parser.add_argument('--test_data', default='custom102', type=str, choices=['coco', 'custom102', 'seaships', 'modd/*'])

    # model
    parser.add_argument('--backbone_arch', default='resnet50', choices=['resnet50', 'resnet50_tune'], type=str, help='')
    parser.add_argument('--backbone_dir', default='./backbone', type=str, help='')

    # cluster
    parser.add_argument('--cluster', default='kmeans', type=str, choices=['kmeans', 'GM'])
    parser.add_argument('--num_cluster', default=5, type=int, help='how many k-menas clusters')

    # score
    parser.add_argument('--score_matrix', default='euclidean', type=str, choices=['euclidean', 'mahalanobis', 'cosineSim'])

    # threshold
    # parser.add_argument('--conf_threshold', default='euclidean', type=int, help='threshold made by yolo confidence score')

    args = parser.parse_args()

    # save path
    args.train_feat_path = os.path.join(root, 'ood/cache/feature/', f'train_{args.backbone_arch}_{args.train_data}.pkl')
    args.test_feat_path = os.path.join(root, 'ood/cache/feature/', f'test_{args.backbone_arch}_{args.test_data}.pkl')
    args.cluster_path = os.path.join(root, 'ood/cache/cluster/', f'{args.cluster}_{args.backbone_arch}_{args.train_data}.pkl')
    args.cov_matrix_path = os.path.join(root, 'ood/cache/cov_matrix/', f'{args.cluster}_{args.backbone_arch}_{args.train_data}.pkl')
    args.threshold_path = os.path.join(root, 'ood/cache/threshold/', f'{args.cluster}_{args.backbone_arch}_{args.train_data}.json')
    args.score_path = os.path.join(root, 'ood/scores/', f'{args.cluster}_{args.backbone_arch}_{args.train_data}_{args.test_data}.pkl')

    return args