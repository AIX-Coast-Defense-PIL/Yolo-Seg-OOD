import argparse
import os

def get_args(root):
    parser = argparse.ArgumentParser(description='OOD detection')

    # learning setting
    parser.add_argument('--feat', default=True, type=bool, help='image feature does not exist. So need to make')
    parser.add_argument('--mode', default='both', type=str, choices=['train', 'test', 'both'], help='train or eval')
    parser.add_argument('--train_bs', default=16, type=int, help='training batch size')
    parser.add_argument('--gpu', default=0, type=int, help='gpu number')

    # data path
    parser.add_argument('--train_data', default='./datasets/seaships', type=str)
    parser.add_argument('--add_train_data', default=None, type=str)
    parser.add_argument('--test_data', default='./datasets/custom102', type=str, 
                        choices=['./datasets/coco', './datasets/custom102', './datasets/seaships', './datasets/modd/*', './datasets/nexreal/*'])

    # model
    parser.add_argument('--backbone_arch', default='resnet50', choices=['resnet50', 'resnet50_tune'], type=str, help='')
    parser.add_argument('--backbone_weight', default='./ood/backbone/resnet_funed_e100.pth', type=str, help='Path to backbone weight')

    # cluster
    parser.add_argument('--cluster', default='kmeans', type=str, choices=['kmeans', 'GM'])
    parser.add_argument('--num_cluster', default=5, type=int, help='how many k-menas clusters')

    # score
    parser.add_argument('--score_matrix', default='euclidean', type=str, choices=['euclidean', 'mahalanobis', 'cosineSim'])

    # threshold
    # parser.add_argument('--conf_threshold', default='euclidean', type=int, help='threshold made by yolo confidence score')

    args = parser.parse_args()

    # save path
    train_data_name = args.train_data.split('/')[-1]
    test_data_name = args.test_data.split('/')[-1]
    if args.add_train_data not in [None, 'None']:
        train_data_name = 'seaships_' + train_data_name
        test_data_name = 'seaships_' + test_data_name
    args.train_feat_path = os.path.join(root, 'ood/cache/feature/', f'train_{args.backbone_arch}_{train_data_name}.pkl')
    args.test_feat_path = os.path.join(root, 'ood/cache/feature/', f'test_{args.backbone_arch}_{test_data_name}.pkl')
    args.cluster_path = os.path.join(root, 'ood/cache/cluster/', f'{args.cluster}_{args.backbone_arch}_{train_data_name}.pkl')
    args.cov_matrix_path = os.path.join(root, 'ood/cache/cov_matrix/', f'{args.cluster}_{args.backbone_arch}_{train_data_name}.pkl')
    args.threshold_path = os.path.join(root, 'ood/cache/threshold/', f'{args.cluster}_{args.backbone_arch}_{train_data_name}.json')
    args.score_path = os.path.join(root, 'ood/scores/', f'{args.cluster}_{args.backbone_arch}_{train_data_name}_{test_data_name}.pkl')

    return args