timestamp=`date +%Y%m%d%H%M%S`

python test.py \
--source ./datasets/custom102/images \
--name custom102_$timestamp \
--conf-thres 0.05 \
--ood-thres 18 \
--exist-ok \
--no-save \
--save-boundary-data \
--threshold_path ./ood/cache/threshold/kmeans_resnet50_seaships.json \
--cluster_path ./ood/cache/cluster/kmeans_resnet50_seaships.pkl