timestamp=`date +%Y%m%d%H%M%S`

python test.py \
--source rtsp://admin:admin@147.46.89.127:8554/test \
--name rtsp_$timestamp \
--conf-thres 0.05 \
--ood-thres 18 \
--exist-ok \
--save-boundary-data \
--threshold_path ./ood/cache/threshold/kmeans_resnet50_seaships.json \
--cluster_path ./ood/cache/cluster/kmeans_resnet50_seaships.pkl