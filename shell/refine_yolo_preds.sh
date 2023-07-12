data_dir=./ood/datasets/known_data

python utils/refine.py \
--dataset_dir $data_dir \
--json_fname yolov7_preds/yolov7_preds_filtered.json