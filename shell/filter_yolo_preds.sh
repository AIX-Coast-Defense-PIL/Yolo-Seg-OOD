echo "Start filtering YOLO-v7 predictions! \n"
data_dir=./data_example

python seg/predict.py \
--dataset_dir $data_dir/images \
--yolo_preds_dir $data_dir

python utils/refine.py \
--dataset_dir $data_dir \
--json_fname yolov7_preds/yolov7_preds_filtered.json

echo "YOLO-v7 prediction filtering Done! \n"