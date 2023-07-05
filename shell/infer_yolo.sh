echo "Start YOLO-v7 prediction! \n"

python yolov7/test.py \
--data ./data_example/data_example.yaml \
--task test \
--save-json \
--project ./data_example \
--name yolov7_preds \
--exist-ok

echo "YOLO-v7 prediction Done! \n"