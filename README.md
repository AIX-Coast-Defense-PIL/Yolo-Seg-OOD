# OpenSet ObjectDetection by YOLO + OOD

## Setting
- Ubuntu 16.04
- python=3.9


### YOLOv7
- install requirements.txt
```
cd yolov7
pip install -r requirements.txt
```

- prepare COCO dataset
```
bash scripts/get_coco.sh
```
delete `*.cache` files if they exist in the `.\datasets\*` directory

- test
    - download weight(ex. yolov7.pt) on the [website](https://github.com/wongkinyiu/yolov7)
```
python detect.py
```


### Segmentation
- bbox filtering
```
cd seg
python predict.py
```


### OOD
- install requirements.txt
```
cd ood
pip install -r requirements.txt
```

- install etc.
```
pip3 install -U scikit-learn
```

## Test
```
python test.py
```