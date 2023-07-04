dir_name=data_example

python ood/main.py \
--mode train \
--data_root . \
--train_data $dir_name \
--backbone_arch resnet50_tune \
--backbone_weight ./ood/backbone/resnet_e100_$dir_name.pth