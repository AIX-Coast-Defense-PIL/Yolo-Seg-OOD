echo "Start OOD backbone (ResNet-50) train! \n"

python ood/backbone.py \
--data_root ./data_example \
--backbone_weight ./ood/backbone/resnet_e100_data_example.pth

echo "OOD backbone (ResNet-50) train Done! \n"