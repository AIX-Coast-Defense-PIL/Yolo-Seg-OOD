echo "Start OOD cluster (K-Means) train! \n"

python ood/main.py \
--mode train \
--data_root . \
--train_data data_example \

echo "OOD cluster (K-Means) train Done! \n"