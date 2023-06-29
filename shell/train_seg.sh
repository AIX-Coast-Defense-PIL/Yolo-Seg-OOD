timestamp=`date +%Y%m%d%H%M%S`
dataset=mastr1478
model=wodis

model_name=$model\_$dataset
log_dir=seg/output/logs/$model_name/$timestamp
mkdir -p $log_dir

python seg/train.py \
--source /home/leeyoonji/workspace/git/datasets/mastr1478/images \
--model wodis \
--model_name $model_name \
--workers 2 \
--validation \
--batch_size 4 \
--epochs 100 \
--separation_loss cwsl \
--separation_loss_lambda 0.01 \
--output_dir seg/output \
--datetime $timestamp #&>> $log_dir/$model_name\_$timestamp.log