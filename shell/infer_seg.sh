timestamp=`date +%Y%m%d%H%M%S`
dataset=aihub
dir_name=pretrained_mastr1478
model=wodis
task=cwsl_brightness
# dataset=seaships_all

log_dir=WaSR/output/predictions/$dir_name/$timestamp\_$model\_$task\_$dataset
mkdir -p $log_dir

python WaSR/predict.py \
--dataset_config WaSR/configs/$dataset\_test.yaml \
--model $model \
--weights /home/leeyoonji/workspace/git/WaSR/nex_output/wodis_mastr1478/20230412132104_cwsl_brightness/checkpoints/epoch=82-step=12200.ckpt \
--output_dir $log_dir \
--batch_size 12 \
--mode eval
# --mode pred