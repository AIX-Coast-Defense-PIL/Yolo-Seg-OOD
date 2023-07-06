timestamp=`date +%Y%m%d%H%M%S`

python seg/train.py \
--source /home/leeyoonji/workspace/git/datasets/mastr1478/images \
--model wodis \
--model_name mastr1478_wodis \
--workers 2 \
--validation \
--batch_size 4 \
--epochs 100 \
--separation_loss cwsl \
--separation_loss_lambda 0.01 \
--output_dir seg/output \
--datetime $timestamp