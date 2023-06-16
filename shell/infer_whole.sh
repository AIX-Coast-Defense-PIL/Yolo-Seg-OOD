timestamp=`date +%Y%m%d%H%M%S`

log_dir=runs/detect/$timestamp
mkdir -p $log_dir

python test.py \
--source ./datasets/custom102/images \
--name $timestamp \
--calc-performance True \
--conf-thres 0.05 \
--ood-thres 87 \
--exist-ok &>> $log_dir/logs_$timestamp.log
