export GLUE_DIR=$HOME/COS568-DistLM-SP26/glue_data
export TASK_NAME=RTE

rank=$1
ip_prefix=10.10.1
master_suffix=2
master_port=29500

python3 task2a/run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir $HOME/$TASK_NAME/ \
  --overwrite_output_dir \
  --master_ip $ip_prefix"."$master_suffix \
  --master_port $port \
  --world_size 4 \
  --local_rank $rank
