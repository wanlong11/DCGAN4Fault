python trainNLAresnet18.py \
        --data_dir ./data/trainData \
        --SB_before 10 \
        --n_epochs  100 \
        --eval_interval 1 \
        --tensorboard_dir /tf_logs/NLARESNET18/ \
        --save_dir NLARESNET18 \
        --project_name NLAresnet18
