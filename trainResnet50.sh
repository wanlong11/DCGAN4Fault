python trainResnet50.py \
        --data_dir ./data/trainData \
        --SB_before 10 \
        --n_epochs  100 \
        --eval_interval 1 \
        --tensorboard_dir /tf_logs/RESNET50/ \
        --save_dir RESNET50 \
        --project_name resnet50
