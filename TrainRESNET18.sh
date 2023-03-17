python trainResnet.py \
        --data_dir ./data/trainData \
        --SB_before 50 \
        --n_epochs  100 \
        --eval_interval 1 \
        --tensorboard_dir /tf_logs/RESNET18/ \
        --save_dir RESNET18 \
        --test_dir ./data/testData \
        --project_name resnet18
