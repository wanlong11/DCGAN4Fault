python trainNLAresnet50.py \
        --data_dir ./data/trainData \
        --SB_before 10 \
        --n_epochs  100 \
        --eval_interval 1 \
        --tensorboard_dir /tf_logs/NLARESNET50/ \
        --save_dir NLARESNET50 \
        --project_name NLAresnet50 \
        --test_dir ./data/testData
