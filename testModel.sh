export MODELPATH=""

python testModel.py \
        --data_dir ./data/trainData \
        --test_data_dir ./data/testData \
        --checkpoint $MODELPATH \
        --batch_size 1 \



