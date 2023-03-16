export num=3
python dcgan.py \
  --n_epochs 1000 \
  --SB_before 998 \
  --img_dir ../data/SDP/data_$num/ \
  --project_name SATEF$num \
  --batch_size 64 \
  --lr 0.001
