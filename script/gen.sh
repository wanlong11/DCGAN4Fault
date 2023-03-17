for num in {1..4}
do
python dcganGen.py \
        --model_file ./SATEF"$num"_result/SATEF"$num"/G3last.pt \
        --out_dir SATEF"$num"_GEN \
        --gen_num 100 \
        --img_size 512
done