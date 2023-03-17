#创建总文件夹
mkdir -p ../data/trainData
mkdir -p ../data/testData


# 分割正确数据到train、test
mkdir -p ../data/trainData/data_0
numfile=$(ls -l ../data/SDP/data_0 | wc -l)
# shellcheck disable=SC2004
num_to_copy=$(($numfile / 2))
# shellcheck disable=SC2012
ls ../data/SDP/data_0 | head -n $num_to_copy | xargs -I {} cp ../data/SDP/data_0/{} ../data/trainData/data_0/
mkdir -p ../data/testData/data_0
# shellcheck disable=SC2012
ls ../data/SDP/data_0 | tail -n $num_to_copy | xargs -I {} cp ../data/SDP/data_0/{} ../data/testData/data_0/
echo 第0个文件构造完毕


#每个类虚拟数据增加个数
export NUM_TO_ADD=1000


for i in {1..4}; do
  mkdir -p ../data/trainData/data_"$i"
  ls ../data/GEN/SATEF"$i"_GEN/ | head -n $NUM_TO_ADD | xargs -I {} cp ../data/GEN/SATEF"$i"_GEN/{} ../data/trainData/data_$i/
  # shellcheck disable=SC2012
  numfile=$(ls -l ../data/SDP/data_"$i" | wc -l)
  # shellcheck disable=SC2004
  num_to_copy=$(($numfile / 2))
  # shellcheck disable=SC2012
  ls ../data/SDP/data_"$i" | head -n $num_to_copy | xargs -I {} cp ../data/SDP/data_"$i"/{} ../data/trainData/data_$i/

  # shellcheck disable=SC2086
  mkdir -p ../data/testData/data_$i
  # shellcheck disable=SC2012
  ls ../data/SDP/data_"$i" | tail -n $num_to_copy | xargs -I {} cp ../data/SDP/data_"$i"/{} ../data/testData/data_$i/

  echo 第 "$i" 个文件构造完毕
done
