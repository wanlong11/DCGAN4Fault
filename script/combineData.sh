#创建总文件夹
mkdir -p ../data/trainData
mkdir -p ../data/testData

for i in {1..4}; do
  mkdir -p ../data/trainData/data_"$i"
  cp ../data/GEN/SATEF"$i"_GEN/* ../data/trainData/data_"$i"/
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
