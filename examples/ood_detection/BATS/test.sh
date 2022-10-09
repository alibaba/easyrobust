
DataSet=$1
METHOD=$2
OUT_DATA=$3
BATS=$4

CUDA_VISIBLE_DEVICES=2 python test_ood.py \
--name test_${METHOD}_${OUT_DATA} \
--in_datadir ./ID_OOD_dataset/ \
--out_datadir ./ID_OOD_dataset/${OUT_DATA} \
--batch 1 \
--bats ${BATS} \
--dataset ${DataSet} \
--logdir checkpoints/test_log \
--score ${METHOD}
