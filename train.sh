#!/bin/sh

while getopts p:n:s:w:y:a:l: flag
do
    case "${flag}" in 
        p) pos=${OPTARG};;
        n) neg=${OPTARG};;
        s) strong=${OPTARG};;
        w) weak=${OPTARG};;
        y) syn=${OPTARG};;
        a) ant=${OPTARG};;
        l) lr=${OPTARG};;
    esac
done

PARAMETERS="$pos"_"$neg"'_'"$strong"'_'"$weak"'_'"$syn"'_'"$ant"'_'"$lr"
OUTPUT_DIR="./output/"$PARAMETERS'/'
LOG_DIR="./log/"
CKPT_DIR="./ckpt/"$PARAMETERS'/'

mkdir $OUTPUT_DIR
mkdir $CKPT_DIR


LOG_PATH=$LOG_DIR$PARAMETERS".txt"
FIGURE_PATH=$LOG_DIR$PARAMETERS".png"
CKPT_PATH=$CKPT_DIR"hg2vec.ckpt"
OUTPUT_PATH=$OUTPUT_DIR"hg2vec.txt"


time python main.py --emb_dimension 300  --output_per_epoch=True --num_workers 32 --window 5 --beta_pos $pos  \
--beta_neg $neg --beta_strong $strong --beta_weak $weak --beta_syn $syn --beta_ant $ant --batch_size 16  \
--epochs 5 --lr $lr --output_vector_path=$OUTPUT_PATH --output_ckpt_path=$CKPT_PATH \
--neg_size 5 --strong_size 5 --weak_size 5 --syn_size 5 --ant_size 5

python evaluate.py $OUTPUT_DIR"hg2vec.txt" 