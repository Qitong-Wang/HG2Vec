OUTPUT_DIR="./output/"
LOG_DIR="./log/"
CKPT_DIR="./ckpt/"
rm -r $OUTPUT_DIR
mkdir $OUTPUT_DIR
rm -r $LOG_DIR
mkdir $LOG_DIR
rm -r $CKPT_DIR
mkdir $CKPT_DIR

sh ./train.sh -p 1.0 -n 3.5 -s 0.6 -w 0.4 -y 1.0 -a 1.0 -l 0.003