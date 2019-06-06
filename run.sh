#!/bin/bash
input_file=$1
output_file=$2
BERT_BASE_DIR=$3

layers=-6
align_strategy=mean

cur_dir=$PWD
# cur_dir=$(realpath "$0" | sed 's|\(.*\)/.*|\1|')
  
python $cur_dir/extract_features.py \
    --input_file=$input_file \
    --output_file=$input_file.json \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --layers=$layers \
    --max_seq_length=256 \
    --batch_size=8 \
    --do_lower_case=False

python $cur_dir/get_aligned_bert_emb.py \
    --input_file $input_file.json \
    --mode $align_strategy \
    --output_file $output_file \

# rm the derivative file
rm -f $input_file.json
