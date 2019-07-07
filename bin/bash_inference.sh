#!/usr/bin/env bash

python3 -m pdb translate.py \
    -model /home/murphyhuang/dev/mldata/OpenNMT/Summarization/model/harvard_pretrained_gigaword_nocopy_acc_51.33_ppl_12.74_e20.pt \
    -src /home/murphyhuang/dev/mldata/OpenNMT/Summarization/sumdata/mini/valid_mini_tmp.article.filter.txt \
    -output /tmp/pred.txt \
    -batch_size 1 \
    -replace_unk -verbose
