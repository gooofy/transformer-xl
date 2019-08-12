#!/bin/bash

if [[ $1 == 'train' ]]; then

    # rm -rf LM-TFM-sentencepiece/

    echo 'Run training...'
    python train_sp.py \
        --cuda \
        --data ../data/encoded-de/ \
        --sp_model ../sp-model.model \
        --work_dir de61M-root \
        --adaptive \
        --n_layer 16 \
        --d_model 410 \
        --n_head 10 \
        --d_head 41 \
        --d_inner 2100 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.0001 \
        --warmup_step 0 \
        --num_epochs 3 \
        --tgt_len 150 \
        --mem_len 150 \
        --eval_tgt_len 150 \
        --batch_size 26 \
        --log-interval 10 \
        --eval-interval 500 \
        ${@:2}
    # python train.py \
    #     --cuda \
    #     --data ../data/encoded-de/ \
    #     --dataset sentencepiece \
    #     --restart \
    #     --restart_dir LM-TFM-sentencepiece/20190803-205205 \
    #     --adaptive \
    #     --n_layer 16 \
    #     --d_model 410 \
    #     --n_head 10 \
    #     --d_head 41 \
    #     --d_inner 2100 \
    #     --dropout 0.1 \
    #     --dropatt 0.0 \
    #     --optim adam \
    #     --lr 0.0001 \
    #     --warmup_step 0 \
    #     --max_step 200000 \
    #     --tgt_len 150 \
    #     --mem_len 150 \
    #     --eval_tgt_len 150 \
    #     --batch_size 26 \
    #     --log-interval 10 \
    #     --eval-interval 500 \
    #     ${@:2}
    # python train.py \
    #     --cuda \
    #     --data ../data/wikitext-103/ \
    #     --dataset wt103 \
    #     --adaptive \
    #     --n_layer 16 \
    #     --d_model 410 \
    #     --n_head 10 \
    #     --d_head 41 \
    #     --d_inner 2100 \
    #     --dropout 0.1 \
    #     --dropatt 0.0 \
    #     --optim adam \
    #     --lr 0.00025 \
    #     --warmup_step 0 \
    #     --max_step 200000 \
    #     --tgt_len 150 \
    #     --mem_len 150 \
    #     --eval_tgt_len 150 \
    #     --batch_size 60 \
    #     --multi_gpu \
    #     --gpu0_bsz 4 \
    #     --scheduler constant \
    #     ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 64 \
        --mem_len 640 \
        --clamp_len 400 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
