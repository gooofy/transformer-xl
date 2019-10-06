#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json

import readline
# from pathlib import Path
# from lm import inference

import numpy as np
import sentencepiece as spm

import torch

from mem_transformer import MemTransformerLM
from inference import ModelWrapper

MODEL_PATH = 'de163M-base-root'
SP_MODEL_PATH = 'de163M-base-root/sp-model.model'

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# TOKENS_TO_GENERATE = 42
TOKENS_TO_GENERATE = 256
MIN_TOKENS_TO_GENERATE = 32

TOP_K = 8

#
# load model
#

print ("loading model from %s ..." % MODEL_PATH)

mw = ModelWrapper.load(MODEL_PATH, SP_MODEL_PATH, DEVICE)

while True:

    txt = input("(q: quit) > ") 

    if txt == 'q':
        break

    tokens = mw.tokenize(txt)
    print(tokens)
    if len(tokens)<20:
        print ("not enough context, please try again")
        continue

    for i in range(TOKENS_TO_GENERATE):

        # generate TOP_K potential next tokens
        ntk = mw.next_top_k(tokens, TOP_K)

        # import pdb; pdb.set_trace()

        # convert log probs to real probs
        logprobs = np.array(list(map(lambda a: a[0], ntk)))
        probs = np.exp(logprobs) / np.exp(logprobs).sum()

        # pick next token randomly according to probs distribution
        next_token_n = np.random.choice(TOP_K, p=probs)
        next_token = ntk[next_token_n][1]
       
        tokens.append(next_token)

        if next_token[0] == '▁':
            sys.stdout.write(' ')
        sys.stdout.write(mw.sp_model.DecodePieces([next_token]))
        sys.stdout.flush()

        # find a good point to stop
        # if (i > MIN_TOKENS_TO_GENERATE) and \
        #    ( ('.' in next_token) or ('!' in next_token) or ('?' in next_token) ):
        #     break

    print()

