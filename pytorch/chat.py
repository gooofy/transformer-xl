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

DEVICE = 'cuda'

# TOKENS_TO_GENERATE = 42
TOKENS_TO_GENERATE = 256
MIN_TOKENS_TO_GENERATE = 32

TOP_K = 8

#
# load model
#

print ("loading model from %s ..." % MODEL_PATH)

mw = ModelWrapper.load(MODEL_PATH, SP_MODEL_PATH, DEVICE)

# txt = "Die Forschung an der künstlichen Intelligenz"
# txt = "Die Hitzewelle, mit der sich die amerikanische Metropole New York derzeit rumschlägt, wird langsam zum immer größeren Problem: Bei Temperaturen von 40 Grad waren rund 50.000 Haushalte in Brooklyn und in Westchester County am Sonntag ohne Strom. Das teilten"

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
        # ntk = mw.get_next_top_k(tokens, TOP_K)

        # print (ntk)

        # import pdb; pdb.set_trace()

        # convert log probs to real probs
        logprobs = np.array(list(map(lambda a: a[0], ntk)))
        probs = np.exp(logprobs) / np.exp(logprobs).sum()

        # pick next token randomly according to probs distribution
        next_token_n = np.random.choice(TOP_K, p=probs)
        next_token = ntk[next_token_n][1]
        # print (next_token)
       
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

sys.exit(0)


# txt = "Die Elektromobilität"
# txt = "Domenico Perrino"
# txt = "Die EnBW"
# txt = "Mein Name ist Bimbo"

while True:

    txt = input("(q: quit) > ") 

    if txt == 'q':
        break

    tokens = mw.tokenize(txt)

    sys.stdout.write(txt)

    for i in range(TOKENS_TO_GENERATE):

        # generate TOP_K potential next tokens
        ntk = mw.get_next_top_k(tokens, TOP_K)

        # convert log probs to real probs
        logprobs = np.array(list(map(lambda a: a[0], ntk)))
        probs = np.exp(logprobs) / np.exp(logprobs).sum()

        # pick next token randomly according to probs distribution
        next_token_n = np.random.choice(TOP_K, p=probs)
        next_token = ntk[next_token_n][1]
        # print (next_token)
       
        tokens.append(next_token)

        if next_token[0] == '▁':
            sys.stdout.write(' ')
        sys.stdout.write(mw.sp_model.DecodePieces([next_token]))
        sys.stdout.flush()

        # find a good point to stop
        if (i > MIN_TOKENS_TO_GENERATE) and \
           ( ('.' in next_token) or ('!' in next_token) or ('?' in next_token) ):
            break

    print()

#print(mw.sp_model.DecodePieces(tokens))

# lp = mw.get_log_probs(tokens)
# 
# print(lp)
# 
# for idx in range(len(tokens)):
#     log_probs = mw.get_log_probs(tokens)[idx]
#     top_indices = torch.argsort(log_probs)[-TOP_K:]
#     top_log_probs = log_probs[top_indices]
# 
#     top_k_log_probs = [(log_prob.item(), mw.sp_model.IdToPiece(idx.item()))
#                       for idx, log_prob in
#                       reversed(list(zip(top_indices, top_log_probs)))]
# 
#     print ("%2d: %s -> %s" % (idx, tokens[idx], top_k_log_probs))
# 
# 
# sys.exit(0)
