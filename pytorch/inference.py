#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json

from pathlib import Path
from typing import List, Tuple

import torch
import torch.cuda
import sentencepiece as spm

from mem_transformer import MemTransformerLM

class ModelWrapper:

    def __init__(self, model, sp_model, device):

        self.model      = model
        self.sp_model   = sp_model
        self.device     = device
        self.model.eval()

    @classmethod
    def load (cls, model_path, sp_model_path, device):

        paramspath = os.path.join(model_path, 'params.json')
        with open(paramspath, 'r') as paramsf:
            xl_params = json.loads(paramsf.read())

        print (repr(xl_params))

        # cutoffs, tie_projs = [], [False]
        # cutoffs = [3500, 7500, 37500]
        # tie_projs += [True] * len(cutoffs)

        model = MemTransformerLM(xl_params['ntokens'],                         # 50000,
                                 xl_params['n_layer'],                         # 16,
                                 xl_params['n_head'],                          # 10,
                                 xl_params['d_model'],                         # 410,
                                 xl_params['d_head'],                          # 41,
                                 xl_params['d_inner'],                         # 2100,
                                 0.0,                                          # no dropout, 
                                 0.0,                                          # no dropatt,
                                 tie_weight     = xl_params['tie_weight'],     # True, 
                                 d_embed        = xl_params['d_embed'],        # 410, 
                                 div_val        = xl_params['div_val'],        # 1,
                                 tie_projs      = xl_params['tie_projs'],      # [False, True, True, True] 
                                 pre_lnorm      = xl_params['pre_lnorm'],      # False, 
                                 tgt_len        = xl_params['tgt_len'],        # 150,
                                 ext_len        = xl_params['ext_len'],        # 0, 
                                 mem_len        = xl_params['mem_len'],        # 150, 
                                 cutoffs        = xl_params['cutoffs'],        # [3500, 7500, 37500],
                                 same_length    = xl_params['same_length'],    # False,
                                 attn_type      = xl_params['attn_type'],      # 0,
                                 clamp_len      = xl_params['clamp_len'],      # -1, 
                                 sample_softmax = xl_params['sample_softmax']) # -1

        state_dict_path = os.path.join(model_path, 'valid_state_dict.pt')
        print ("loading weights %s ..." % state_dict_path)
        model.load_state_dict(torch.load(state_dict_path, map_location=torch.device(device)))
        print ("loading weights %s ... done." % state_dict_path)

        # with open(os.path.join(MODEL_PATH, 'model.pt'), 'rb') as f:
        #     model = torch.load(f)
        # model.apply(update_dropout)
        # model.apply(update_dropatt)

        para_model = model.to(device)

        # print ("loading model %s ... done." % MODEL_PATH)

        print ("loading sp model from %s ..." % sp_model_path)
        sp_model = spm.SentencePieceProcessor()
        sp_model.load(sp_model_path)
        print ("loading sp model from %s ... done." % sp_model_path)

        return cls(para_model, sp_model, device)

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            tokens.extend(self.sp_model.encode_as_pieces(line))
            # FIXME
            # assert not self.vocab.add_double_eos
            # if self.vocab.add_eos and i != len(lines) - 1:
            #     tokens.append(self.vocab.EOS)
        return tokens

    def get_log_probs(self, tokens: List[str]) -> torch.Tensor:
        """ Return log probabilities for next tokens.
        Shape of returned tensor is len(tokens) x len(self.vocab),
        where the first element contains log probabilities for tokens
        after the first, and last element log probabilities for tokens
        after the last one.
        """
        if not tokens:
            raise ValueError('tokens must be non-empty')

        # import pdb; pdb.set_trace()

        # all_xs = self.vocab.convert_to_tensor(tokens)
        all_xs = torch.LongTensor([ self.sp_model.PieceToId(token) for token in tokens ])

        all_log_probs = []
        with torch.no_grad():
            mems = tuple()
            input_len = self.model.tgt_len
            for idx in range(0, len(all_xs), input_len):
                xs = all_xs[idx: idx + input_len]
                xs = xs.to(device=self.device)
                batch_dim = 1  # batch size dimension is 1
                xs = xs.unsqueeze(batch_dim)
                log_probs, mems = self.model(xs, None, *mems)
                log_probs = log_probs.squeeze(batch_dim).data.cpu()
                all_log_probs.append(log_probs)
        return torch.cat(all_log_probs)

    def get_occurred_log_probs(
            self, tokens: List[str]) -> List[Tuple[str, float]]:
        """ Same as get_log_probs, but return a list of len(tokens) - 1,
        where log probs correspond to actually occurred tokens.
        """
        log_probs = self.get_log_probs(tokens)
        occured_log_probs = []
        for idx, token in enumerate(tokens[1:]):
            token_idx = self.vocab.sym2idx[token]
            occured_log_probs.append((token, float(log_probs[idx, token_idx])))
        return occured_log_probs

    def next_top_k( self, tokens: List[str], top_k: int = 40) -> List[Tuple[str, float]]:
        """ Return top k next tokens and their log probabilities.
        """
        log_probs = self.get_log_probs(tokens)[-1]
        top_indices = torch.argsort(log_probs)[-top_k:]
        top_log_probs = log_probs[top_indices]

        return [(log_prob.item(), self.sp_model.IdToPiece(idx.item()))
                for idx, log_prob in
                reversed(list(zip(top_indices, top_log_probs)))]

    def sample_next(self, tokens: List[str], top_k: int = 40) -> str:
        """ Sample next token from a multinomial distribution.
        """
        log_probs = self.get_log_probs(tokens)[-1]
        top_indices = torch.argsort(log_probs)[-top_k:]
        top_probs = log_probs[top_indices].double().exp()
        sampled_idx = top_indices[torch.multinomial(top_probs, 1).item()].item()
        return self.vocab.idx2sym[sampled_idx]

    def sample_text_iter(self, text: str, top_k: int = 40):
        """ An iterator yielding pieces of generated text, resulting text
        can be obtained by joining all of them with an empty string.
        """
        # TODO for longer texts we want to use memory and don't feed all tokens
        tokens = self.tokenize(text)
        while True:
            next_token = self.sample_next(tokens, top_k=top_k)
            yield (self.sp_processor.DecodePieces([tokens[-1], next_token])
                   [len(self.sp_processor.DecodePieces([tokens[-1]])):])
            tokens.append(next_token)

