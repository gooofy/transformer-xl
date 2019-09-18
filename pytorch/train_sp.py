# coding: utf-8
import argparse
import time
import math
import os, sys
import json
import random

from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm

import json_log_plots

from mem_transformer import MemTransformerLM
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str,
                    help='location of the data corpus')
parser.add_argument('--work_dir', type=str,
                    help='experiment work dir')
parser.add_argument('--sp_model', type=str,
                    help='sentencepiece model')
parser.add_argument('--n_layer', type=int, default=12,
                    help='number of total layers')
parser.add_argument('--n_head', type=int, default=10,
                    help='number of heads')
parser.add_argument('--d_head', type=int, default=50,
                    help='head dimension')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension')
parser.add_argument('--d_model', type=int, default=500,
                    help='model dimension')
parser.add_argument('--d_inner', type=int, default=1000,
                    help='inner dimension in FF')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='global dropout rate')
parser.add_argument('--dropatt', type=float, default=0.0,
                    help='attention probability dropout rate')
parser.add_argument('--init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--init_range', type=float, default=0.1,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range', type=float, default=0.01,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std', type=float, default=0.02,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std', type=float, default=0.01,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--optim', default='adam', type=str,
                    choices=['adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--warmup_step', type=int, default=0,
                    help='upper warm up step limit')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0,
                    help='minimum learning rate during annealing')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--clip_nonemb', action='store_true',
                    help='only clip the gradient of non-embedding params')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='number of epochs to train, default: 10')
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size')
parser.add_argument('--tgt_len', type=int, default=70,
                    help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=50,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--adaptive', action='store_true',
                    help='use adaptive softmax')
parser.add_argument('--div_val', type=int, default=1,
                    help='divident value for adapative input and softmax')
parser.add_argument('--pre_lnorm', action='store_true',
                    help='apply LayerNorm to the input instead of the output')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--eval-interval', type=int, default=4000,
                    help='evaluation interval')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--same_length', action='store_true',
                    help='use the same attn length for all tokens')
parser.add_argument('--attn_type', type=int, default=0,
                    help='attention type. 0 for ours, 1 for Shaw et al,'
                    '2 for Vaswani et al, 3 for Al Rfou et al.')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='use the same pos embeddings after clamp_len')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--gpu0_bsz', type=int, default=-1,
                    help='batch size on gpu 0')
parser.add_argument('--sample_softmax', type=int, default=-1,
                    help='number of samples in sampled softmax')
parser.add_argument('--patience', type=int, default=0,
                    help='patience')
parser.add_argument('--finetune_v2', action='store_true',
                    help='finetune v2')
parser.add_argument('--finetune_v3', action='store_true',
                    help='finetune v3')
args = parser.parse_args()
args.tied = not args.not_tied

if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.ext_len >= 0, 'extended context length must be non-negative'

logging = create_exp_dir(args.work_dir,
    scripts_to_save=['train.py', 'mem_transformer.py'], debug=args.debug)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        logging('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

###############################################################################
# Load data
###############################################################################

class SPVocab(object):

    def __init__(self, sp_model):
        self.sp_model = sp_model

    def get_symbols(self, indices):
        # return self.sp_model.DecodePieces(map(lambda id: self.sp_model.IdToPiece(id), indices.tolist()))
        return [ self.sp_model.IdToPiece(id) for id in indices.tolist() ]

    def len(self):
        return len(self.sp_model)

class SPCorpus(object):
    def __init__(self, dataset_path, sp_model_path, batch_size, tgt_len, device):

        self.batch_size = batch_size
        self.tgt_len    = tgt_len
        self.device     = device

        logging('Loading sentencepiece model from %s' % sp_model_path)
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(sp_model_path)
        logging('Vocabulary size: %d' % len(self.sp_model))

        UNK = '<unk>'
        END_OF_LINE = '<endofline>'
        END_OF_TEXT = '<endoftext>'

        self.id_unk = self.sp_model.PieceToId(UNK)
        self.id_eot = self.sp_model.PieceToId(END_OF_TEXT)
        self.id_eol = self.sp_model.PieceToId(END_OF_LINE)

        logging ("sp ids: %s->%d %s->%d %s->%d" % (UNK, self.id_unk, END_OF_TEXT, self.id_eot, END_OF_LINE, self.id_eol))

        self.vocab = SPVocab(self.sp_model)

        logging(f'Loading dataset from {dataset_path}')
        valid_dataset = np.load(os.path.join(dataset_path, 'valid.npy'))
        train_dataset = np.load(os.path.join(dataset_path, 'train.npy'))
        test_dataset  = np.load(os.path.join(dataset_path, 'test.npy'))

        logging(f'Train      dataset has {len(train_dataset):,} tokens')
        logging(f'Validation dataset has {len(valid_dataset):,} tokens')
        logging(f'test       dataset has {len(test_dataset):,} tokens')

        self.valid = valid_dataset.astype(np.int64)
        self.train = train_dataset.astype(np.int64)
        self.test  = test_dataset.astype (np.int64)

    def get_vocab_size(self):
        return len(self.sp_model)

    def _get_data(self, split):
        if split == 'train':
            return self.train
        elif split == 'valid':
            return self.valid
        return self.test

    def get_num_batches(self, split):

        data = self._get_data(split)

        # number of tgt_len samples we can generate from data
        n_samples = (len(data)-1) // self.tgt_len
        logging ("number of %s samples: %d" % (split, n_samples))

        # Work out how cleanly we can divide the dataset into batch_size parts.
        n_batches = n_samples // self.batch_size
        logging ("number of %s batches: %d" % (split, n_batches))

        return n_batches

    def get_batch(self, split, i_batch):

        data = self._get_data(split)

        n_samples = (len(data)-1) // self.tgt_len

        n_batches = n_samples // self.batch_size

        i_batch = i_batch % n_batches

        # start indices (in tokens) for each batch
        batch_indices = [n_batches * self.tgt_len * bi + i_batch * self.tgt_len for bi in range(self.batch_size)]

        # print (batch_indices)

        batch_x = torch.LongTensor( [ [ data[batch_indices[bi] + ci    ] for bi in range(self.batch_size) ] for ci in range(self.tgt_len) ] ).to(self.device)
        batch_y = torch.LongTensor( [ [ data[batch_indices[bi] + ci + 1] for bi in range(self.batch_size) ] for ci in range(self.tgt_len) ] ).to(self.device)
        return batch_x, batch_y


corpus = SPCorpus(args.data, args.sp_model, args.batch_size, args.tgt_len, device)
ntokens = corpus.get_vocab_size()

n_batches_per_epoch = corpus.get_num_batches('train')

max_step = args.num_epochs * n_batches_per_epoch

# corpus.get_batch('train', 0)
# corpus.get_batch('train', 1)

args.n_token = ntokens

# eval_batch_size = 10
# tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
#     device=device, ext_len=args.ext_len)
# va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len,
#     device=device, ext_len=args.ext_len)
# te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len,
#     device=device, ext_len=args.ext_len)

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]
if args.adaptive:
    # cutoffs = [20000, 40000, 200000]
    cutoffs = [3500, 7500, 37500] # FIXME: make relative to ntokens
    tie_projs += [True] * len(cutoffs)

###############################################################################
# Build the model
###############################################################################
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)

def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout

def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt

model_pt_path = os.path.join(args.work_dir, 'cur_model.pt')

if os.path.exists(model_pt_path):

    logging ("restoring model from %s ..." % model_pt_path)

    with open(model_pt_path, 'rb') as f:
        model = torch.load(f)
    model.apply(update_dropout)
    model.apply(update_dropatt)

else:
    logging ("creating MemTransformerLM from scratch ...")

    model = MemTransformerLM(ntokens, args.n_layer, args.n_head, args.d_model,
        args.d_head, args.d_inner, args.dropout, args.dropatt,
        tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
        tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
        ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
        same_length=args.same_length, attn_type=args.attn_type,
        clamp_len=args.clamp_len, sample_softmax=args.sample_softmax)

    model_args = {
        'ntokens'       : ntokens,
        'n_layer'       : args.n_layer,
        'n_head'        : args.n_head,
        'd_model'       : args.d_model,
        'd_head'        : args.d_head,
        'd_inner'       : args.d_inner,
        'dropout'       : args.dropout,
        'dropatt'       : args.dropatt,
        'tie_weight'    : args.tied,
        'd_embed'       : args.d_embed,
        'div_val'       : args.div_val,
        'tie_projs'     : tie_projs,
        'pre_lnorm'     : args.pre_lnorm,
        'tgt_len'       : args.tgt_len,
        'ext_len'       : args.ext_len,
        'mem_len'       : args.mem_len,
        'cutoffs'       : cutoffs,
        'same_length'   : args.same_length,
        'attn_type'     : args.attn_type,
        'clamp_len'     : args.clamp_len,
        'sample_softmax': args.sample_softmax,
        }

    logging ("model args: %s" % repr(model_args))

    args_path = os.path.join(args.work_dir, 'params.json') 
    with open(args_path, 'w') as argsf:
        argsf.write(json.dumps(model_args))
    logging ("%s written." % args_path)

    model.apply(weights_init)
    model.word_emb.apply(weights_init) # ensure embedding init is not overridden by out_layer in case of weight sharing

args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

para_model = model.to(device)

#### optimizer
if args.optim.lower() == 'sgd':
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SGD(sparse_params, lr=args.lr * 2)
        optimizer = optim.SGD(dense_params, lr=args.lr, momentum=args.mom)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
            momentum=args.mom)
elif args.optim.lower() == 'adam':
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
        optimizer = optim.Adam(dense_params, lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optim.lower() == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

#### scheduler
if args.scheduler == 'cosine':
    # here we do not set eta_min to lr_min to be backward compatible
    # because in previous versions eta_min is default to 0
    # rather than the default value of lr_min 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
        max_step, eta_min=args.eta_min) # should use eta_min arg
    if args.sample_softmax > 0:
        scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(optimizer_sparse,
            max_step, eta_min=args.eta_min) # should use eta_min arg
elif args.scheduler == 'inv_sqrt':
    # originally used for Transformer (in Attention is all you need)
    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == 0 and args.warmup_step == 0:
            return 1.
        else:
            return 1. / (step ** 0.5) if step > args.warmup_step \
                   else step / (args.warmup_step ** 1.5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
elif args.scheduler == 'dev_perf':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
    if args.sample_softmax > 0:
        scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sparse,
            factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
elif args.scheduler == 'constant':
    pass

optimizer_pt_path = os.path.join(args.work_dir, 'cur_optimizer.pt')

if os.path.exists(optimizer_pt_path):

    logging ("loading optimizer from %s ..." % optimizer_pt_path)

    with open(optimizer_pt_path, 'rb') as f:
        opt_state_dict = torch.load(f)
        optimizer.load_state_dict(opt_state_dict)

n_steps_txt_path = os.path.join(args.work_dir, 'n_steps.txt')
n_restart_step = 0
if os.path.exists(n_steps_txt_path):
    with open(n_steps_txt_path, 'r') as f:
        n_restart_step = int(f.read().strip())

logging ("restarting from step #%d" % n_restart_step)

logging('=' * 100)
for k in sorted(args.__dict__):
    logging('    - {} : {}'.format(k, args.__dict__[k]))
logging('=' * 100)
logging('#params = {}'.format(args.n_all_param))
logging('#non emb params = {}'.format(args.n_nonemb_param))

###############################################################################
# Training code
###############################################################################

def evaluate(split):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        model.reset_length(args.eval_tgt_len,
            args.ext_len+args.tgt_len-args.eval_tgt_len, args.mem_len)
    else:
        model.reset_length(args.eval_tgt_len,
            args.ext_len, args.mem_len+args.tgt_len-args.eval_tgt_len)

    max_eval_step = corpus.get_num_batches(split)

    # Evaluation
    total_loss = 0.0
    with torch.no_grad():
        mems = tuple()

        for eval_step in range(max_eval_step):
        # for i, (data, target, seq_len) in enumerate(eval_iter):

            data, target = corpus.get_batch(split, eval_step)

            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += loss.float().item()

            if (eval_step > 0) and (eval_step % 10 == 0):
                print ("| eval batch %6d/%6d [%7.2f%%] | avg loss: %7.3f" % (eval_step, max_eval_step, eval_step * 100.0 / max_eval_step, total_loss / eval_step))

    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    return total_loss / max_eval_step

def save_model(model_name):

    logging ("saving %s model..." % model_name)
    with open(os.path.join(args.work_dir, '%s_model.pt' % model_name), 'wb') as f:
        torch.save(model, f)
    with open(os.path.join(args.work_dir, '%s_optimizer.pt' % model_name), 'wb') as f:
        torch.save(optimizer.state_dict(), f)
    with open(os.path.join(args.work_dir, '%s_state_dict.pt' % model_name), 'wb') as f:
        torch.save (model.state_dict(), f)
    logging ("saving %s model... done." % model_name)


def train():
    # Turn on training mode which enables dropout.
    global train_loss, best_val_loss, eval_start_time, log_start_time, corpus, n_batches_per_epoch, max_step
    model.train()
    mems = tuple()
    sample_from_batch = 0

    for train_step in range(n_restart_step, max_step):

        data, target = corpus.get_batch('train', train_step)

        model.zero_grad()

        # logging ("=====>", data, target, seq_len)
        # logging ("%%%%%% data.shape=", data.shape, "target.shape=", target.shape)

        # logging ( list(zip(data[:, 3].tolist(), corpus.vocab.get_symbols(data[:, 3])  )) )
        # logging ( "====> data   :", data[:, 3] )
        # logging ( "====> target :", target[:, 3] )

        # print ( "====> data   [batch #%02d]: %s" % (sample_from_batch, ' '.join(corpus.vocab.get_symbols(data[:, sample_from_batch]))[:120] ))
        # print ( "====> data   [batch #%02d]: %s" % (sample_from_batch, ' '.join(corpus.vocab.get_symbols(data[:, sample_from_batch])) ))
        # print ( "====> target [batch #%02d]: %s" % (sample_from_batch, ' '.join(corpus.vocab.get_symbols(target[:, sample_from_batch]))[:120] ))

        ret = para_model(data, target, *mems)
        loss, mems = ret[0], ret[1:]
        loss = loss.float().mean().type_as(loss)
        loss.backward()
        train_loss += loss.float().item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        if args.sample_softmax > 0:
            optimizer_sparse.step()

        # step-wise learning rate annealing
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
                if args.sample_softmax > 0:
                    optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
            else:
                if args.scheduler == 'cosine':
                    scheduler.step(train_step)
                    if args.sample_softmax > 0:
                        scheduler_sparse.step(train_step)
        elif args.scheduler == 'inv_sqrt':
            scheduler.step(train_step)

        if (train_step>n_restart_step) and (train_step % args.log_interval == 0):
            cur_loss = train_loss / args.log_interval
            elapsed = time.time() - log_start_time

            epoch = train_step / n_batches_per_epoch + 1

            log_str = '| epoch %2d/%2d batch %6d/%6d [%7.2f%%] | lr %.3g | ms/batch %5.2f | loss %7.3f' % (
                epoch, args.num_epochs, train_step % n_batches_per_epoch, n_batches_per_epoch, train_step * 100.0 / max_step, 
                optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss)

            log_str += ' | ppl {:9.3f}'.format(math.exp(cur_loss))
            logging(log_str)

            json_log_plots.write_event(Path(args.work_dir), step=train_step, loss=cur_loss, lr=optimizer.param_groups[0]['lr']*100000.0)

            train_loss = 0
            log_start_time = time.time()

            sample_from_batch = random.randint(0, args.batch_size-1)

        if (train_step>n_restart_step) and (train_step % args.eval_interval == 0):
            save_model('cur')
            with open(n_steps_txt_path, 'w') as f:
                f.write("%s\n" % train_step)

            logging ("evaluating model...")
            val_loss = evaluate('valid')

            json_log_plots.write_event(Path(args.work_dir), step=train_step, val_loss=val_loss)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                logging ("best valid loss so far.")
                save_model('valid')
                best_val_loss = val_loss

            # # dev-performance based learning rate annealing
            # if args.scheduler == 'dev_perf':
            #     scheduler.step(val_loss)
            #     if args.sample_softmax > 0:
            #         scheduler_sparse.step(val_loss)

            # eval_start_time = time.time()

# Loop over epochs.
train_loss = 0
best_val_loss = None

log_start_time = time.time()
eval_start_time = time.time()

logging ("training starts...")

# At any point you can hit Ctrl + C to break out of training early.
try:
    train()
    logging('Training finished.')
except KeyboardInterrupt:
    logging('-' * 100)
    logging('Exiting from training early')

# FIXME

# logging('Load the best saved model.')
# with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
#     model = torch.load(f)
# para_model = model.to(device)
# 
# logging('Run on test data.')
# test_loss = evaluate(te_iter)
# logging('=' * 100)
# logging('| End of training | test loss {:5.2f} | test ppl {:9.3f}'.format(
#     test_loss, math.exp(test_loss)))
# logging('=' * 100)

