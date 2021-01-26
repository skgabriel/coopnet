import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from text_utils import TextEncoder
import pickle
import numpy as np
from datasets import arxiv2
from utils import (encode_dataset2, iter_data, ResultLogger, make_path)
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
from sklearn.utils import shuffle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_epoch',type=str,default='12')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset',type=str,default='bio') #specify dataset name
    parser.add_argument('--load_disc',type=bool,default=False)
    parser.add_argument('--data_dir',type=str,default='../data')
    parser.add_argument('--n_batch',type=int,default=1)
    parser.add_argument('--topk',type=int,default=3)
    parser.add_argument('--num_cands',type=int,default=10)
    parser.add_argument('--split',type=str,default='test')
    parser.add_argument('--decoding',type=str,default='topk')
args = parser.parse_args()
print(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
n_ctx = 1024

def transform_arxiv(X1,X2):
    n_batch = len(X1)
    delimiter = [encoder['<|TL;DR|>']]
    end_token = [encoder['<|endoftext|>']]
    xmb = np.zeros((n_batch, n_ctx), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    for i, (x1,x2), in enumerate(zip(X1,X2)):
        new_x1 = x1[:800]
        new_x2 = x2[:200]
        x12 = new_x1 + delimiter
        x13 = new_x2 + end_token
        xmb[i,:len(x12)] = x12
        xmb[i,len(x12):len(x12)+len(x13)] = x13 
        mmb[i,:len(x12)] = 1
        mmb[i,:len(x12)+len(x13)] = 1
    return xmb, mmb

device = torch.device(device)
text_encoder = GPT2Tokenizer.from_pretrained('gpt2-medium')
encoder = text_encoder.encoder
encoder['<|TL;DR|>'] = len(encoder)
decoder = text_encoder.decoder
decoder[len(decoder)] = '<|TL;DR|>'

text_encoder.encoder = encoder
text_encoder.decoder = decoder
n_vocab = len(text_encoder.encoder)

best_model = 'best_params_' + args.load_epoch
unprocessed_p = os.path.join(args.data_dir, 'all_' + args.dataset + '.pkl')
model_path = './' + args.model_dir + '/' + 'model/' + best_model
print('loading transformed data')
if args.dataset == 'cs':
   not_cat = 'physics'
else:
   not_cat = None

try:
   teX, teM, te_ids =  pickle.load(open(os.path.join(args.data_dir, args.split + '_gpt2_' + args.dataset + '.pkl'),'rb'))
except:
   try:
        data_dump = pickle.load(open(unprocessed_p,'rb'))
        if args.split == 'test':
           teX1, teX2, te_ids = data_dump[2]
        else:
           teX1, teX2, te_ids = data_dump[1]
   except:
        ((trX1, trX2, tr_ids),
         (vaX1, vaX2, va_ids),
         (teX1, teX2, te_ids)) = encode_dataset2(*arxiv2(args.data_dir, use_cat=True, cat=args.dataset,not_cat=not_cat),encoder=text_encoder)
        pickle.dump([(trX1,trX2,tr_ids), (vaX1,vaX2,va_ids), (teX1, teX2,te_ids)], open(unprocessed_p,'wb'))
   teX, teM = transform_arxiv(teX1, teX2)
   pickle.dump((teX,teM, te_ids), open(os.path.join(args.data_dir, args.split + '_gpt2_' + args.dataset + '.pkl'),'wb'))

n_gpu = 1
print("device", device, "n_gpu", n_gpu)
n_updates = 0
n_batch_test = args.n_batch
gen_len = 150


def topk(model, XMB,i, n=1,k=args.topk):
    import copy
    probs = [[] for  j in range(n)]
    gens = []
    XMB_intro = torch.stack([copy.deepcopy(XMB[0]) for j in range(n)],dim=0)
    seq_done = [False for j in range(n)]
    for step in range(gen_len):
        logits, _ =  model(XMB_intro[:,:i+1+step])
        logits = torch.nn.functional.softmax(logits[:,-1,:], dim=-1)
        for j in range(n):
            values, indices  = logits[j].unsqueeze(0).sort(descending=True)
            next_indices = indices[:, :k].gather(-1, torch.multinomial(values[:, :k], 1))
            XMB_intro[j,i+1+step] = next_indices.view(-1).long() 
            probs[j].append(np.log(float(values[:,int(XMB_intro[j,i+1+step])])))
            if encoder['<|endoftext|>'] in XMB_intro[j]:
               seq_done[j] = True
        if False not in seq_done:
           break
    for j in range(XMB_intro.size(0)):
        gens.append(XMB_intro[j,i+1:].tolist())
    probs = [np.mean(p) for p in probs]
    return gens, probs


model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model.resize_token_embeddings(n_vocab)
model = model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

 
def get_token(next_idx):
    try:
       return text_encoder.decoder[next_idx]
    except:
       return next_idx

def clean_gen(gen):
    if type(gen) != list:
       gen = gen.tolist()
    gen = [w for w in gen if w != 0]
    gen = [get_token(idx) for idx in gen]
    if '<unk>' in gen:
       gen = [t for t in gen if t != '<unk>']
    gen = "".join([word.replace("Ġ", " ") for word in gen])
    gen = gen.replace("<|endoftext|>","")
    return gen

n_updates = 0 
output_folder = args.dataset + '_gpt2_gen' + '/' + args.split
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for xmb, id in iter_data(teX,te_ids,n_batch=n_batch_test,truncate=True,verbose=True):
    if args.dataset == 'aan':
       id = id[0]
       id = [decoder[t] for t in id]
       id = ''.join(id)
    if args.dataset != 'aan':
       id = id[0]
    gen_file = open(os.path.join('./' + output_folder, str(id) + '.txt'),'w')
    i_1 = 0
    while xmb[0,i_1] != encoder['<|TL;DR|>']:
       i_1 += 1 
    xmb = xmb[0,:i_1+1] 
    new_xmb = np.zeros((1, len(xmb) + gen_len))
    new_xmb[0,:i_1+1] = xmb 
    XMB_intro = torch.Tensor(new_xmb).long().to(device)
    with torch.no_grad():
          gens, probs = topk(model, XMB_intro, i=i_1, k=args.topk, n=args.num_cands)
    gens = [clean_gen(g) for g in gens]
    for i in range(len(gens)):
        gen_file.write(gens[i] + '\n' + 'prob: ' + str(probs[i]) + '\n')
    n_updates += 1



