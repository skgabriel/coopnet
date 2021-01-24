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
    parser.add_argument('--save_folder',type=str,default='')
    parser.add_argument('--data_dir',type=str,default='/net/nfs2.corp/ca-data/')
    parser.add_argument('--n_batch',type=int,default=1)
    parser.add_argument('--topk',type=int,default=3)
    parser.add_argument('--num_cands',type=int,default=10)
    parser.add_argument('--split',type=str,default='val') #test or val?
    parser.add_argument('--decoding',type=str,default='topk')
    parser.add_argument('--s_split',type=int,default=0)
    parser.add_argument('--e_split',type=int,default=300)
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
    delimiter = [encoder['<|TL;DR|>']] #[encoder['ĠTL']] + [encoder[';']] + [encoder['DR']]
    end_token = [encoder['<|endoftext|>']] #[encoder['<']] + [encoder['end']] + [encoder['>']]
    xmb = np.zeros((n_batch, n_ctx), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    max_len = 256-1 #512-1  
    for i, (x1,x2), in enumerate(zip(X1,X2)):
        new_x1 = x1[:800] #[:400]
        new_x2 = x2[:200] #[:(n_ctx-400-6)]
        x12 = new_x1 + delimiter
        x13 = new_x2 + end_token #+ [clf_token]
        try:
            xmb[i,:len(x12)] = x12
            xmb[i,len(x12):len(x12)+len(x13)] = x13 
            mmb[i,:len(x12)] = 1
            mmb[i,:len(x12)+len(x13)] = 1
        except:
            import ipdb; ipdb.set_trace()
#    import ipdb; ipdb.set_trace()
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

if args.dataset == 'bio':
   unprocessed_p = 'all_bio.pkl' 
   model_path = '/net/nfs2.corp/ca-data/factsumm/bio_gpt2/model/' + best_model

if args.dataset == 'cs':
   unprocessed_p = 'data_dump_comsci.pkl'
   model_path = '/net/nfs2.corp/ca-data/factsumm/cs_gpt2/model/' + best_model

if args.dataset == 'aan':
   unprocessed_p = 'all_aan.pkl'
   model_path = '/net/nfs2.corp/ca-data/factsumm/aan_gpt2/model/' + best_model

if args.dataset != 'aan':
   unprocessed_p = os.path.join(args.data_dir, 'gpt2_' + unprocessed_p)
else:
   unprocessed_p = os.path.join(args.data_dir, unprocessed_p)
print('loading transformed data')

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
         (teX1, teX2, te_ids)) = encode_dataset2(*arxiv2(args.data_dir, use_cat=True, cat=args.dataset,not_cat='physics'),encoder=text_encoder)
        pickle.dump([(trX1,trX2,tr_ids), (vaX1,vaX2,va_ids), (teX1, teX2,te_ids)], open(unprocessed_p,'wb'))
   teX, teM = transform_arxiv(teX1, teX2)
   pickle.dump((teX,teM, te_ids), open(os.path.join(args.data_dir, args.split + '_gpt2_' + args.dataset + '.pkl'),'wb'))

n_gpu = 1
print("device", device, "n_gpu", n_gpu)
n_updates = 0
n_batch_test = args.n_batch
gen_len = 150


def topk(model, XMB,i, n=1,k=args.topk, mem=None,use_pointer=None,use_scores=None):
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
for xmb, id in iter_data(teX,te_ids,n_batch=n_batch_test,truncate=True,verbose=True):
    if args.dataset == 'aan':
       id = id[0]
       id = [decoder[t] for t in id]
       id = ''.join(id)
#    if str(id) + '.txt' not in os.listdir('./' + output_folder):
#       print(str(id) + '.txt')
#    continue
#       continue
#    else:
#       print(str(id) + '.txt')
    if n_updates < args.s_split:
       n_updates += 1
       continue
    if n_updates == args.e_split:
       break
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



