import argparse
import os
import json
import torch
import torch.nn.functional as F
from transformers import *
import numpy as np
import json
import ast

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset',type=str,default='aan')
    parser.add_argument('--gen_folder',type=str,default='./gens')
    parser.add_argument('--source_folder', type=str, default='./sources')
    parser.add_argument('--aan_model_path',type=str,default='./aan_model/model/best_params') 
    parser.add_argument('--bio_model_path',type=str,default='./bio_model/model/best_params')
    parser.add_argument('--gen_model_path',type=str,default='./gen_model/aan/model/best_params')
args = parser.parse_args()
print(args)

def clean_json(s):
    s = ast.literal_eval(s)
    return s

def truncation(s,tokenizer):
    if tokenizer.decode(s[-1]) != '[SEP]':
       return s + tokenizer.encode('[SEP]')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

stopwords = [l.strip() for l in open('stopwords.txt').readlines()]

#specify dataset subset 
subset = args.subset

id_key = 'id'
if subset == 'aan':
   id_key = '_file_id'

gen_folder = args.gen_folder
source_file = [clean_json(l) for l in open(args.source_folder + '/' + 'test_' + subset + '.txt').readlines()]
source_file = dict([(d[id_key].replace('test-',''),d['article']) for d in source_file])

all_files = [f for f in os.listdir(gen_folder)]
files = [f for f in os.listdir(gen_folder) if f.split('_')[-2] == '0']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dh_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
dh_model.bert = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')
dh_model = dh_model.to(device)
if subset == 'aan' or subset == 'cs':
   disc_path = args.aan_model_path
else:
   disc_path = args.bio_model_path
dh_model.load_state_dict(torch.load(disc_path))
dh_model.eval()
disc_encoder = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')


text_encoder = GPT2Tokenizer.from_pretrained('gpt2-medium')
encoder = text_encoder.encoder
encoder['<|TL;DR|>'] = len(encoder)
decoder = text_encoder.decoder
decoder[len(decoder)] = '<|TL;DR|>'
text_encoder.encoder = encoder
text_encoder.decoder = decoder
n_vocab =  len(text_encoder.encoder)

model_path = args.gen_model_path
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model.resize_token_embeddings(n_vocab)
model = model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

def disc_score(abstract, model):
    score_ = []
    for sent in range(len(abstract)-1):
        sent1 = abstract[sent]
        sent2 = abstract[sent+1]
        input_ = torch.LongTensor(truncation(disc_encoder.encode(sent1)[:200],disc_encoder) + truncation(disc_encoder.encode(sent2)[1:201],disc_encoder)).unsqueeze(0).to(device)
        output_ = np.log(F.softmax(model(input_)[0]).tolist()[0][1])
        score_.append(output_)
    return np.mean(score_)

def ordering(abstract):
    abstract = abstract[:10]

    score = 0

    if abstract[0] == 'background_label':
       score += 1
    else:
       score -= 1

    if abstract[-1] == 'result_label':
       score += 1
    else:
       score -= 1

    for sent in range(1,len(abstract)):
        if abstract[sent] == 'background_label':
           if abstract[sent-1] == 'background_label':
              score += 1
           else:
              score -= 1
        if abstract[sent] == 'method_label':
           if abstract[sent-1] == 'background_label' or abstract[sent-1] == 'method_label' or abstract[sent-1] == 'objective_label':
              score += 1
           else:
              score -= 1

        if abstract[sent] == 'objective_label':
           if abstract[sent-1] == 'background_label' or abstract[sent-1] == 'objective_label' or abstract[sent-1] == 'method_label':
              score += 1
           else:
              score -= 1

        if abstract[sent] == 'result_label':
           if abstract[sent-1] == 'objective_label' or abstract[sent-1] == 'method_label' or abstract[sent-1] == 'other_label':
              score += 1
           else:
              score -= 1
    normalized_score = np.log((score + 11.0)/22.0)
    return normalized_score


def replace_stop(w):
    if w in stopwords:
       return ' '
    else:
       return w

def clean(phrase):
    if phrase[-1] == ' ':
       phrase = phrase[:-1]
    if len(phrase) < 1:
       return phrase
    if phrase[0] == ' ':
       phrase = phrase[1:]
    phrase = [replace_stop(w) for w in phrase.split(' ')]
    phrase = ' '.join(phrase)
    phrase = phrase.replace('  ','<split>')
    phrase = phrase.split('<split>')
    phrase = [p for p in phrase if len(p) > 1]
    return phrase

import itertools

def factual(abstract,scores, source):
    threshold = abs(max(scores)-.1)
    keep = [0] * len(abstract)
    for word in range(len(abstract)):
        if scores[word] >= threshold or (word > 0 and keep[word-1] == 1 and scores[word] >= threshold-.05):
           keep[word] = 1
        if "##" in abstract[word] and word != 0 and keep[word-1] == 1:
           keep[word] = 1
    phrase = ''
    phrases = []
    for word in range(len(keep)):
        if keep[word] == 1:
           if "##" in abstract[word]:
              phrase += abstract[word].replace("##","")
           else:
              if phrase != '':
                 phrase += ' '
              phrase += abstract[word]
        else:
           if phrase != '':
              phrases.append(phrase)
              phrase = ''
    if phrase != '':
       phrases.append(phrase)
    phrases = [clean(p.replace('.','').replace(",","")) for p in phrases if len(p) > 1]
    phrases = list(set(itertools.chain.from_iterable(phrases)))
    if len(phrases) == 0:
       return 0
    score = np.log(len([p for p in phrases if p in source])/float(len(phrases)))
    return score

def model_score(abstract, source, model):
    input_ = torch.LongTensor(text_encoder.encode(source)[:800] + [text_encoder.encoder['<|TL;DR|>']] + text_encoder.encode(' '.join(abstract))[:200] + [text_encoder.encoder['<|endoftext|>']]).to(device)
    loss = model(input_,labels=input_)
    loss = -float(loss[0])
    return loss

#BACKGROUND, METHOD, OBJECTIVE, RESULT, OTHER
results = open('results_' + subset + '.json','w')
already = []
for f in files:
    id = f.split('_')[-1].replace('.json','')
    if id in already:
       continue
    already.append(id)
    names = [f for f in all_files if f.split('_')[-1].replace('.json','') == id]
    cands = [json.load(open(gen_folder + '/' + f)) for f in names]
    source = source_file[id]
    if type(source) == list:
       source = ' '.join(source)
    disc_labels = [data['labels'] for data in cands]
    abstract = [data['sentences'] for data in cands]
    prob = [model_score(abstract[i], source, model) for i in range(len(cands))]
    score = [disc_score(abstract[i], dh_model) for i in range(len(cands))]
    coverage = [np.log(len(set(disc_labels[i]))/5.0) for i in range(len(cands))]
    order = [ordering(disc_labels[i]) for i in range(len(cands))]
    f_score = [factual(data['tokens'],data['scores'],source) for data in cands]
    entry = {"id":id,"files":names,"prob":prob,"adj":score,"cov":coverage,"order":order,"fact":f_score}
    results.write(str(json.dumps(entry)) + '\n')
