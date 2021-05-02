import json
import numpy as np
import ast
from operator import add

subset = 'aan'
model = 'gpt2' #lstm #'gpt2'
key_ = '_full'
if model == 'lstm':
   key_ = '_lstm'
gen_folder = './final_eval_gens/' + model + '/' + subset + '/'
source_folder = './final_eval_gens/' + subset + '_gens' + key_

if model == 'gpt2':
   file = [json.loads(l) for l in open('results_' + subset + '.json').readlines()] #[ast.literal_eval(l.strip().replace("'", '"')) for l in open('results_' + subset + '.json').readlines()]
else:
   file = [json.loads(l) for l in open('results_lstm_' + subset + '.json').readlines()]

delta = .5
folders = ['org','adj','cov','order']
if model == 'gpt2':
   folders.append('fact')
for f in file:
    prob_a = np.argmax(f['prob'])
    if f['fact'] == 0:
       f['fact'] = [0] * len(f['prob'])
    adj_a = np.argmax(list( map(add, [l * delta for l in f['prob']],[l* delta for l in f['adj']])))
    cov_a = np.argmax(list( map(add, [l * delta for l in f['prob']],[l* delta for l in f['cov']])))
    order_a = np.argmax(list( map(add, [l * delta for l in f['prob']],[l* delta for l in f['order']])))
    if model == 'gpt2':
       fact_a = np.argmax(list( map(add, [l * delta for l in f['prob']],[l* delta for l in f['fact']])))
       f_fact = f['files'][fact_a]
    f_prob = f['files'][prob_a]
    f_adj = f['files'][adj_a]
    f_cov = f['files'][cov_a]
    f_order = f['files'][order_a]
    retrieved = [f_prob,f_adj,f_cov,f_order]
    if model == 'gpt2':
       retrieved.append(f_fact)
    for r in range(len(retrieved)):
        summ = json.load(open(source_folder + '/' + retrieved[r]))
        id = f['id']
        summ = summ['sentences']
        if subset == 'aan':
           id = str(abs(ord(id[0]))) + id[1:].replace('-','')
        write_ = open(gen_folder + folders[r] + '/' + 'test-' + id + '.txt','w')
        for s in summ:
            write_.write(s + '\n')
