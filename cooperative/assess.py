import argparse
import json
import numpy as np
import ast
from operator import add

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

subset = args.subset 
gen_folder = args.gen_folder
source_folder = args.source_folder

file = [json.loads(l) for l in open('results_' + subset + '.json').readlines()]

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
    fact_a = np.argmax(list( map(add, [l * delta for l in f['prob']],[l* delta for l in f['fact']])))
    f_fact = f['files'][fact_a]
    f_prob = f['files'][prob_a]
    f_adj = f['files'][adj_a]
    f_cov = f['files'][cov_a]
    f_order = f['files'][order_a]
    retrieved = [f_prob,f_adj,f_cov,f_order]
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
