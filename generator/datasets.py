import os
import csv
import json
import numpy as np
import random 
import nltk
from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import pkgutil
import encodings

seed = 3535999445

def choose_pair(idx,sentences):
    i  = random.random()
    j  = random.random()
    if i < (1.0/2) or len([s for s in sentences if (s != sentences[idx] and s != sentences[idx+1])]) < 1:
       s1 = sentences[idx]
       s2 = sentences[idx+1]
       if j < (1.0/2):
          return (s1,s2,0)
       else:
          return (s2,s1,1)
    else:
       s1 = sentences[idx]
       s2 = random.sample([s for s in sentences if (s != sentences[idx] and s != sentences[idx+1])],1)
       if j < (1.0/2):
          return (s1,s2,0)
       else:
          return (s2,s1,1)

def check_list(string):
    if type(string) == list:
       return string[0]
    else:
       return string

def load_json(f):
    return [json.loads(l) for l in open(f).readlines()]

from transformers import BertTokenizer 
import nltk 
def align_ex(ex,wikidata,encoder):
    evidence = ex['evidence']
    pointers = [l[2:] for l in evidence[0]]
    contexts = []
    spans = []
    labels = []
    for p in pointers:
        context = [l['text'] for l in wikidata if l['id'] == p[0]][0]
        span = p[1]
        context = nltk.tokenize.sent_tokenize(context)
        sent = encoder.tokenize(context[span])
        context1 = [encoder['[CLS]']] + encoder.tokenize(context[:span])
        context2 = encoder.tokenize(context[span+1:])
        label = ([0] * len(context1)) + ([1] * len(sent)) + ([0] * len(context2))[:500]
        label = label + ([0] * (500 - len(label)))
        labels.append(label)
        contexts.append(context)
        spans.append(span)
        import pdb; pdb.set_trace()
    return contexts, spans, labels

def _factdisc(data_dir, wikidata, encoder, split):
    data = load_json(os.path.join(data_dir, split + '.jsonl'))
    data = [align_exp(ex,wikidata,encoder) for ex in data]
    contexts = [d[0] for d in data]
    spans = [d[1] for d in data]
    labels = [d[2] for d in data]
    return contexts, spans, labels

import itertools 
def factdisc(data_dir):
    wikidata = [load_json('./wiki-pages/' + f) for f in os.listdir('./wiki-pages') if f.endswith('.jsonl')]
    wikidata = list(itertools.chain.from_iterable(wikidata))
    encoder = BertTokenizer.from_pretrained('bert-base-uncased')
    trc, trs, trl = _factdisc(data_dir, wikidata, encoder, 'train')
    vac, vas, val = _factdisc(data_dir, wikidata, encoder, 'val')
    tec, tes, tel = _factdisc(data_dir, wikidata, encoder, 'test')
    return (trc, trs, trl), (vac, vas, val), (tec, tes, tel)

def _arxiv(data_dir, split):
    files = [f for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('.json')]
    s1s =  []
    s2s = []
    labels = []
    for f in files:
        f = open(os.path.join(data_dir, split, f), encoding='utf_8')
        data = json.load(f)
        abstract = data["abstract"]	
        #import ipdb; ipdb.set_trace()
        if (len(abstract) > 0):
           pairs = [choose_pair(i, abstract) for i in range(len(abstract)-1)]
           s1s.extend([p[0] for p in pairs])
           s2s.extend([check_list(p[1]) for p in pairs])
           labels.extend([p[2] for p in pairs])
    return s1s, s2s, labels        

def arxiv(data_dir):
    ts1s, ts2s, tlabels = _arxiv(data_dir, 'train')
    vts1s, vs2s, vlabels = _arxiv(data_dir, 'val')
    tes1s, tes2s, telabels = _arxiv(data_dir, 'test')
    return (ts1s, ts2s, tlabels), (vts1s, vs2s, vlabels), (tes1s, tes2s, telabels)

def _wikihow(data_dir, split,topic=None):
    files = [f for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('.json')]
    abstracts = []
    articles = []
    if topic != None:
       split = split + '_topics'
    for f in files:
        f = open(os.path.join(data_dir, split, f), encoding='utf_8')
        try:
           data = json.load(f)
           if topic != None:
              f_topic = data["topic"]
              if f_topic != topic:
                 continue
        except:
           print(f)
           continue
        abstract = data["headline"]
        article = data["text"]
        abstract = abstract.replace('+',' + ')
        abstract = abstract.replace('-',' - ')
        article = article.replace('+',' + ')
        article = article.replace('-',' - ')
        abstract = abstract.replace('(',' ( ')
        abstract = abstract.replace(')',' ) ')
        abstract = abstract.replace('!',' ! ')
        article = article.replace('(',' ( ')
        article = article.replace(')',' ) ')
        article = article.replace('!',' ! ')

        abstract = abstract.replace('.',' . ')
        abstract = abstract.replace('\n',' ')
        abstract = abstract.lower()
        abstract = abstract.replace("\'","")
        abstract = abstract.replace(','," ")
        article = article.replace('.',' . ')
        article = article.replace('\n',' ')
        article = article.replace('"',' " ')
        article = article.lower()
        article = article.replace("\'"," '")
        article = article.replace(","," , ")
        abstract = abstract.replace('"',' " ')
        article = article.replace('?',' ? ')
        abstract = abstract.replace('?',' ? ')

        article = article.replace('  ',' ')
        abstract = abstract.replace('  ',' ')
        abstracts.append(abstract)
        articles.append(article)
    return articles,abstracts

def wikihow(data_dir,topic=None):
    train_articles, train_abstracts = _wikihow(data_dir, 'train',topic)
    val_articles, val_abstracts   = _wikihow(data_dir, 'val',topic)
    test_articles, test_abstracts = _wikihow(data_dir, 'test',topic)
    return (train_articles, train_abstracts), (val_articles, val_abstracts), (test_articles, test_abstracts)

def _adjacent(data_dir,split):
    files = [f for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('.json')]
    sents = []
    opts = []
    labels = []
    if split == 'train':
       limit = 100000
    else:
       limit = 50000
    for f in files[:limit]:
        f = open(os.path.join(data_dir,split,f),encoding='utf_8')
        data = json.load(f)
        try:
           abstract =  data["abstract"]
        except:
           abstract = data["headline"]
        if (len(abstract) > 2):
           sent = random.sample(list(range(len(abstract))),1)[0]
           try:
               option1 = random.sample([s for s in list(range(len(abstract))) if s == sent + 1],1)[0]
#               option1 = random.sample([s for s in list(range(len(abstract))) if s <= sent + 1 and s  >= sent-1],1)[0]
           except:
               print('issue with opt1')
               import ipdb;  ipdb.set_trace()
           try:
               option2 = random.sample([s for s in list(range(len(abstract))) if s != sent + 1],1)[0]
#               option2 = random.sample([s for s in list(range(len(abstract))) if s > sent + 2 or s < sent - 2],1)[0]
           except:
               print('issue with  opt2')
               print(sent)
               print(abstract)
               print([s for s in list(range(len(abstract))) if s > sent + 1 or s < sent-1])
               import ipdb; ipdb.set_trace()
           #print('original sentence: ' + abstract[sent])
           #print('adjacent sentence:' + abstract[option1])
           #print('non-adjacent sentence:' + abstract[option2])
           #import ipdb; ipdb.set_trace()
           sents.append(abstract[sent])
           opts.append(abstract[option1])
           labels.append(1)
           sents.append(abstract[sent])
           opts.append(abstract[option2])
           labels.append(0)
    return sents,  opts, labels

def adjacent(data_dir):
    tsents, topts, tlabels = _adjacent(data_dir, 'train')
    vsents, vopts, vlabels = _adjacent(data_dir, 'val')
    tesents, teopts, telabels = _adjacent(data_dir, 'test')
    return  (tsents, topts, tlabels), (vsents, vopts, vlabels), (tesents, teopts, telabels)    

def _arxiv2(data_dir, split,use_cat=False,cat=None,not_cat=None):
    files = [f for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('.json') and 'conclusion' not in f]
    abstracts = []
    articles = []
    #if split == 'val':
    #   limit = 500
    #else:
    #   limit = 10000
    for f in files:
        f = open(os.path.join(data_dir, split, f), encoding='utf_8')
        try:
           data = json.load(f)
        except:
           import ipdb; ipdb.set_trace()
        if type(data) == list:
           data = data[0]

        if use_cat == True and cat not in data['category']:
           continue
        if not_cat != None and not_cat in data['category']:
           continue
        abstract = data["abstract"]
        try:
           article = data["article"]
        except:
           article = data["intro"]
        if type(abstract) == str:
           abstracts.append(abstract)
           articles.append(article)
        else:
           abstracts.append(' '.join(abstract))
           articles.append(' '.join(article))
    return articles,abstracts


def _reddit(data_dir, split,use_subreddit=None,subreddit=''):
    files = [f for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('.json')]
    abstracts = []
    articles = []
    for f in files[:40000]:
        print(f)
        f = open(os.path.join(data_dir, split, f),encoding="utf8", errors='ignore')
        data = json.load(f)
        abstract = data["summary"]
        article = data["content"]
        if use_subreddit == "True" and subreddit in data["subreddit"]:
           category = data["subreddit"]
           articles.append(article + ' <sub> '  + category + ' <sub> ')       
        else:
           articles.append(article)
        abstracts.append(abstract)
    return articles,abstracts

def _arxiv3(data_dir, split,use_cat=False,cat=None,not_cat=None):
    files = [f for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('.json') and 'conclusion' not in f]
    abstracts = []
    articles = []
    #if split == 'val':
    #   limit = 500
    #else:
    #   limit = 10000
    for f in files:
        f = open(os.path.join(data_dir, split, f), encoding='utf_8')
        try:
           data = json.load(f)
        except:
           import ipdb; ipdb.set_trace()
        if type(data) == list:
           data = data[0]

        if use_cat == True and cat not in data['category']:
           continue
        if not_cat != None and not_cat in data['category']:
           continue
        abstract = data["abstract"]
        article = data["article"].split(' ')[-350:]
        article = ' '.join(article)
        abstracts.append(abstract)
        articles.append(article)
    return article, abstract


def _arxiv4(data_dir, split,use_cat=False,cat=None,not_cat=None):
    files = [f for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('.json') and 'conclusion' not in f]
    abstracts = []
    articles = []
    #if split == 'val':
    #   limit = 500
    #else:
    #   limit = 10000
    for f in files:
        f = open(os.path.join(data_dir, split, f), encoding='utf_8')
        try:
           data = json.load(f)
        except:
           import ipdb; ipdb.set_trace()
        if type(data) == list:
           data = data[0]

        if use_cat == True and cat not in data['category']:
           continue
        if not_cat != None and not_cat in data['category']:
           continue
        abstract = data["abstract"]
        if len(data["article"].split(' ')) > 350:
           article = ' '.join(data["article"].split(' ')[:175] + data["article"].split(' ')[-175:])
        else:
           article = data["article"]
        abstracts.append(abstract)
        articles.append(article)
    return articles,abstracts

def _arxiv5(data_dir, split,use_cat=False,cat=None,not_cat=None):
    files = [f for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('.json') and 'conclusion' in f]
    abstracts = []
    articles = []
    #if split == 'val':
    #   limit = 500
    #else:
    #   limit = 10000
    for f in files:
        f = open(os.path.join(data_dir, split, f), encoding='utf_8')
        try:
           data = json.load(f)
        except:
           import ipdb; ipdb.set_trace()
        if type(data) == list:
           data = data[0]

        if use_cat == True and cat not in data['category']:
           continue
        if not_cat != None and not_cat in data['category']:
           continue
        abstract = data["abstract"]
        article = ' '.join(data["article"].split(' ')[:250] + data["conclusion"].split(' ')[:250])
        abstracts.append(abstract)
        articles.append(article)
    return articles,abstracts

def _arxiv6(data_dir, split,use_cat=False,cat=None,not_cat=None):
    files = [f for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('.json') and 'conclusion' in f]
    abstracts = []
    articles = []
    #if split == 'val':
    #   limit = 500
    #else:
    #   limit = 10000
    for f in files:
        f = open(os.path.join(data_dir, split, f), encoding='utf_8')
        try:
           data = json.load(f)
        except:
           import ipdb; ipdb.set_trace()
        if type(data) == list:
           data = data[0]

        if use_cat == True and cat not in data['category']:
           continue
        if not_cat != None and not_cat in data['category']:
           continue
        abstract = data["abstract"]
        article =  data["conclusion"]
        abstracts.append(abstract)
        articles.append(article)
    return articles,abstracts

from summa import summarizer

def remove_intro_part(s):
    if '(' and ')' in s:
      return s.split('(')[0] + s.split(')')[1]
    else:
      return s

def _arxiv_textrank(data_dir, split,use_cat=False,cat=None,not_cat=None):
    files = [f for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('.json') and 'conclusion' not in f]
    abstracts = []
    articles = []
    #if split == 'val':
    #   limit = 500
    #else:
    #   limit = 10000
    for f in files:
        f = open(os.path.join(data_dir, split, f), encoding='utf_8')
        try:
           data = json.load(f)
        except:
           import ipdb; ipdb.set_trace()
        if type(data) == list:
           data = data[0]

        if use_cat == True and cat not in data['category']:
           continue
        if not_cat != None and not_cat in data['category']:
           continue
        abstract = data["abstract"]
        article = data["article"]

        article = article.replace(' ? ','')
        article = nltk.tokenize.sent_tokenize(article)
        article = [s for s in article if 'et al' not in s]
        article = [remove_intro_part(s) for s in article]
        article = [s for s in article if len(s) > 3]
        article = ' '.join(article)
        article = summarizer.summarize(article,words=350)
        abstracts.append(abstract)
        articles.append(article)
    return articles,abstracts

def _arxiv7(data_dir,split,use_cat=False,cat=None,not_cat=None,agent='first',full_abstract=True):
    files = [f for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('.json') and 'conclusion' not in f]
    abstracts = []
    articles = []
    for f in files:
        f = open(os.path.join(data_dir, split, f), encoding='utf_8')
        try:
           data = json.load(f)
        except:
           import ipdb; ipdb.set_trace()
        if type(data) == list:
           data = data[0]

        if use_cat == True and cat not in data['category']:
           continue
        if not_cat != None and not_cat in data['category']:
           continue
        if agent == 'first':
           article = data["article"]
           article_len = len(article)
           article = article[:int(article_len/2.0)]
        else:
           article = data["article"]
           article_len = len(article)
           article = article[-int(article_len/2.0):]
        if full_abstract:
           abstract = data["abstract"]
        else:
            if agent == 'first':
               abstract = data["abstract"]
               abstract_len = len(abstract)
               abstract = abstract[:int(abstract_len/2.0)]
            else:
               abstract = data["abstract"]
               abstract_len = len(abstract)
               abstract = abstract[-int(abstract_len/2.0):]
        abstracts.append(abstract)
        articles.append(article)
    return articles,abstracts


def arxiv_textrank(data_dir,use_cat=False,cat=None,not_cat=None):
    train_articles, train_abstracts = _arxiv_textrank(data_dir, 'train',use_cat,cat,not_cat)
    val_articles, val_abstracts   = _arxiv_textrank(data_dir, 'val',use_cat,cat,not_cat)
    test_articles, test_abstracts = _arxiv_textrank(data_dir, 'test',use_cat,cat,not_cat)
    return (train_articles, train_abstracts), (val_articles, val_abstracts), (test_articles, test_abstracts)


def arxiv7(data_dir,use_cat=False,cat=None,not_cat=None,agent='first',full_abstract=True):
    train_articles, train_abstracts = _arxiv7(data_dir, 'train',use_cat,cat,not_cat)
    val_articles, val_abstracts   = _arxiv7(data_dir, 'val',use_cat,cat,not_cat)
    test_articles, test_abstracts = _arxiv7(data_dir, 'test',use_cat,cat,not_cat)
    return (train_articles, train_abstracts), (val_articles, val_abstracts), (test_articles, test_abstracts)

def arxiv2(data_dir,use_cat=False,cat=None,not_cat=None):
    train_articles, train_abstracts = _arxiv2(data_dir, 'train',use_cat,cat,not_cat)
    val_articles, val_abstracts   = _arxiv2(data_dir, 'val',use_cat,cat,not_cat)
    test_articles, test_abstracts = _arxiv2(data_dir, 'test',use_cat,cat,not_cat)
    return (train_articles, train_abstracts), (val_articles, val_abstracts), (test_articles, test_abstracts)

def arxiv3(data_dir,use_cat=False,cat=None,not_cat=None):
    train_articles, train_abstracts = _arxiv3(data_dir, 'train',use_cat,cat,not_cat)
    val_articles, val_abstracts   = _arxiv3(data_dir, 'val',use_cat,cat,not_cat)
    test_articles, test_abstracts = _arxiv3(data_dir, 'test',use_cat,cat,not_cat)
    return (train_articles, train_abstracts), (val_articles, val_abstracts), (test_articles, test_abstracts)

def arxiv4(data_dir,use_cat=False,cat=None,not_cat=None):
    train_articles, train_abstracts = _arxiv4(data_dir, 'train',use_cat,cat,not_cat)
    val_articles, val_abstracts   = _arxiv4(data_dir, 'val',use_cat,cat,not_cat)
    test_articles, test_abstracts = _arxiv4(data_dir, 'test',use_cat,cat,not_cat)
    return (train_articles, train_abstracts), (val_articles, val_abstracts), (test_articles, test_abstracts)

def arxiv5(data_dir,use_cat=False,cat=None,not_cat=None):
    train_articles, train_abstracts = _arxiv5(data_dir, 'train',use_cat,cat,not_cat)
    val_articles, val_abstracts   = _arxiv5(data_dir, 'val',use_cat,cat,not_cat)
    test_articles, test_abstracts = _arxiv5(data_dir, 'test',use_cat,cat,not_cat)
    return (train_articles, train_abstracts), (val_articles, val_abstracts), (test_articles, test_abstracts)

def arxiv6(data_dir,use_cat=False,cat=None,not_cat=None):
    train_articles, train_abstracts = _arxiv6(data_dir, 'train',use_cat,cat,not_cat)
    val_articles, val_abstracts   = _arxiv6(data_dir, 'val',use_cat,cat,not_cat)
    test_articles, test_abstracts = _arxiv6(data_dir, 'test',use_cat,cat,not_cat)
    return (train_articles, train_abstracts), (val_articles, val_abstracts), (test_articles, test_abstracts)

def reddit(data_dir,use_cat=False,cat=None):
    train_articles, train_abstracts = _reddit(data_dir, 'train')
    val_articles, val_abstracts   = _reddit(data_dir, 'valid')
    test_articles, test_abstracts = _reddit(data_dir, 'test')
    return (train_articles, train_abstracts), (val_articles, val_abstracts), (test_articles, test_abstracts)

def _rocstories(path):
    with open(path, encoding='utf_8') as f:
        f = csv.reader(f)
        st = []
        ct1 = []
        ct2 = []
        y = []
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:
                s = ' '.join(line[1:5])
                c1 = line[5]
                c2 = line[6]
                st.append(s)
                ct1.append(c1)
                ct2.append(c2)
                y.append(int(line[-1])-1)
        return st, ct1, ct2, y

def rocstories(data_dir, n_train=1497, n_valid=374):
    storys, comps1, comps2, ys = _rocstories(os.path.join(data_dir, 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'))
    teX1, teX2, teX3, _ = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys = train_test_split(storys, comps1, comps2, ys, test_size=n_valid, random_state=seed)
    trX1, trX2, trX3 = [], [], []
    trY = []
    for s, c1, c2, y in zip(tr_storys, tr_comps1, tr_comps2, tr_ys):
        trX1.append(s)
        trX2.append(c1)
        trX3.append(c2)
        trY.append(y)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    for s, c1, c2, y in zip(va_storys, va_comps1, va_comps2, va_ys):
        vaX1.append(s)
        vaX2.append(c1)
        vaX3.append(c2)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)
