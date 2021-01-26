import os
import json

seed = 3535999445

#data loading helper
#specify domain category and any overlapping category
def _arxiv2(data_dir, split,use_cat=False,cat=None,not_cat=None):
    files = [f for f in os.listdir(os.path.join(data_dir, split)) if f.endswith('.json')]
    abstracts = []
    articles = []
    ids = []
    for f in files:
        f = open(os.path.join(data_dir, split, f), encoding='utf_8')
        data = json.load(f)
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
        ids.append(data['id'])
    return articles,abstracts, ids

#data loading 
def arxiv2(data_dir,use_cat=False,cat=None,not_cat=None):
    train_articles, train_abstracts, train_ids = _arxiv2(data_dir, 'train',use_cat,cat,not_cat)
    val_articles, val_abstracts, val_ids   = _arxiv2(data_dir, 'val',use_cat,cat,not_cat)
    test_articles, test_abstracts, test_ids = _arxiv2(data_dir, 'test',use_cat,cat,not_cat)
    return (train_articles, train_abstracts, train_ids), (val_articles, val_abstracts, val_ids), (test_articles, test_abstracts, test_ids)
