import json

def _adj_sent(data_dir, split, use_cat, cat, not_cat='physics'):
    data = [json.loads(l) for l in open(cat + '_adv_sents_' + split + '.jsonl').readlines()]
    s1 = [d['sent1'] for d in data] 
    s2 = [d['sent2'] for d in data]
    labels = [d['label'] for d in data]  
    ids = [i for i in range(len(data))]
    return s1, s2, labels, ids
  
def adj_sent(data_dir,use_cat=False,cat=None,not_cat=None):
    train_s1, train_s2, train_labels, train_ids = _adj_sent(data_dir, 'train',use_cat,cat,not_cat)
    val_s1, val_s2, val_labels, val_ids   = _adj_sent(data_dir, 'val',use_cat,cat,not_cat)
    test_s1, test_s2, test_labels, test_ids  = _adj_sent(data_dir, 'test',use_cat,cat,not_cat)
    return (train_s1, train_s2, train_labels, train_ids), (val_s1, val_s2, val_labels, val_ids), (test_s1, test_s2, test_labels, test_ids)
  
