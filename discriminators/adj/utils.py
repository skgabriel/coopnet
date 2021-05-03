import os
import sys
import json
import time
import numpy as np

def encode_dataset_bert_sent(*splits, encoder):
    encoded_splits = []
    for split in splits:
        fields = []
        for field in split:
            if isinstance(field[0], str):
                field = [encoder.encode(x) for x in field]
            fields.append(field)
        encoded_splits.append(fields)
    return encoded_splits
  
def iter_data(*datas, n_batch=128, truncate=False, verbose=False, max_batches=float("inf")):
    n = len(datas[0])
    if truncate:
        n = (n//n_batch)*n_batch
    n = min(n, max_batches*n_batch)
    n_batches = 0
    if verbose:
        f = sys.stderr
    else:
        f = open(os.devnull, 'w')
    for i in tqdm(range(0, n, n_batch), total=n//n_batch, file=f, ncols=80, leave=False):
        if n_batches >= max_batches: raise StopIteration
        if len(datas) == 1:
            yield datas[0][i:i+n_batch]
        else:
            yield (d[i:i+n_batch] for d in datas)
        n_batches += 1
        
class ResultLogger(object):
    def __init__(self, path, *args, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.time()
        self.f_log = open(make_path(path), 'w')
        self.f_log.write(json.dumps(kwargs)+'\n')

    def log(self, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.time()
        self.f_log.write(json.dumps(kwargs)+'\n')
        self.f_log.flush()

    def close(self):
        self.f_log.close()
        
def make_path(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return f
