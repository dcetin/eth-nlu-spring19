import urllib.request
from itertools import islice
import os
import pickle
import numpy as np

def download_file(filename, url):
    with urllib.request.urlopen(url) as response, open(filename, 'wb') as f:
        shutil.copyfileobj(response, f)

def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, n))
        if not chunk:
            return
        yield chunk
        
def head(it, take=5):
    return list(islice(it, take))

def load_or_compute(fname_cache, func, *args, **kwargs):
    if os.path.isfile(fname_cache):
        with open(fname_cache, 'rb') as file:
            return pickle.load(file)
    else:
        obj = func(*args, **kwargs)
        with open(fname_cache, 'wb') as file:
            pickle.dump(obj, file)
            return obj

def get_bestval_epoch(fname):
    fname += '/training-report.tsv'
    with open(fname) as f:
        content = f.readlines()
    content = np.asarray([[float(a) for a in x.strip().split('\t')] for x in content[1:]])
    return np.argmax(content[:,2])+1