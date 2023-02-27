import os
import sys
import numpy as np
from tqdm import tqdm
import pickle
import json
import argparse
import pickle
import pandas as pd
from transformers import BartTokenizer
from trie import Trie

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

with open('../benchmarks/aap/target_kb.json', 'r') as f:
    cui2str = json.load(f)

entities = []
for cui in cui2str:
    entities += cui2str[cui]
trie = Trie([16]+list(tokenizer(' ' + entity.lower())['input_ids'][1:]) for entity in tqdm(entities)).trie_dict
with open('../benchmarks/aap/trie.pkl', 'wb') as w_f:
    pickle.dump(trie, w_f)
print("finish running!")
