import sys
import os
from nltk.corpus import wordnet
from pathlib import Path

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(CURR_DIR)
P = PATH.parent
for i in range(3): # add parent path, height = 3
    P = P.parent
    sys.path.append(str(P.absolute()))
import pickle
from dataclasses import dataclass
import random
import torch
import numpy as np
import transformers

def getDefinition(word):
    syns = wordnet.synsets(word)
    first_word = ""
    definitions = []
    for idx,syn in enumerate(syns):
        if idx==0:
            first_word = syn.name().split(".")[0].replace('_'," ")
        if syn.name().split(".")[0].replace('_'," ")==first_word:
            definitions.append(syn.definition())
    if len(definitions)==0:
        for idx,syn in enumerate(syns):
            definitions.append(syn.definition())
    return definitions


def obtainSynonymAndDefinition(word:str,pos='v'):
    syns = wordnet.synsets(word)
    synAndDefinition = []
    synonymWords = []
    for syn in syns:
        pos_ = syn.name().split(".")[1]
        if pos_!=pos:
            continue
        synAndDefinition.append(
            (
                syn.name().split(".")[0].replace('_'," "),
                syn.definition()
                )
        )
        
        
        if synAndDefinition[-1][0] not in synonymWords and synAndDefinition[-1][0]!=word:
            synonymWords.append(synAndDefinition[-1][0])
    return synAndDefinition,synonymWords

def save(obj, path_name):
    print("save to(保存到) ",path_name)
    with open(path_name, 'wb') as file:
        pickle.dump(obj, file)

def load(path_name: object) -> object:
    with open(path_name, 'rb') as file:
        return pickle.load(file)


def set_global_random_seed(seed):
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


@dataclass
class MDFeat:
    sentence: str
    verb_idx: int
    verb:str
    label: int
    addinfo1: object
    addinfo2: object
