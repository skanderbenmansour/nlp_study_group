import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)

from tqdm.notebook import tqdm
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors

import zipfile
from datetime import datetime
from glob import glob
import os
import json

def process_review(review):
    chars = ['/','\\','>','<','-','br']
    chars.extend('1 2 3 4 5 6 7 8 9 0'.split())
    for char in chars:
        review = review.replace(char,'')
    
    tokens = word_tokenize(review)
    tokens = [t.lower() for t in tokens]
    return tokens

def get_run_version():
    model_dir = '/content/drive/My Drive/colab_data/model_checkpoints/*'
  
    files = glob(model_dir)
    return f'v{str(len(files))}'

def setup_dir(version):
    model_dir = f'/content/drive/My Drive/colab_data/model_checkpoints/{version}'
    os.mkdir(model_dir)
    log_dir = f'/content/drive/My Drive/colab_data/model_checkpoints/{version}/logs'
    os.mkdir(log_dir)
    return model_dir,log_dir

def save_params(version,params):
    param_path = f'/content/drive/My Drive/colab_data/model_checkpoints/{version}/param.json'
    with open(param_path, 'w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

def save_eval(version,results):
    eval_path = f'/content/drive/My Drive/colab_data/model_checkpoints/{version}/eval.json'
    with open(eval_path, 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)


def predict_sentence(sentence, word2idx, glove, model, device):
    vec = make_wv_input(sentence, word2idx, glove)
    vec = vec.to(device)
    h = model.init_hidden()
    h = tuple([each.data for each in h])

    probs, h = model(vec, h)
    pred = probs.argmax().cpu().numpy()

    return pred

def make_wv_input(sentence, word2idx, glove):
    vec = []
    for word in sentence:
        if word in word2idx:
            vec.append(glove[word])
    vec = np.vstack(vec)
    ten = torch.tensor(vec,dtype=torch.float)
    return ten

def prep_input(sentence, word2idx, glove, label, device):
    vec = make_wv_input(sentence, word2idx, glove)

    if label == 0:
        target = torch.tensor([1, 0], dtype=torch.float)
    else:
        target = torch.tensor([0, 1], dtype=torch.float)

    vec, target = vec.to(device), target.to(device)

    return vec, target

def make_loss_plot(loss_history,val_loss_min,model_dir):
    loss_array = np.array(loss_history)
    train_loss = loss_array[:, 0]
    val_loss = loss_array[:, 1]

    x = np.arange(1, loss_array.shape[0] + 1)
    y = train_loss
    plt.plot(x, y, c='blue', label='train loss')

    x = np.arange(1, loss_array.shape[0] + 1)
    y = val_loss
    plt.plot(x, y, c='red', label='val loss')

    x0 = np.array([1, loss_array.shape[0]])
    y0 = np.array([val_loss_min, val_loss_min])
    plt.plot(x0, y0, c='black', label='val loss min', linestyle='--')

    plt.grid(which='both')
    plt.legend()
    fig, ax = plt.gcf(), plt.gca()
    ax.set_xticks(x)
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Mean Loss')

    fig.set_size_inches(20, 7)
    fig_path = os.path.join(model_dir, 'loss.png')
    plt.savefig(fig_path)