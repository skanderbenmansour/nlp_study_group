import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
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
import pickle
from datetime import datetime
from glob import glob
import os
import json
import random

from torch.nn.utils.rnn import pack_sequence,pad_sequence,pack_padded_sequence,pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append('../')
from py_files import models

import csv

def process_review(review):
    chars = ['/','\\','>','<','-','br']
    chars.extend('1 2 3 4 5 6 7 8 9 0'.split())
    for char in chars:
        review = review.replace(char,'')
    
    tokens = word_tokenize(review)
    tokens = [t.lower() for t in tokens]
    return tokens

def get_run_version(checkpoint_dir='/content/drive/My Drive/colab_data/model_checkpoints'):
    model_dir = os.path.join(checkpoint_dir,'*')
  
    files = glob(model_dir)
    return f'v{str(len(files))}'

def setup_dir(version,checkpoint_dir='/content/drive/My Drive/colab_data/model_checkpoints'):
    model_dir = os.path.join(checkpoint_dir, f'{version}')
    os.mkdir(model_dir)
    log_dir = os.path.join(checkpoint_dir, f'{version}/logs')
    os.mkdir(log_dir)
    return model_dir,log_dir

def save_params(version,params,checkpoint_dir='/content/drive/My Drive/colab_data/model_checkpoints'):
    param_path = os.path.join(checkpoint_dir, f'{version}/param.json')
    with open(param_path, 'w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

def save_eval(version,results,checkpoint_dir='/content/drive/My Drive/colab_data/model_checkpoints'):
    eval_path = os.path.join(checkpoint_dir, f'{version}/eval.json')
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

def label_to_tensor(label):
    if label == 0:
        target = torch.tensor([1, 0], dtype=torch.float)
    else:
        target = torch.tensor([0, 1], dtype=torch.float)
    return target

def prep_input(sentence, word2idx, glove, label, device):
    vec = make_wv_input(sentence, word2idx, glove)
    target = label_to_tensor(label)

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

def load_processed_reviews(data_path):
    pickle_path = os.path.join(data_path, 'processed_data.pickle')
    with open(pickle_path, 'rb') as f:
        train_data, test_data, val_data = pickle.load(f)
    return train_data, test_data, val_data


def load_and_process_stanford_data(data_path, save=False):
    zf = zipfile.ZipFile(data_path + 'train.csv.zip')
    train = pd.read_csv(zf.open('train.csv'))

    zf = zipfile.ZipFile(data_path + 'test.csv.zip')
    test_val = pd.read_csv(zf.open('test.csv'))

    val = pd.concat([test_val[:2500], test_val[12500:12500 + 2500]])
    test = pd.concat([test_val[2500:12500], test_val[12500 + 2500:]])

    processed_list = []

    for data in [train,test,val]:
        labels = list(data.sentiment.values)
        reviews = list(data.review.values)

        all_words = [process_review(review) for review in tqdm(reviews)]

        processed_data = list(zip(all_words, labels))
        random.shuffle(processed_data)
        processed_list.append(processed_data)

    train_data, test_data, val_data = processed_list

    if save:
        pickle_path = os.path.join(data_path, 'processed_data.pickle')
        with open(pickle_path, 'wb') as f:
            pickle.dump((train_data, test_data, val_data), f)

    return train_data, test_data, val_data


def load_and_process_w2v_data(data_path, save=False):
    zf = zipfile.ZipFile(data_path + 'labeledTrainData.tsv.zip')
    df = pd.read_csv(zf.open('labeledTrainData.tsv'),sep='\t')
    train = df[:15000]
    val = df[15000:20000]
    test = df[20000:]

    processed_list = []

    for data in [train,test,val]:
        labels = list(data.sentiment.values)
        reviews = list(data.review.values)

        all_words = [process_review(review) for review in tqdm(reviews)]

        processed_data = list(zip(all_words, labels))
        random.shuffle(processed_data)
        processed_list.append(processed_data)

    train_data, test_data, val_data = processed_list

    if save:
        pickle_path = os.path.join(data_path, 'processed_data.pickle')
        with open(pickle_path, 'wb') as f:
            pickle.dump((train_data, test_data, val_data), f)

    return train_data, test_data, val_data

def load_tensor_data(data,word2idx):
    idx_tensors = []
    label_tensors = []
    for sentence, label in data:
        idx_tensors.append(sentence_to_idx(sentence,word2idx))
        label_tensors.append(label_to_tensor(label))

    lengths = [ten.shape[0] for ten in idx_tensors]
    lengths_tensor = torch.tensor(lengths)
    label_tensor = torch.stack(label_tensors)

    padded = pad_sequence(idx_tensors, padding_value=99999, batch_first=True)

    tensor_data = TensorDataset(padded, lengths_tensor, label_tensor)
    return tensor_data

def load_all_tensor_data(train,test,val,batch_size,word2idx):
    train_dataset = load_tensor_data(train,word2idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = load_tensor_data(test,word2idx)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = load_tensor_data(val,word2idx)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader,test_loader,val_loader

def sentence_to_idx(sentence,word2idx,max_len=1000):
    idx = [word2idx[word] for word in sentence if word in word2idx]
    return torch.tensor(idx[:max_len])

def predict_sentence_batch(sentence, word2idx, model, device, is_cuda):
    idx_tensors = []
    idx_tensors.append(sentence_to_idx(sentence, word2idx))
    idx_tensors.extend([torch.zeros(1)] * (model.batch_size - 1))

    lengths_list = [ten.shape[0] for ten in idx_tensors]
    lengths = torch.tensor(lengths_list)

    inputs = pad_sequence(idx_tensors, padding_value=99999, batch_first=True)


    h = model.init_hidden()
    inputs, lengths = inputs.to(device), lengths.to(device)
    probs, h = model(inputs, lengths, h)

    first_sentence_prob = probs[0]

    if is_cuda:
        pred = first_sentence_prob.argmax().cpu().numpy()
    else:
        pred = first_sentence_prob.argmax().detach().numpy()

    return pred


def create_inference_model(model_version,model_name,glove,is_cuda,device,checkpoint_dir='/content/drive/My Drive/colab_data/model_checkpoints'):
    model_dir = os.path.join(checkpoint_dir,model_version)
    param_path = os.path.join(model_dir,'param.json')
    with open(param_path,'r') as f:
        params = json.load(f)

    num_labels = params.get('num_labels')
    vocab_size = params.get('vocab_size')
    embedding_dim = params.get('embedding_dim')
    num_layers = params.get('num_layers')
    hidden = params.get('hidden')
    p_dropout = params.get('p_dropout')
    batch_size = params.get('batch_size')

    model = models.lstm_clf_batch(num_labels, vocab_size, embedding_dim, num_layers, hidden, batch_size, glove.vectors,
                                  device, p_dropout)
    batch_size = 1

    inf_model = models.lstm_clf_batch(num_labels, vocab_size, embedding_dim, num_layers, hidden, batch_size, glove.vectors,
                                  device, p_dropout)
    inf_model.to(device)

    load_path = os.path.join(model_dir, model_name)
    if is_cuda:
        model.load_state_dict(torch.load(load_path))
    else:
        model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))

    inf_model.load_state_dict(model.state_dict())
    return inf_model

def predict_from_inf_model(sentence, word2idx, inf_model, device, is_cuda, return_prob=False):
    idx_tensors = []
    idx_tensors.append(sentence_to_idx(sentence, word2idx))

    lengths_list = [ten.shape[0] for ten in idx_tensors]
    lengths = torch.tensor(lengths_list)

    inputs = pad_sequence(idx_tensors, padding_value=99999, batch_first=True)

    h = inf_model.init_hidden()
    inputs, lengths = inputs.to(device), lengths.to(device)
    probs, h = inf_model(inputs, lengths, h)

    prob = probs[0]

    if return_prob:
        if is_cuda:
            return prob.detach().cpu().numpy()
        return prob.detach().numpy()

    if is_cuda:
        return prob.argmax().detach().cpu().numpy()
    return prob.argmax().detach().numpy()


def init_csv(fields,csv_path='/content/drive/My Drive/colab_data/grid_search.csv'):
    with open(csv_path,'a') as f:
        writer = csv.writer(f)
        writer.writerow((fields))


def write_result_row(result,csv_path='/content/drive/My Drive/colab_data/grid_search.csv'):
    with open(csv_path,'a') as f:
        writer = csv.writer(f)
        writer.writerow((result))


def eval_model(train_loader,val_loader,test_loader,model,model_version,device,is_cuda,loss_function):
    results = {}

    names = 'train val test'.split()

    loaders = [train_loader, val_loader, test_loader]
    num_batches = 10e3

    val_loss = []
    for name, loader in zip(names, loaders):
        results[name] = {}
        eval_loss = []
        batch_idx = 0
        num_correct = 0
        total = 0
        for inputs, lengths, labels in loader:
            if batch_idx < num_batches:
                model.eval()
                h = model.init_hidden()
                inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)
                probs, h = model(inputs, lengths, h)
                loss = loss_function(probs, labels)
                label = labels[:, 1]

                if is_cuda:
                    pred = probs.argmax(axis=1).cpu().numpy()
                    label = labels[:, 1].cpu().numpy()
                else:
                    pred = probs.argmax(axis=1).detach().numpy()
                    label = labels[:, 1].detach().numpy()

                correct = pred == label
                num_correct += correct.sum()
                total += label.shape[0]

                eval_loss.append(loss.item())

                batch_idx += 1

        mean_loss = round(np.mean(eval_loss), 3)
        acc = round(num_correct * 100 / total, 2)

        # print(f'----{name} set----'.upper())
        # print(f'{name} loss of {mean_loss}')
        # print(f'{name} accuracy of {acc}')

        results[name]['mean_loss'] = mean_loss
        results[name]['acc'] = acc

    save_eval(model_version, results)
    return results


def train_model(model_dir,params,train_loader,val_loader,model,model_version,device,is_cuda,loss_function):
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    save_path = os.path.join(model_dir, 'glove_lstm_batch.pt')
    val_loss_min = np.Inf

    num_epochs = params['num_epochs']
    num_batches = 10e3
    loss_history = []
    last_save_epoch = 0
    decrease_threshold = 3

    for epoch in range(num_epochs):
        train_start = datetime.now()
        batch_idx = 0
        train_loss = []
        model.train()
        for inputs, lengths, labels in train_loader:
            if batch_idx < num_batches:
                model.zero_grad()
                h = model.init_hidden()
                inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)
                probs, h = model(inputs, lengths, h)
                loss = loss_function(probs, labels)

                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

                batch_idx += 1

        train_end = datetime.now()
        diff = (train_end - train_start).seconds
        print(f'{diff} seconds to train epoch')

        batch_idx = 0
        val_loss = []
        for inputs, lengths, labels in val_loader:

            if batch_idx < num_batches:
                model.eval()
                h = model.init_hidden()
                inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)
                probs, h = model(inputs, lengths, h)
                loss = loss_function(probs, labels)

                val_loss.append(loss.item())

                batch_idx += 1

        mean_train_loss = np.mean(train_loss)
        mean_val_loss = np.mean(val_loss)

        loss_history.append((mean_train_loss, mean_val_loss))
        print(f'Epoch {epoch + 1}/{num_epochs}: train loss of {mean_train_loss:.3f}, val loss of {mean_val_loss:.3f}')

        if mean_val_loss <= val_loss_min:
            print(f'Val loss decreased {val_loss_min:.3f} --> {mean_val_loss:.3f} saving model...')
            torch.save(model.state_dict(), save_path)
            val_loss_min = mean_val_loss
            last_save_epoch = epoch

        if (epoch - last_save_epoch) - 1 == decrease_threshold:
            print(f'Val loss has not decreased for {decrease_threshold} epochs...terminating...')
            return loss_history,val_loss_min

        #writer.add_scalars('Loss', {'train': mean_train_loss, 'val': mean_val_loss, }, epoch + 1)

    return loss_history,val_loss_min

def load_model_checkpoint(model_version,is_cuda,model,checkpoint_dir='/content/drive/My Drive/colab_data/model_checkpoints'):
    model_dir = os.path.join(checkpoint_dir,model_version)

    load_path = os.path.join(model_dir, 'glove_lstm_batch.pt')

    if is_cuda:
        model.load_state_dict(torch.load(load_path))
    else:
        model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))

    return model