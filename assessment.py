"""
Code for classification of text articles
"""
import os
import argparse
import random
import re

import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchtext import data
from torchtext.data import Iterator, BucketIterator

BATCH_SIZE = 32
DEFAULT_ARTICLE_FOLDER = "./articles"
SEED = 0
DEFAULT_TRAIN_DEVICE = 'cpu'


class DataFrameDataset(data.Dataset):
    """Class for using pandas DataFrames as a datasource"""
    def __init__(self, examples, fields, filter_pred=None):
        """
        Create a dataset from a pandas dataframe of examples
        """
        self.examples = examples.apply(SeriesExample.fromSeries,
                                       args=(fields, ),
                                       axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]


class SeriesExample(data.Example):
    """Class to convert a pandas Series to an Example"""
    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        for key, field in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                                 "the input data".format(key))
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex


class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var).transpose(0, 1)
            if self.y_vars is not None:
                y = torch.cat([
                    getattr(batch, feat).unsqueeze(1) for feat in self.y_vars
                ],
                              dim=1).float()
            else:
                y = torch.zeros((1))

            yield (x, y)

    def __len__(self):
        return len(self.dl)


class SimpleLSTM(nn.Module):
    def __init__(self, hidden_dim, emb_dim=300, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.encoder = nn.LSTM(emb_dim,
                               hidden_dim,
                               num_layers=1,
                               dropout=dropout)
        self.linear_layers = []
        self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
        self.linear_layers.append(nn.ReLU())
        self.linear_layers.append(nn.Dropout(dropout))
        self.linear_layers.append(nn.Linear(hidden_dim // 2, hidden_dim // 4))
        self.linear_layers.append(nn.ReLU())
        self.linear_layers.append(nn.Dropout(dropout))
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.predictor = nn.Linear(hidden_dim // 4, 5)

    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
        preds = self.predictor(feature)
        return preds


def pre_clean_text(text):
    # replaces unnesessary symbols from text
    return re.sub('[.,\"\'\\\/\n-]', ' ', text)


def train(model, opt, loss_func, train_dl, valid_dl):
    epoch = 0
    while 1:
        epoch += 1
        running_loss = 0.0
        running_corrects = 0
        val_accu = 0.0
        model.train()  # turn on training mode
        for x, y in tqdm.tqdm(train_dl):
            opt.zero_grad()
            preds = model(x)
            loss = loss_func(preds, y)
            loss.backward()
            opt.step()

            preds = 1 / (1 + torch.exp(-preds))
            val_accu += torch.max(y, 1)[1].eq(torch.max(preds,
                                                        1)[1]).sum().item()

            running_loss += loss.data * x.size(0)

        epoch_loss = running_loss / len(train_data)
        epoch_accu = val_accu / len(train_data)

        # calculate the validation loss for this epoch
        val_loss = 0.0
        val_accu = 0.0
        model.eval()  # turn on evaluation mode

        for x, y in valid_dl:
            preds = model(x)
            loss = loss_func(preds, y)
            val_loss += loss.data * x.size(0)
            preds = 1 / (1 + torch.exp(-preds))
            val_accu += torch.max(y, 1)[1].eq(torch.max(preds,
                                                        1)[1]).sum().item()

        val_loss /= len(valid_data)
        val_accu /= len(valid_data)
        print(
            f'Epoch: {epoch}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}'
        )
        print(
            f'Epoch: {epoch}, Training Accuracy: {epoch_accu:.4f}, Validation Accuracy: {val_accu:.4f}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        help="Input folder for model to train defaults to './articles'")

    parser.add_argument(
        "-d",
        "--device",
        help="What device to use for traning, defaults to 'cpu', can be 'cuda'"
    )

    args = parser.parse_args()

    if args.input:
        ARTICLE_FOLDER = args.input
    else:
        ARTICLE_FOLDER = DEFAULT_ARTICLE_FOLDER

    if args.device:
        TRAIN_DEVICE = args.device
    else:
        TRAIN_DEVICE = DEFAULT_TRAIN_DEVICE

    torch.manual_seed(SEED)
    device = torch.device(TRAIN_DEVICE)

    try:
        theme_folders = next(os.walk(ARTICLE_FOLDER))[1]  #get only folders
    except StopIteration:
        print(f"No directory found for '{ARTICLE_FOLDER}', exiting")
        raise SystemExit

    print(f"Avalible themes: {theme_folders}")

    df_data = []
    for theme in theme_folders:
        theme_files = next(os.walk(f"{ARTICLE_FOLDER}/{theme}"))[2]
        for theme_file in theme_files:
            with open(f"{ARTICLE_FOLDER}/{theme}/{theme_file}") as file_data:
                try:
                    file_data_read = file_data.read()
                    if len(file_data_read) > 4000:
                        file_data_read = file_data_read[:4000]
                    df_data.append([pre_clean_text(file_data_read), theme])
                except UnicodeDecodeError:
                    print(
                        f"Error in decoding file {theme_file}, in theme {theme}"
                    )

    TEXT = data.Field(sequential=True, tokenize='spacy', batch_first=True)
    THEME = data.LabelField(batch_first=True, use_vocab=False)
    fields = {'text': TEXT}
    for theme in theme_folders:
        fields[theme] = THEME

    print(fields)

    df = pd.DataFrame(df_data, columns=['text', 'theme'])

    df = pd.concat([df, pd.get_dummies(df['theme'])], axis=1).drop(['theme'],
                                                                   axis=1)

    training_data = DataFrameDataset(df, fields)

    train_data, valid_data = training_data.split(
        split_ratio=0.9, random_state=random.seed(SEED))

    #initialize glove embeddings
    TEXT.build_vocab(train_data, max_size=1000, vectors='glove.6B.200d')

    train_iter, val_iter = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_sizes=(BATCH_SIZE, BATCH_SIZE),
        device=device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        repeat=False)

    train_dl = BatchWrapper(train_iter, "text", theme_folders)
    valid_dl = BatchWrapper(val_iter, "text", theme_folders)

    em_sz = 100
    nh = 128
    model = SimpleLSTM(nh, emb_dim=em_sz)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.BCEWithLogitsLoss()

    train(model, opt, loss_func, train_dl, valid_dl)
