"""
Code for classification of text articles
"""
import argparse
import random
import re

import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchtext import data

import helpers

BATCH_SIZE = 32
DEFAULT_ARTICLE_FOLDER = "./articles"
SEED = 0
DEFAULT_TRAIN_DEVICE = 'cpu'


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

    theme_folders = helpers.get_article_themes(ARTICLE_FOLDER)

    TEXT = data.Field(sequential=True, tokenize='spacy', batch_first=True)
    THEME = data.LabelField(batch_first=True, use_vocab=False)
    fields = {'text': TEXT}
    for theme in theme_folders:
        fields[theme] = THEME

    print(fields)

    df = helpers.get_articles(ARTICLE_FOLDER, theme_folders)

    training_data = helpers.DataFrameDataset(df, fields)

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

    train_dl = helpers.BatchWrapper(train_iter, "text", theme_folders)
    valid_dl = helpers.BatchWrapper(val_iter, "text", theme_folders)

    em_sz = 100
    nh = 128
    model = SimpleLSTM(nh, emb_dim=em_sz)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.BCEWithLogitsLoss()

    train(model, opt, loss_func, train_dl, valid_dl)
