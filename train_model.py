"""
Code for classification of text articles
"""
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data

import helpers
import model_helpers
import simplelstm

BATCH_SIZE = 32
DEFAULT_ARTICLE_FOLDER = "./articles"
SEED = 0
DEFAULT_TRAIN_DEVICE = 'cpu'

if __name__ == "__main__":
    args = helpers.parse_args()

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

    model = simplelstm.SimpleLSTM(vocab_size=len(TEXT.vocab),
                                  hidden_dim=128,
                                  emb_dim=100)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.BCEWithLogitsLoss()

    model_helpers.train(model,
                        optimizer=opt,
                        loss_function=loss_func,
                        train_dataset=train_dl,
                        validation_dataset=valid_dl)

    model_helpers.save_model(model, "trained_model.pth")
    model_helpers.save_vocab(TEXT.vocab, "trained_vocab.vcb")
