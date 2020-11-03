"""
Code for classification of text articles
"""

import torch
import pandas as pd
from torchtext import data

import helpers
import model_helpers

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
    device = torch.device(TRAIN_DEVICE)

    theme_folders = helpers.get_article_themes(ARTICLE_FOLDER)
    theme_folders = ['business', 'sport', 'entertainment', 'tech', 'politics']
    print(
        "THEME FOLDERS IS IN FIXED ORDER, IN PRODUCTION IT SHOULD BE SORTED OR FIXED IN CONFIGURATION FILE"
    )

    TEXT = data.Field(sequential=True, tokenize='spacy', batch_first=True)
    THEME = data.LabelField(batch_first=True, use_vocab=False)
    fields = {'text': TEXT}
    for theme in theme_folders:
        fields[theme] = THEME

    df = helpers.get_articles(ARTICLE_FOLDER, theme_folders)

    test_data = helpers.DataFrameDataset(df, fields)

    vocab = model_helpers.load_vocab("trained_vocab.vcb")
    TEXT.vocab = vocab

    test_iter = data.Iterator(test_data,
                              batch_size=BATCH_SIZE,
                              device=device,
                              sort=False,
                              train=False,
                              sort_within_batch=False,
                              repeat=False)

    test_dl = helpers.BatchWrapper(test_iter, "text", theme_folders)

    from simplelstm import SimpleLSTM  # needed for correct init of model
    model = model_helpers.load_model("trained_model.pth",
                                     map_location=torch.device('cpu'))

    predictions = model_helpers.predict(model, test_dl)

    pd.concat([df, pd.DataFrame(predictions, columns=theme_folders)],
              axis=1).to_csv("predictions.csv")
