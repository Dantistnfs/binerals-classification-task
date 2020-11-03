"""
Various helper functions for loading and processing text data
"""
import os
import argparse
import re
import pandas as pd

import torch
import torchtext


class DataFrameDataset(torchtext.data.Dataset):
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


class SeriesExample(torchtext.data.Example):
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


def pre_clean_text(text):
    """Replaces unnesessary symbols from text """
    return re.sub(r"[.,\"'\\\/\n-]", ' ', text)


def get_article_themes(article_folder):
    try:
        theme_folders = next(os.walk(article_folder))[1]  #get only folders
    except StopIteration as exception:
        print(f"No directory found for '{article_folder}', exiting")
        raise exception
    print(f"Avalible themes: {theme_folders}")
    return theme_folders


def get_articles(article_folder, theme_folders):
    df_data = []
    for theme in theme_folders:
        theme_files = next(os.walk(f"{article_folder}/{theme}"))[2]
        for theme_file in theme_files:
            with open(f"{article_folder}/{theme}/{theme_file}") as file_data:
                try:
                    file_data_read = file_data.read()
                    if len(file_data_read) > 4000:
                        file_data_read = file_data_read[:4000]
                    df_data.append([pre_clean_text(file_data_read), theme])
                except UnicodeDecodeError:
                    print(
                        f"Error in decoding file {theme_file}, in theme {theme}"
                    )

    text_df = pd.DataFrame(df_data, columns=['text', 'theme'])

    for theme in theme_folders:  # this forces certain order of columns for one-hot encoding
        text_df.loc[text_df['theme'] == theme, theme] = 1

    text_df = text_df.drop('theme', axis=1)
    text_df = text_df.fillna(0)
    return text_df


def parse_args():
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
    return parser.parse_args()
