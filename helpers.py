"""
Various helper functions for loading and processing text data
"""
import os
import pandas as pd

from torchtext import data


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


def get_article_themes(article_folder):
    try:
        theme_folders = next(os.walk(ARTICLE_FOLDER))[1]  #get only folders
    except StopIteration:
        print(f"No directory found for '{ARTICLE_FOLDER}', exiting")
        raise SystemExit
    print(f"Avalible themes: {theme_folders}")
    return theme_folders


def get_articles(article_folder, theme_folders):
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

    df = pd.DataFrame(df_data, columns=['text', 'theme'])
    df = pd.concat([df, pd.get_dummies(df['theme'])], axis=1).drop(['theme'],
                                                                   axis=1)

    return df
