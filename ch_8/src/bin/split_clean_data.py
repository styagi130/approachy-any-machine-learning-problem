import re
import string
import argparse
import pathlib
import sys

import pandas as pd
import numpy as np
from sklearn import model_selection

sys.path.append("./../../")
from src.utils import io


def clean_fn(text, config):
    """
        Function to remove all the punctuation and digits to get clean data
        :param text: Takes text to clean as parameter
        :param config: SimpleNamespace configuration object
        :return: Clean text
    """
    punctuations = set(string.punctuation)
    punctuation_to_remove = punctuations - set(config.data["allowed_punct"])
    trans_table = str.maketrans("", "", "".join(punctuation_to_remove))

    ## Remove speaker Tags:
    no_spk_tag = re.sub(r"<#.*>", "", text)
    merge_space = re.sub(r"\s+", " ", no_spk_tag)

    # Remove punctuations
    no_punct = merge_space.translate(trans_table)
    return no_punct 


def split_clean(config):
    """
        Function to clean and split data into test and trainsets
    """
    df = pd.read_csv(config.total_datapath, usecols = config.data["columns"])

    # Clean data
    df.loc[:,"text"] = df.loc[:,"text"].apply(clean_fn, args=(config, ))

    # Split data
    df_train, df_test = model_selection.train_test_split(df, test_size= config.data["test_size"], stratify = df.loc[:, "tagInt"].values)


    # print stats
    print("~~~~~~~~~~~~Data stats are~~~~~~~~~~~~~")
    print (f"TOTAL data is: {df.shape}")
    print (df.loc[:, "tagInt"].value_counts())

    print("~~~~~~~~~~~~test stats are~~~~~~~~~~~~~")
    print (f"TOTAL data is: {df_test.shape}")
    print (df_test.loc[:, "tagInt"].value_counts())

    if "train_fold_optimization" in vars(config):
        df_train, df_train_fold = model_selection.train_test_split(df_train, test_size= config.data["fold_size"], stratify = df_train.loc[:, "tagInt"].values)
        print("~~~~~~~~~~~~train folds data stats are~~~~~~~~~~~~~")
        print (f"TOTAL data is: {df_train_fold.shape}")
        print (df_train_fold.loc[:, "tagInt"].value_counts())



    print("~~~~~~~~~~~~train stats are~~~~~~~~~~~~~")
    print (f"TOTAL data is: {df_train.shape}")
    print (df_train.loc[:, "tagInt"].value_counts())

    # Save dataframes
    df_train.to_csv(config.train_filepath, index=None, columns=config.data["columns"])
    df_test.to_csv(config.test_filepath, index=None, columns=config.data["columns"])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = "Split and clean data, i.e remove special symbols etc")
    parser.add_argument("config_filepath", metavar="fc", type=pathlib.Path, help="Configuration filepath")
    args = parser.parse_args()

    global config
    config = io.load_config(args.config_filepath)
    split_clean(config)