# Use this script to create folds
import argparse
import pathlib
import sys

import pandas as pd
import numpy

from sklearn import model_selection

sys.path.append("./../../")
from src.utils import io

def create_folds(config):
    """
        Function to create folds and persist on disc
        :param config: A SimpleNamespace config object
    """
    train_df = pd.read_csv(config.train_filepath, usecols=config.data["columns"])
    train_df.loc[:, "fold"] = -1

    skf = model_selection.StratifiedKFold(n_splits=config.num_folds)

    for fold,(train_index, val_index) in enumerate(skf.split(train_df, train_df.loc[:,"tagInt"])):
        train_df.loc[val_index, "fold"] = fold

    train_df.to_csv(config.train_fold_filepath, index=None)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = "Use this to create a stratified k-folded dataframe")
    parser.add_argument("config_filepath", metavar="fc", type=pathlib.Path, help="config filepath")
    args = parser.parse_args()

    config = io.load_config(args.config_filepath)
    create_folds(config)
