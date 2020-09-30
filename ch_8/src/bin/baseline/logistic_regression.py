# Baseline logistic regression model
import argparse
import pathlib
import time
import sys

import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize

from sklearn import linear_model
from sklearn import model_selection
from sklearn.feature_extraction import text
from sklearn import metrics
from sklearn import pipeline

sys.path.append("./../../../")
from src.utils import io

def main(config):
    """
        Function to run the training process
        :param config: SimpleNamespace config object
    """
    train_df = pd.read_csv(config.train_filepath, usecols=config.data["columns"])
    train_df.rename(columns = config.data["names"], inplace = True)

    test_df = pd.read_csv(config.test_filepath, usecols=config.data["columns"])
    test_df.rename(columns = config.data["names"], inplace = True)

    print (f"Train labels distribution:\n{train_df.loc[:,'labels'].value_counts()}")
    print (f"Test labels distribution:\n{test_df.loc[:,'labels'].value_counts()}\n")

    # Create validation folds
    train_df["fold"] = -1
    skf = model_selection.StratifiedKFold(n_splits=config.num_folds)
    for fold, (train_index, val_index) in enumerate(skf.split(train_df, train_df.loc[:, "labels"])):
        train_df.loc[val_index, "fold"] = fold

    # Iterate folds and run train_mdoels
    model_list = []
    folds = np.unique(train_df.loc[:, "fold"].values)
    
    print ("\n~~~~Running training and cross validation~~~~\n")
    for fold in folds:
        start_time = time.perf_counter()

        train_fold = train_df.loc[train_df.loc[:,"fold"]!=fold,:].reset_index(drop=True)
        val_fold = train_df.loc[train_df.loc[:,"fold"]==fold,:].reset_index(drop=True)

        text_encoder = text.TfidfVectorizer(tokenizer = word_tokenize)
        clf = linear_model.LogisticRegression(max_iter=config.max_iter, n_jobs= config.n_jobs)

        model = pipeline.Pipeline([('text_enc', text_encoder), ('clf', clf)])
        model.fit(train_fold.loc[:, "feature"].values, train_fold.loc[:, "labels"].values)

        # Evaluate score
        val_probs = model.predict_proba(val_fold.loc[:, "feature"].values)
        roc_auc_score = metrics.roc_auc_score(val_fold.loc[:, "labels"].values, val_probs[:, 1])
        # Calculate accuracy
        val_tags = model.predict(val_fold.loc[:, "feature"].values)
        accuracy_score = metrics.accuracy_score(val_fold.loc[:, "labels"].values, val_tags)

        # Append in the list to evaluate on test
        model_list.append(model)

        print (f"fold: {fold}, roc auc socre: {roc_auc_score:.2f}, accuracy_score: {accuracy_score:.2f}, fold time: {(time.perf_counter()-start_time):.2f} seconds")
    
    # Run evaluation on tests
    print ("\n~~~~Running evaluation on held out test data~~~~\n")
    running_score, running_accuracy = 0, 0
    for fold, model in enumerate(model_list):
        test_text = test_df.loc[:, "feature"].values
        test_class = test_df.loc[:, "labels"].values

        label_probs = model.predict_proba(test_text)[:, 1]
        label_tags = model.predict(test_text)

        # Calculate accuracy
        accuracy_score = metrics.accuracy_score(test_class, label_tags)
        running_accuracy += accuracy_score

        roc_auc_score = metrics.roc_auc_score(test_class, label_probs)
        running_score += roc_auc_score

        print (f"FOLD: {fold}, auc_roc_score on held out test: {roc_auc_score}, accuracy: {accuracy_score}")
        
    print (f"Average auc_roc_score acorss all folds on test data is: {running_score/len(model_list)}")
    print (f"Average accuracy acorss all folds on test data is: {running_accuracy/len(model_list)}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train a baseline logistic regression model")
    parser.add_argument("config_filepath", metavar="fc", type=pathlib.Path, help="Configuration filepath")
    args = parser.parse_args()

    config = io.load_config(args.config_filepath)
    main(config)